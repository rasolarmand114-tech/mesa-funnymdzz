/*
 * Copyright © 2021 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <string.h>

#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_event.h"
#include "panvk_priv_bo.h"

#include "util/log.h"

#include "vk_log.h"

#include "../../lib/kmod/kbase_jm.h"

/* The soft-event ioctl only ever touches a single status byte, but we give
 * it a whole tiny BO of its own (rather than suballocating out of some
 * shared pool) so panvk_priv_bo_flush()/_invalidate() on it can't stomp on,
 * or be stomped on by, unrelated data sharing the same cacheline. */
#define PANVK_EVENT_BO_SIZE 64

static inline volatile uint8_t *
panvk_event_status_ptr(const struct panvk_event *event)
{
   return (volatile uint8_t *)event->bo->addr.host;
}

bool
panvk_per_arch(event_is_set)(const struct panvk_event *event)
{
   /* The byte we're reading was last written by the kernel (in response to
    * kbase_jm_soft_event_update()), not by us, so make sure we're not
    * looking at a stale CPU-cached copy of it. */
   panvk_priv_bo_invalidate(event->bo, 0, PANVK_EVENT_BO_SIZE);

   return *panvk_event_status_ptr(event) == (uint8_t)KBASE_JM_SOFT_EVENT_SET;
}

bool
panvk_per_arch(event_update)(struct panvk_device *dev,
                             struct panvk_event *event,
                             enum kbase_jm_soft_event_status status)
{
   if (kbase_jm_soft_event_update(dev->kmod.dev, event->bo->addr.dev,
                                  status)) {
      mesa_loge("panvk: kbase_jm_soft_event_update() failed: %s",
                strerror(errno));
      return false;
   }

   return true;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(CreateEvent)(VkDevice _device,
                            const VkEventCreateInfo *pCreateInfo,
                            const VkAllocationCallbacks *pAllocator,
                            VkEvent *pEvent)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   struct panvk_event *event = vk_object_zalloc(
      &device->vk, pAllocator, sizeof(*event), VK_OBJECT_TYPE_EVENT);
   if (!event)
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result =
      panvk_priv_bo_create(device, PANVK_EVENT_BO_SIZE, 0,
                           VK_SYSTEM_ALLOCATION_SCOPE_DEVICE, &event->bo);
   if (result != VK_SUCCESS) {
      vk_object_free(&device->vk, pAllocator, event);
      return result;
   }

   /* Start out RESET, same initial state a freshly created (non-SIGNALED)
    * DRM syncobj used to have. */
   if (!panvk_per_arch(event_update)(device, event,
                                     KBASE_JM_SOFT_EVENT_RESET)) {
      panvk_priv_bo_unref(event->bo);
      vk_object_free(&device->vk, pAllocator, event);
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   *pEvent = panvk_event_to_handle(event);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(DestroyEvent)(VkDevice _device, VkEvent _event,
                             const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   VK_FROM_HANDLE(panvk_event, event, _event);

   if (!event)
      return;

   panvk_priv_bo_unref(event->bo);
   vk_object_free(&device->vk, pAllocator, event);
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(GetEventStatus)(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(panvk_event, event, _event);

   return panvk_per_arch(event_is_set)(event) ? VK_EVENT_SET : VK_EVENT_RESET;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(SetEvent)(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   VK_FROM_HANDLE(panvk_event, event, _event);

   if (!panvk_per_arch(event_update)(device, event, KBASE_JM_SOFT_EVENT_SET))
      return VK_ERROR_DEVICE_LOST;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(ResetEvent)(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   VK_FROM_HANDLE(panvk_event, event, _event);

   if (!panvk_per_arch(event_update)(device, event,
                                     KBASE_JM_SOFT_EVENT_RESET))
      return VK_ERROR_DEVICE_LOST;

   return VK_SUCCESS;
}
