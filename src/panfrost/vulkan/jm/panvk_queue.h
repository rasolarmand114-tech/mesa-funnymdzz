/*
 * Copyright © 2021 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_QUEUE_H
#define PANVK_QUEUE_H

#ifndef PAN_ARCH
#error "PAN_ARCH must be defined"
#endif

#include <stdint.h>

#include "panvk_device.h"

#include "vk_queue.h"

struct panvk_gpu_queue {
   struct vk_queue vk;

   /* HW atom id of the most recently submitted kbase JM atom on this queue
    * (0 == none submitted yet). Passed as
    * kbase_jm_atom_desc::depends_on_atom for the next atom we submit, so
    * completion order always matches submission order.
    *
    * kbase_jm_atom_desc only supports a single explicit predecessor and
    * completion is only ever reported asynchronously via a poll()+read()
    * event stream on the device fd (kbase_jm_wait_event()) -- there is no
    * DRM syncobj/fence to hook external waiters into. Rather than run a
    * background thread to demultiplex that event stream, this queue uses a
    * fully synchronous submission model: panvk_per_arch(gpu_queue_submit)()
    * blocks until every atom it submits has completed before returning.
    * That keeps this struct (and QueueWaitIdle) trivial, at the cost of not
    * overlapping consecutive vkQueueSubmit()s on this queue. If that
    * overlap ever matters, this is the field to grow into a proper
    * in-flight tracker.
    *
    * This also assumes a single panvk_gpu_queue per VkDevice: Mali JM
    * hardware only exposes one job-manager submission context per kbase
    * fd, matching the existing "no queue priorities" limitation in
    * panvk_per_arch(create_gpu_queue)().
    */
   uint8_t jm_last_atom;
};

VK_DEFINE_HANDLE_CASTS(panvk_gpu_queue, vk.base, VkQueue, VK_OBJECT_TYPE_QUEUE)

VkResult panvk_per_arch(create_gpu_queue)(
   struct panvk_device *device, const VkDeviceQueueCreateInfo *create_info,
   uint32_t queue_idx, struct vk_queue **out_queue);
void panvk_per_arch(destroy_gpu_queue)(struct vk_queue *vk_queue);
VkResult panvk_per_arch(gpu_queue_submit)(struct vk_queue *vk_queue,
                                          struct vk_queue_submit *vk_submit);
VkResult panvk_per_arch(gpu_queue_check_status)(struct vk_queue *vk_queue);

#endif
