/*
 * Copyright © 2021 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_EVENT_H
#define PANVK_EVENT_H

#include <stdbool.h>
#include <stdint.h>

#include "vk_object.h"

#include "kbase_jm.h"

struct panvk_priv_bo;
struct panvk_device;

/* A VkEvent is backed by a kbase JM "soft-event" byte
 * (BASE_JD_SOFT_EVENT_SET / BASE_JD_SOFT_EVENT_RESET) living inside a tiny
 * priv BO. Updates go through kbase_jm_soft_event_update() so the kernel's
 * own soft-event state stays authoritative -- we never just poke the
 * mapping from the host -- which is what makes this "GPU-visible" rather
 * than a plain CPU-only flag.
 */
struct panvk_event {
   struct vk_object_base base;
   struct panvk_priv_bo *bo;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(panvk_event, base, VkEvent, VK_OBJECT_TYPE_EVENT)

/* True if the event is currently observed as SET from the host. Used by the
 * gpu queue to implement vkCmdWaitEvents2() as a host-side spin: kbase_jm.h
 * doesn't expose a GPU-job form of a soft-event wait that could be chained
 * into a job chain, only this host-visible byte. */
bool panvk_per_arch(event_is_set)(const struct panvk_event *event);

/* Push a new status to the event's soft-event byte through the kernel
 * (KBASE_JM_SOFT_EVENT_SET or KBASE_JM_SOFT_EVENT_RESET). Returns false on
 * ioctl failure, in which case the caller should treat the device as
 * lost. */
bool panvk_per_arch(event_update)(struct panvk_device *dev,
                                  struct panvk_event *event,
                                  enum kbase_jm_soft_event_status status);

#endif
