/*
 * Copyright © 2021 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_KBASE_FD_H
#define PANVK_KBASE_FD_H

#include "panvk_device.h"

/* Both panvk_vX_gpu_queue.c (KBASE_IOCTL_JOB_SUBMIT + the read()/poll()
 * event stream) and panvk_vX_event.c (KBASE_IOCTL_SOFT_EVENT_UPDATE) need
 * a plain POSIX fd for the open /dev/mali* character device to call
 * ioctl()/read()/poll() on directly.
 *
 * This is exactly what the common Vulkan runtime device object already
 * carries as `vk_device::drm_fd` -- panvk sets this to the fd used to
 * open the kernel device at VkDevice creation time (see the "panvk: Use
 * vk_device::drm_fd instead of going back to the physical device" commit
 * in Mesa's history), regardless of whether that fd genuinely belongs to
 * a DRM node or, as here, a kbase character device. There's no need to
 * reach into pan_kmod's backend-private struct for this.
 *
 * If this specific fork renamed or removed that field, or stores the
 * kbase fd somewhere else on panvk_device instead, swap the one line
 * below accordingly -- everything else in the gpu_queue/event rewrite is
 * independent of exactly where this fd comes from.
 */
static inline int
panvk_kbase_raw_fd(struct panvk_device *dev)
{
   return dev->vk.drm_fd;
}

#endif /* PANVK_KBASE_FD_H */
