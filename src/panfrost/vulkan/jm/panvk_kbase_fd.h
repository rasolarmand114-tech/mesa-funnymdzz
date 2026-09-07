/*
 * Copyright © 2021 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_KBASE_FD_H
#define PANVK_KBASE_FD_H

#include "panvk_device.h"

/*
 * KBase backend uses a plain POSIX fd for the Mali character device.
 *
 * This fd is intentionally independent from:
 *
 *   struct vk_device::drm_fd
 *
 * No DRM fd is required here.
 */
static inline int
panvk_kbase_raw_fd(struct panvk_device *dev)
{
   return dev->kbase_fd;
}

#endif /* PANVK_KBASE_FD_H */
