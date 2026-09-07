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
 * ioctl()/read()/poll() on directly, bypassing the fictional kbase_jm.h
 * wrapper this file replaces.
 *
 * dev->kmod.dev is a `struct pan_kmod_dev *` -- Mesa's kernel-agnostic
 * abstraction over panfrost/panthor/kbase backends. Its concrete layout
 * (and whether/where it stashes the raw fd, e.g. behind
 * pan_kmod_dev_fd() or a kbase-backend-private subclass) lives in
 * src/panfrost/lib/kmod/, which wasn't part of the files provided here,
 * so I can't fill this in without guessing at a fork-specific internal
 * API and risking silently reading/casting the wrong field.
 *
 * Replace the body below with whatever this tree's pan_kmod kbase
 * backend actually exposes -- most likely either a public accessor
 * (pan_kmod_dev_fd(dev->kmod.dev)) or, if the kbase backend keeps its
 * own fd separately from generic pan_kmod, a kbase-specific field on
 * dev itself (something like dev->kmod.kbase.fd).
 */
static inline int
panvk_kbase_raw_fd(struct panvk_device *dev)
{
#error \
   "Fill in panvk_kbase_raw_fd() using this tree's pan_kmod/kbase " \
   "backend accessor for the raw device fd (see comment above)."
   return -1;
}

#endif /* PANVK_KBASE_FD_H */
