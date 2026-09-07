#ifndef PANVK_KBASE_FD_H
#define PANVK_KBASE_FD_H

#include "panvk_device.h"

/* IMPORTANT: this must return the *same* fd that was used for the
 * kbase version-check + SET_FLAGS handshake in kbase_kmod_dev_create()
 * (see kbase_kmod.c) -- that handshake is what moves the kernel-side
 * kbase_file for this fd from KBASE_FILE_NEED_VSN to KBASE_FILE_COMPLETE
 * and creates/enables its kbase_context.
 *
 * Opening a fresh fd on /dev/mali0 here (as this used to do) gets a
 * brand new, never-handshaked kbase_file: every ioctl on it other than
 * the version check fails with -EPERM, which is exactly the
 * "KBASE_IOCTL_JOB_SUBMIT failed: Operation not permitted" seen at the
 * first submit, even though memory allocation on the real device fd
 * (dev->kmod.dev->fd) works fine.
 */
static inline int
panvk_kbase_raw_fd(struct panvk_device *dev)
{
   return dev->kmod.dev->fd;
}

#endif /* PANVK_KBASE_FD_H */
