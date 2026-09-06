/*
 * SPDX-License-Identifier: MIT
 *
 * kbase_probe.h — single entry point for opening a /dev/mali* kbase node.
 *
 * The kbase kernel driver ships in two incompatible flavours multiplexed on
 * the *same* ioctl numbers: Job Manager (JM, arch <= 9) and Command Stream
 * Frontend (CSF, arch >= 10). Sending a CSF ioctl (KBASE_IOCTL_CS_*) to a JM
 * kernel, or a JM ioctl to a CSF kernel, does not "mostly work" — it talks
 * past a driver that doesn't implement that request. So the *first* thing
 * that has to happen on a candidate node is the version handshake that tells
 * the two apart; every later call is only meaningful once that's settled.
 *
 * kbase_probe_open() does exactly that up front and returns which of the two
 * families was found, so the caller can branch once, immediately, on a
 * `switch (kind)` and never reach for a CSF-only or JM-only function on the
 * wrong device. Everything downstream (kbase_kmod.c for CSF, kbase_jm.c for
 * JM) still independently refuses to run its own ioctls on the wrong device
 * kind — kbase_probe is the layer that lets a caller avoid hitting that
 * refusal in the first place instead of just handling it.
 */

#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct pan_kmod_dev;
struct pan_kmod_allocator;

enum kbase_probe_kind {
   /* Node did not answer either kbase version handshake (wrong driver,
    * permissions, or not a kbase node at all). */
   KBASE_PROBE_NONE = 0,
   KBASE_PROBE_JM,
   KBASE_PROBE_CSF,
};

/* Does the version handshake ONLY (KBASE_IOCTL_VERSION_CHECK_CSF, then
 * _JM) on an already-open fd and reports which flavour answered, without
 * touching SET_FLAGS, mmap(), or any other state. Safe to call speculatively
 * against any device node before deciding whether to hand it to
 * kbase_probe_open(); does not consume or invalidate the fd. */
enum kbase_probe_kind kbase_probe_fd_kind(int fd);

/* Full result of kbase_probe_open(): which family was found, plus the
 * fully-initialised pan_kmod_dev for it (NULL on failure). `kind` is
 * KBASE_PROBE_NONE and `dev` is NULL if the node isn't a kbase node at
 * all; `kind` is set but `dev` is NULL if the handshake succeeded and the
 * flavour was identified, but device creation failed afterwards (bad GPU
 * properties, unsupported legacy JM version, OOM, ...). */
struct kbase_probe_result {
   enum kbase_probe_kind kind;
   struct pan_kmod_dev *dev;
};

/* Opens `path` (e.g. "/dev/mali0"), classifies it as CSF or JM with a single
 * up-front handshake, and only then finishes device init down the matching
 * path — kbase_kmod_dev_create() internally, which itself never issues a
 * JM-only or CSF-only ioctl before that same classification has happened.
 * The fd is owned by the returned pan_kmod_dev (PAN_KMOD_DEV_FLAG_OWNS_FD)
 * on success, and closed by this function on failure.
 *
 * Typical caller shape, mirroring exactly what the caller must not do
 * (mixing CSF and JM calls on one handle):
 *
 *   struct kbase_probe_result r = kbase_probe_open("/dev/mali0", alloc);
 *   switch (r.kind) {
 *   case KBASE_PROBE_CSF:
 *      // only kbase_kmod_csf_*() / kbase_kmod_get_csif_props() from here.
 *      break;
 *   case KBASE_PROBE_JM:
 *      // only kbase_jm_*() from here.
 *      break;
 *   case KBASE_PROBE_NONE:
 *      // not a kbase node; try the next candidate.
 *      break;
 *   }
 */
struct kbase_probe_result
kbase_probe_open(const char *path, const struct pan_kmod_allocator *allocator);

#if defined(__cplusplus)
}
#endif
