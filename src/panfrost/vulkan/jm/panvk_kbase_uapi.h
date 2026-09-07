/*
 * Copyright © 2021 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_KBASE_UAPI_H
#define PANVK_KBASE_UAPI_H

/* This header intentionally does NOT redefine any kbase ioctl struct or
 * flag by hand. Those layouts (struct base_jd_atom_v2, BASE_JD_REQ_*,
 * BASE_JD_DEP_TYPE_*, struct base_jd_event_v2, ...) have shifted field
 * names/types across kbase DDK releases (e.g. compat_core_req vs
 * core_req), so hand-copying them here risks a silent ABI mismatch with
 * whatever kernel this is actually run against -- which, for ioctl
 * payloads, means kernel memory corruption, not just a compile error.
 *
 * Instead: vendor the two real uapi headers verbatim from the exact
 * kernel commit you're targeting, e.g.
 *
 *   https://chromium.googlesource.com/chromiumos/third_party/kernel/+/
 *   e63c6fe1315027344dde9c0e2e857ed648499211/drivers/gpu/arm/mali/
 *
 * into this directory:
 *
 *   src/panfrost/lib/kbase_uapi/mali_kbase_ioctl.h
 *   src/panfrost/lib/kbase_uapi/mali_base_kernel.h   (path/name may
 *     differ in this specific tree -- it wasn't listed next to
 *     mali_kbase_ioctl.h in the directory this file's comments were
 *     written against, so track it down via gitiles and drop it in
 *     unmodified)
 *
 * and this file just pulls them in. This is also exactly what
 * proprietary kbase userspace clients (e.g. libmali/gralloc) do: build
 * against a vendored copy of the kernel's own uapi headers rather than
 * reimplementing the ABI.
 */

#include "../../lib/kmod/mali_kbase_ioctl.h"
#include "../../lib/kmod/mali_base_kernel.h"

#endif /* PANVK_KBASE_UAPI_H */
