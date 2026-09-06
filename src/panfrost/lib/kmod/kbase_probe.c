/*
 * SPDX-License-Identifier: MIT
 *
 * kbase_probe.c — single entry point for opening a /dev/mali* kbase node.
 * See kbase_probe.h for why this needs to exist as its own step instead of
 * being folded into kbase_kmod_dev_create().
 */

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "util/log.h"

#include "mali_kbase_ioctl.h"

#include "kbase_probe.h"
#include "kbase_jm.h"
#include "pan_kmod.h"

enum kbase_probe_kind
kbase_probe_fd_kind(int fd)
{
   /* Same handshake kbase_kmod_dev_create() performs, run here purely for
    * classification. Each flavour's kernel driver rejects the other
    * flavour's VERSION_CHECK ioctl number with -EPERM, which is what makes
    * probing reliable; the handshake itself is idempotent; running it here
    * and again inside kbase_kmod_dev_create() a moment later is safe. */
   struct kbase_ioctl_version_check ver = { 0 };

   if (ioctl(fd, KBASE_IOCTL_VERSION_CHECK_CSF, &ver) == 0)
      return KBASE_PROBE_CSF;

   if (ioctl(fd, KBASE_IOCTL_VERSION_CHECK_JM, &ver) == 0)
      return KBASE_PROBE_JM;

   return KBASE_PROBE_NONE;
}

struct kbase_probe_result
kbase_probe_open(const char *path, const struct pan_kmod_allocator *allocator)
{
   struct kbase_probe_result result = {
      .kind = KBASE_PROBE_NONE,
      .dev = NULL,
   };

   int fd = open(path, O_RDWR | O_CLOEXEC);
   if (fd < 0) {
      mesa_logd("kbase_probe: failed to open %s: %s", path, strerror(errno));
      return result;
   }

   /* Classify before doing anything else. This is the one branch point:
    * from here down, CSF and JM never share a code path again. Getting this
    * wrong (e.g. skipping straight to a JM- or CSF-specific call because
    * "it's probably a recent chip") is exactly the bug class this file
    * exists to make structurally impossible for callers of this API. */
   enum kbase_probe_kind kind = kbase_probe_fd_kind(fd);

   if (kind == KBASE_PROBE_NONE) {
      mesa_logd("kbase_probe: %s does not answer either kbase version "
                "handshake", path);
      close(fd);
      return result;
   }

   /* pan_kmod_dev_create_with_driver()/kbase_kmod_dev_create() redo the
    * handshake internally and dispatch to the matching init path on their
    * own (CSF: CS_GET_GLB_IFACE + USER register page; JM: nothing extra).
    * They take ownership of the fd on success and close it themselves on
    * failure, so there is nothing left for us to clean up either way. */
   struct pan_kmod_dev *dev = pan_kmod_dev_create_with_driver(
      fd, PAN_KMOD_DEV_FLAG_OWNS_FD, "kbase", NULL, allocator);

   result.kind = kind;
   result.dev = dev;

   if (!dev) {
      mesa_loge("kbase_probe: %s identified as %s but device init failed",
                path, kind == KBASE_PROBE_CSF ? "CSF" : "JM");
      return result;
   }

   /* Cross-check: what the device itself reports post-init must agree with
    * what the handshake said up front. A mismatch here would mean the two
    * classification paths have drifted apart, which is worse than either
    * being wrong on its own — so treat it as a hard failure rather than
    * silently trusting one side. */
   enum kbase_gfx_dev_kind confirmed = kbase_gfx_dev_kind(dev);
   bool agrees = (kind == KBASE_PROBE_CSF && confirmed == KBASE_GFX_DEV_CSF) ||
                 (kind == KBASE_PROBE_JM && confirmed == KBASE_GFX_DEV_JM);

   if (!agrees) {
      mesa_loge("kbase_probe: %s: pre-init handshake (%s) disagrees with "
                "post-init device kind — refusing to hand back a device",
                path, kind == KBASE_PROBE_CSF ? "CSF" : "JM");
      pan_kmod_dev_destroy(dev);
      result.dev = NULL;
      return result;
   }

   mesa_logi("kbase_probe: %s opened as %s", path,
             kind == KBASE_PROBE_CSF ? "CSF" : "JM");

   return result;
}
