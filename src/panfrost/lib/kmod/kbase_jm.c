/*
 * SPDX-License-Identifier: MIT
 *
 * kbase_jm.c — Job Manager (JM) command-submission backend for the ARM Mali
 * kbase kmod driver (/dev/mali*).
 *
 * WHAT THIS FILE IS
 * ------------------
 * kbase_kmod.c already implements device enumeration, memory management,
 * and the CSF (arch >= 10) queue-group / queue / tiler-heap submission path
 * (kbase_kmod_csf_group_create / _queue_bind / _queue_kick / _wait_event),
 * but explicitly leaves JM (arch <= 9) command submission unimplemented
 * (see the header comment in kbase_kmod.c: "Command submission (CSF queue
 * groups / JM job atoms) is not wired up yet").
 *
 * This file fills in the missing half: JM job-atom submission, job-slot
 * (JS_PRESENT / JS_FEATURES_n register) introspection, and JM completion-
 * event handling.
 *
 * WHAT THIS FILE IS NOT
 * ----------------------
 * - It does not modify kbase_kmod.c or kbase_kmod.h.
 * - It does not read, call, or re-implement any CSF ioctl, register, or
 *   data structure. CSF submission keeps going exclusively through
 *   kbase_kmod_csf_* in kbase_kmod.c.
 * - The only thing shared between the two paths is the device handle
 *   (struct pan_kmod_dev *) and the read-only kbase_gfx_dev_kind() probe
 *   below, which every public function in this file calls first to refuse
 *   to touch a CSF device.
 *
 * ABI note
 * --------
 * Built against the real jm/mali_kbase_jm_ioctl.h (uAPI 11.46). Two extra
 * ioctls from that header are used besides KBASE_IOCTL_JOB_SUBMIT:
 *   - KBASE_IOCTL_SOFT_EVENT_UPDATE (nr 28): CPU-side set/reset of a
 *     BASE_JD_REQ_SOFT_EVENT_WAIT/SET/RESET soft-job's event, exposed here
 *     as kbase_jm_soft_event_update().
 *   - KBASE_IOCTL_POST_TERM (nr 4): wakes up a blocked reader on this fd at
 *     context teardown, exposed here as kbase_jm_post_term().
 * Note that this header's KBASE_IOCTL_VERSION_CHECK (nr 0) is what
 * kbase_kmod.c calls KBASE_IOCTL_VERSION_CHECK_JM, and its
 * KBASE_IOCTL_VERSION_CHECK_RESERVED (nr 52) is deliberately the same
 * ioctl number the CSF header uses for its *real* version-check request —
 * that collision is exactly what makes the CSF/JM handshake probe in
 * kbase_kmod_dev_create() reliable (each flavour's kernel driver rejects
 * the other flavour's version-check number with -EPERM).
 */

#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "util/log.h"
#include "util/macros.h"

#include "drm-uapi/mali_kbase_ioctl.h"
#include "drm-uapi/mali_base_kernel.h"
#include "drm-uapi/mali_base_jm_kernel.h"
#include "drm-uapi/mali_kbase_jm_ioctl.h"

#include "kbase_kmod.h"
#include "kbase_jm.h"

/* pan_kmod_dev / pan_kmod_driver_version_at_least() live here. */
#include "pan_kmod.h"

/* -------------------------------------------------------------------------
 * Section 0 — CSF vs JM device-kind probe
 *
 * This is the "check which device we have, then choose the branch" step
 * the two command-submission families are dispatched from. It is derived
 * purely from the public struct pan_kmod_dev::driver.version that
 * kbase_kmod_dev_create() already fills in from the VERSION_CHECK
 * handshake (CSF reports uAPI 1.x, JM reports uAPI 11.x — see the
 * handshake comment at the top of kbase_kmod.c). No private field of
 * kbase_kmod.c's internal struct kbase_kmod_dev is touched, and no ioctl
 * is re-issued.
 * ---------------------------------------------------------------------- */

/* Declared with external linkage in kbase_kmod.c (the definition sits at
 * the bottom of that file); referenced here only to assert that a
 * pan_kmod_dev really was created by the kbase backend before we trust its
 * ->driver.version to mean what kbase means by it. */
extern const struct pan_kmod_ops kbase_kmod_ops;

enum kbase_gfx_dev_kind
kbase_gfx_dev_kind(const struct pan_kmod_dev *dev)
{
   if (!dev || dev->ops != &kbase_kmod_ops)
      return KBASE_GFX_DEV_UNKNOWN;

   /* Values match the handshake in kbase_kmod_dev_create(): the CSF ABI is
    * versioned 1.x, the JM ABI (kbase_kmod.c rejects anything older) is
    * versioned >= 11.x. These two ranges never overlap, so a plain
    * major-version test is sufficient and cannot misclassify. */
   if (dev->driver.version.major == 1)
      return KBASE_GFX_DEV_CSF;

   if (dev->driver.version.major >= 11)
      return KBASE_GFX_DEV_JM;

   return KBASE_GFX_DEV_UNKNOWN;
}

/* Every public entry point in this file starts with this guard. Kept as a
 * macro (rather than a helper returning bool) so the caller's own return
 * statement/value stays visible at the call site. */
#define KBASE_JM_REQUIRE_JM_DEVICE(dev, ret_on_fail)                        \
   do {                                                                     \
      if (kbase_gfx_dev_kind(dev) != KBASE_GFX_DEV_JM) {                    \
         mesa_loge("kbase_jm: refusing to run a JM-only operation on a "    \
                   "non-JM (CSF or unrecognised) device");                 \
         errno = ENOTSUP;                                                   \
         return (ret_on_fail);                                             \
      }                                                                     \
   } while (0)

/* -------------------------------------------------------------------------
 * Section 1 — GPU-properties helper (self-contained: does not call any
 * static helper from kbase_kmod.c)
 * ---------------------------------------------------------------------- */

/* Same little TLV format KBASE_IOCTL_GET_GPUPROPS returns, documented next
 * to kbase_kmod.c's own (static, private) copy of this parser:
 *   4-byte header (key << 2) | size_code, followed by 1/2/4/8 bytes of
 *   value depending on size_code. */
static uint64_t
kbase_jm_gpuprop_get(const uint8_t *buf, size_t buf_size,
                     uint32_t target_key, uint64_t default_val)
{
   size_t offset = 0;

   while (offset + 4 <= buf_size) {
      uint32_t hdr;
      memcpy(&hdr, buf + offset, 4);
      offset += 4;

      uint32_t key = hdr >> 2;
      uint32_t size_code = hdr & 0x3;
      uint32_t val_size = 1u << size_code;

      if (offset + val_size > buf_size)
         break;

      if (key == target_key) {
         uint64_t val = 0;
         memcpy(&val, buf + offset, val_size);
         return val;
      }

      offset += val_size;
   }

   return default_val;
}

static uint8_t *
kbase_jm_get_gpuprops(int fd, size_t *out_size)
{
   struct kbase_ioctl_get_gpuprops req = { 0 };
   int ret = ioctl(fd, KBASE_IOCTL_GET_GPUPROPS, &req);
   if (ret < 0) {
      mesa_loge("kbase_jm: KBASE_IOCTL_GET_GPUPROPS (probe) failed: %s",
                strerror(errno));
      return NULL;
   }

   size_t size = (size_t)ret;
   if (size == 0) {
      mesa_loge("kbase_jm: KBASE_IOCTL_GET_GPUPROPS returned zero size");
      return NULL;
   }

   uint8_t *buf = malloc(size);
   if (!buf)
      return NULL;

   req.buffer = (uintptr_t)buf;
   req.size = (uint32_t)size;

   if (ioctl(fd, KBASE_IOCTL_GET_GPUPROPS, &req) < 0) {
      mesa_loge("kbase_jm: KBASE_IOCTL_GET_GPUPROPS (fill) failed: %s",
                strerror(errno));
      free(buf);
      return NULL;
   }

   *out_size = size;
   return buf;
}

/* KBASE_GPUPROP_RAW_JS_* keys (from kbase_uapi.h): JS_PRESENT is 34,
 * JS_FEATURES_0..15 are 35..50 (one raw hardware register each). */
#define KBASE_JM_GPUPROP_RAW_JS_PRESENT      34
#define KBASE_JM_GPUPROP_RAW_JS_FEATURES_0   35

/* -------------------------------------------------------------------------
 * Section 2 — Job-slot (JS_PRESENT / JS_FEATURES_n) introspection
 * ---------------------------------------------------------------------- */

int
kbase_jm_query_job_slots(struct pan_kmod_dev *dev,
                          struct kbase_jm_job_slot_info *out)
{
   KBASE_JM_REQUIRE_JM_DEVICE(dev, -1);

   memset(out, 0, sizeof(*out));

   size_t props_size = 0;
   uint8_t *props = kbase_jm_get_gpuprops(dev->fd, &props_size);
   if (!props)
      return -1;

   /* JS_PRESENT is a bitmask: bit n set means job slot n exists. This is
    * the literal hardware JS_PRESENT register value, not a derived count,
    * so a slot count on real hardware line up 1:1 with popcount(). */
   uint32_t js_present =
      (uint32_t)kbase_jm_gpuprop_get(props, props_size,
                                     KBASE_JM_GPUPROP_RAW_JS_PRESENT, 0);

   uint32_t slot_count = 0;
   for (uint32_t slot = 0; slot < KBASE_JM_MAX_JOB_SLOTS; slot++) {
      if (!(js_present & (1u << slot)))
         continue;

      out->features[slot_count] = (uint32_t)kbase_jm_gpuprop_get(
         props, props_size, KBASE_JM_GPUPROP_RAW_JS_FEATURES_0 + slot, 0);
      slot_count++;
   }

   out->slot_count = slot_count;
   free(props);

   if (slot_count == 0) {
      mesa_loge("kbase_jm: JS_PRESENT reported zero job slots");
      errno = EIO;
      return -1;
   }

   mesa_logd("kbase_jm: %u job slot(s), JS_PRESENT=0x%x", slot_count,
             js_present);
   for (uint32_t i = 0; i < slot_count; i++)
      mesa_logd("kbase_jm:   JS%u features=0x%08x%s%s%s%s", i,
                out->features[i],
                (out->features[i] & KBASE_JM_JSn_FEATURE_VERTEX) ? " VERTEX" : "",
                (out->features[i] & KBASE_JM_JSn_FEATURE_TILER) ? " TILER" : "",
                (out->features[i] & KBASE_JM_JSn_FEATURE_FRAGMENT) ? " FRAGMENT" : "",
                (out->features[i] & KBASE_JM_JSn_FEATURE_COMPUTE) ? " COMPUTE" : "");

   return 0;
}

/* Picks the lowest-numbered job slot whose JS_FEATURES_n register
 * advertises every capability bit @required. Returns the slot index, or
 * -1 if no slot qualifies. */
static int
kbase_jm_pick_slot(const struct kbase_jm_job_slot_info *slots,
                   uint32_t required)
{
   for (uint32_t i = 0; i < slots->slot_count; i++) {
      if ((slots->features[i] & required) == required)
         return (int)i;
   }
   return -1;
}

/* -------------------------------------------------------------------------
 * Section 3 — Atom submission
 * ---------------------------------------------------------------------- */

/* Monotonic atom-id allocator. kbase JM atom ids are a per-fd __u8
 * namespace (BASE_JD_ATOM_COUNT = 256); id 0 is reserved by this module to
 * mean "no dependency" (see kbase_jm_atom_desc::depends_on_atom), so the
 * usable range is 1..255 and then wraps. Atom-id lifetime/exhaustion
 * tracking is the caller's responsibility once a real submission queue
 * with many in-flight atoms is built on top of this. */
static uint8_t next_atom_id = 1;

static base_jd_core_req
kbase_jm_core_req_for_kind(enum kbase_jm_atom_kind kind)
{
   switch (kind) {
   case KBASE_JM_ATOM_VERTEX_TILER:
      return BASE_JD_REQ_FS | BASE_JD_REQ_T;
   case KBASE_JM_ATOM_FRAGMENT:
      return BASE_JD_REQ_FS;
   case KBASE_JM_ATOM_COMPUTE:
      return BASE_JD_REQ_CS | BASE_JD_REQ_ONLY_COMPUTE;
   }
   unreachable("invalid kbase_jm_atom_kind");
}

static uint32_t
kbase_jm_required_js_features(enum kbase_jm_atom_kind kind)
{
   switch (kind) {
   case KBASE_JM_ATOM_VERTEX_TILER:
      return KBASE_JM_JSn_FEATURE_VERTEX | KBASE_JM_JSn_FEATURE_TILER;
   case KBASE_JM_ATOM_FRAGMENT:
      return KBASE_JM_JSn_FEATURE_FRAGMENT;
   case KBASE_JM_ATOM_COMPUTE:
      return KBASE_JM_JSn_FEATURE_COMPUTE;
   }
   unreachable("invalid kbase_jm_atom_kind");
}

static base_jd_prio
kbase_jm_base_prio(enum kbase_jm_atom_priority prio)
{
   switch (prio) {
   case KBASE_JM_PRIO_REALTIME: return BASE_JD_PRIO_REALTIME;
   case KBASE_JM_PRIO_HIGH:     return BASE_JD_PRIO_HIGH;
   case KBASE_JM_PRIO_MEDIUM:   return BASE_JD_PRIO_MEDIUM;
   case KBASE_JM_PRIO_LOW:      return BASE_JD_PRIO_LOW;
   }
   unreachable("invalid kbase_jm_atom_priority");
}

int
kbase_jm_atom_submit(struct pan_kmod_dev *dev,
                     const struct kbase_jm_atom_desc *desc)
{
   KBASE_JM_REQUIRE_JM_DEVICE(dev, -1);

   int jobslot = desc->jobslot;
   if (jobslot < 0) {
      struct kbase_jm_job_slot_info slots;
      if (kbase_jm_query_job_slots(dev, &slots))
         return -1;

      jobslot =
         kbase_jm_pick_slot(&slots, kbase_jm_required_js_features(desc->kind));
      if (jobslot < 0) {
         mesa_loge("kbase_jm: no job slot advertises the JS_FEATURES this "
                   "atom kind (%d) requires", desc->kind);
         errno = ENOTSUP;
         return -1;
      }
   }

   uint8_t atom_id = next_atom_id++;
   if (next_atom_id == 0) /* keep 0 reserved for "no dependency" */
      next_atom_id = 1;

   base_jd_atom atom = {
      .seq_nr = atom_id,
      .jc = desc->jc,
      .core_req = kbase_jm_core_req_for_kind(desc->kind),
      .atom_number = atom_id,
      .prio = kbase_jm_base_prio(desc->priority),
      .jobslot = (uint8_t)jobslot,
   };

   /* BASE_JD_REQ_JOB_SLOT is required for the .jobslot field above to be
    * honoured instead of left to the scheduler's own placement. */
   atom.core_req |= BASE_JD_REQ_JOB_SLOT;

   if (desc->depends_on_atom != 0) {
      atom.pre_dep[0].atom_id = desc->depends_on_atom;
      atom.pre_dep[0].dependency_type = BASE_JD_DEP_TYPE_DATA;
   }

   struct kbase_ioctl_job_submit submit = {
      .addr = (uintptr_t)&atom,
      .nr_atoms = 1,
      .stride = sizeof(atom),
   };

   if (ioctl(dev->fd, KBASE_IOCTL_JOB_SUBMIT, &submit)) {
      mesa_loge("kbase_jm: KBASE_IOCTL_JOB_SUBMIT failed: %s",
                strerror(errno));
      return -1;
   }

   mesa_logd("kbase_jm: submitted atom %u (kind=%d, prio=%d, slot=%d, "
             "dep=%u)", atom_id, desc->kind, desc->priority, jobslot,
             desc->depends_on_atom);

   return atom_id;
}

/* -------------------------------------------------------------------------
 * Section 4 — Completion events
 *
 * JM reports job completion by making the kbase fd readable and returning
 * an array of struct base_jd_event_v2 from read(), one entry per completed
 * atom (this is the JM analogue of the CSF notification stream that
 * kbase_kmod_csf_wait_event() polls -- the two payloads are not
 * interchangeable, which is why this is a separate function rather than a
 * shared one).
 * ---------------------------------------------------------------------- */

int
kbase_jm_wait_event(struct pan_kmod_dev *dev, int64_t timeout_ns,
                    uint8_t *out_atom_number, bool *out_succeeded)
{
   KBASE_JM_REQUIRE_JM_DEVICE(dev, -1);

   struct pollfd pfd = { .fd = dev->fd, .events = POLLIN };
   int timeout_ms = timeout_ns < 0 ? -1 : (int)(timeout_ns / 1000000);

   int pret = poll(&pfd, 1, timeout_ms);
   if (pret < 0) {
      mesa_loge("kbase_jm: poll() failed: %s", strerror(errno));
      return -1;
   }
   if (pret == 0)
      return 0; /* timeout, no event */

   struct base_jd_event_v2 event;
   ssize_t n = read(dev->fd, &event, sizeof(event));
   if (n < 0) {
      mesa_loge("kbase_jm: read() of JM event failed: %s", strerror(errno));
      return -1;
   }
   if (n != sizeof(event)) {
      mesa_loge("kbase_jm: short read of JM event (%zd of %zu bytes)", n,
                sizeof(event));
      errno = EIO;
      return -1;
   }

   if (out_atom_number)
      *out_atom_number = event.atom_number;
   if (out_succeeded)
      *out_succeeded = (event.event_code == BASE_JD_EVENT_DONE);

   mesa_logd("kbase_jm: atom %u completed, event_code=0x%x", event.atom_number,
             event.event_code);

   return 1;
}

/* -------------------------------------------------------------------------
 * Section 5 — Soft-events (KBASE_IOCTL_SOFT_EVENT_UPDATE)
 * ---------------------------------------------------------------------- */

int
kbase_jm_soft_event_update(struct pan_kmod_dev *dev, uint64_t event_gpu_va,
                           enum kbase_jm_soft_event_status status)
{
   KBASE_JM_REQUIRE_JM_DEVICE(dev, -1);

   if (event_gpu_va & 0x7) {
      /* BASE_JD_REQ_SOFT_EVENT_* jobs read/write the event as a naturally
       * aligned word; an unaligned address is always a caller bug, so
       * catch it here instead of letting the kernel refuse it later. */
      mesa_loge("kbase_jm: soft-event address 0x%" PRIx64 " is not aligned",
                event_gpu_va);
      errno = EINVAL;
      return -1;
   }

   struct kbase_ioctl_soft_event_update update = {
      .event = event_gpu_va,
      .new_status = (status == KBASE_JM_SOFT_EVENT_SET)
                       ? BASE_JD_SOFT_EVENT_SET
                       : BASE_JD_SOFT_EVENT_RESET,
   };

   if (ioctl(dev->fd, KBASE_IOCTL_SOFT_EVENT_UPDATE, &update)) {
      mesa_loge("kbase_jm: KBASE_IOCTL_SOFT_EVENT_UPDATE failed: %s",
                strerror(errno));
      return -1;
   }

   mesa_logd("kbase_jm: soft-event 0x%" PRIx64 " -> %s", event_gpu_va,
             status == KBASE_JM_SOFT_EVENT_SET ? "SET" : "RESET");

   return 0;
}

/* -------------------------------------------------------------------------
 * Section 6 — Context teardown notification (KBASE_IOCTL_POST_TERM)
 * ---------------------------------------------------------------------- */

int
kbase_jm_post_term(struct pan_kmod_dev *dev)
{
   KBASE_JM_REQUIRE_JM_DEVICE(dev, -1);

   if (ioctl(dev->fd, KBASE_IOCTL_POST_TERM)) {
      mesa_loge("kbase_jm: KBASE_IOCTL_POST_TERM failed: %s",
                strerror(errno));
      return -1;
   }

   mesa_logd("kbase_jm: posted context termination, any blocked "
             "kbase_jm_wait_event() should now return");

   return 0;
}
