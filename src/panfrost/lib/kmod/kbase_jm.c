/*
 * SPDX-License-Identifier: MIT
 *
 * kbase_jm.c — Job Manager (JM) command-submission backend for the ARM Mali
 * kbase kmod driver (/dev/mali*).
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

#include "mali_kbase_ioctl.h"
#include "mali_base_kernel.h"
#include "mali_base_jm_kernel.h"
#include "mali_kbase_jm_ioctl.h"

#include "kbase_kmod.h"
#include "kbase_jm.h"

#include "pan_kmod.h"

extern const struct pan_kmod_ops kbase_kmod_ops;

enum kbase_gfx_dev_kind
kbase_gfx_dev_kind(const struct pan_kmod_dev *dev)
{
   if (!dev || dev->ops != &kbase_kmod_ops)
      return KBASE_GFX_DEV_UNKNOWN;

   if (dev->driver.version.major == 1)
      return KBASE_GFX_DEV_CSF;

   if (dev->driver.version.major >= 11)
      return KBASE_GFX_DEV_JM;

   return KBASE_GFX_DEV_UNKNOWN;
}

#define KBASE_JM_REQUIRE_JM_DEVICE(dev, ret_on_fail)                        \
   do {                                                                     \
      if (kbase_gfx_dev_kind(dev) != KBASE_GFX_DEV_JM) {                    \
         mesa_loge("kbase_jm: refusing to run a JM-only operation on a "    \
                   "non-JM (CSF or unrecognised) device");                 \
         errno = ENOTSUP;                                                   \
         return (ret_on_fail);                                             \
      }                                                                     \
   } while (0)

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

#define KBASE_JM_GPUPROP_RAW_JS_PRESENT      34
#define KBASE_JM_GPUPROP_RAW_JS_FEATURES_0   35

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
   UNREACHABLE("invalid kbase_jm_atom_kind");
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
   UNREACHABLE("invalid kbase_jm_atom_kind");
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
   UNREACHABLE("invalid kbase_jm_atom_priority");
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
   if (next_atom_id == 0)
      next_atom_id = 1;

   base_jd_atom atom = {
      .seq_nr = atom_id,
      .jc = desc->jc,
      .core_req = kbase_jm_core_req_for_kind(desc->kind),
      .atom_number = atom_id,
      .prio = kbase_jm_base_prio(desc->priority),
      .jobslot = (uint8_t)jobslot,
   };

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
      return 0;

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

int
kbase_jm_soft_event_update(struct pan_kmod_dev *dev, uint64_t event_gpu_va,
                           enum kbase_jm_soft_event_status status)
{
   KBASE_JM_REQUIRE_JM_DEVICE(dev, -1);

   if (event_gpu_va & 0x7) {
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
