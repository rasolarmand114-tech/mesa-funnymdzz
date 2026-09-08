/*
 * Copyright © 2021 Collabora Ltd.
 *
 * Derived from tu_device.c which is:
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <inttypes.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "genxml/gen_macros.h"

#include "decode.h"

#include "panvk_cmd_buffer.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_event.h"
#include "panvk_image.h"
#include "panvk_image_view.h"
#include "panvk_instance.h"
#include "panvk_kbase_uapi.h"
#include "panvk_physical_device.h"
#include "panvk_priv_bo.h"
#include "panvk_queue.h"

#include "vk_framebuffer.h"
#include "vk_sync.h"

/*
 * NOTE ON THE PORT AWAY FROM kbase_jm.h, BACK TO RAW IOCTLS
 * ===========================================================
 * This file used to go through the kbase_jm.h add-on to the kmod backend
 * (kbase_jm_atom_submit(), kbase_jm_wait_event(), kbase_jm_query_job_slots(),
 * kbase_jm_soft_event_update(), kbase_jm_post_term()). It now talks to
 * /dev/mali* directly with the ioctls in mali_kbase_ioctl.h (the JM flavour:
 * VERSION_CHECK is ioctl nr 0), the same header this tree vendors as
 * panvk_kbase_uapi.h.
 *
 * mali_kbase_ioctl.h pins down KBASE_IOCTL_JOB_SUBMIT's *outer* shape
 * (struct kbase_ioctl_job_submit: a userspace pointer + count + stride) and
 * KBASE_IOCTL_GET_GPUPROPS's serialised property blob, but it deliberately
 * does not define the per-atom payload that .addr points to (that lives in
 * mali_kbase_jm_ioctl.h upstream, which wasn't part of the header handed to
 * this port) or the struct read() back from the device fd on completion.
 * Those two layouts -- struct base_jd_atom_v2 and struct base_jd_event_v2,
 * plus the base_jd_core_req / base_jd_event_code enums they embed -- are
 * reconstructed below from the public kbase JM ABI (the same ABI the
 * kbase_jm.h wrapper itself sits on top of) and marked
 * PANVK_KBASE_JM_ATOM_ABI_UNVERIFIED. Exactly like the disclaimer already in
 * panvk_kbase_uapi.h: a hand-reconstructed ioctl payload struct is a real
 * ABI-mismatch risk (wrong field order/size here means kernel memory
 * corruption, not a compile error), so before this lands anywhere real,
 * replace the block below with a verbatim copy of mali_kbase_jm_ioctl.h /
 * mali_base_kernel.h from the exact kernel commit being targeted, the same
 * way mali_kbase_ioctl.h itself was vendored.
 *
 * Assumptions this port makes that aren't nailed down by the header we do
 * have:
 *
 *  - struct panvk_gpu_queue (panvk_queue.h) keeps jm_last_atom, jm_vt_slot
 *    and jm_frag_slot exactly as before; only how they're produced and
 *    consumed changes.
 *
 *  - "0 means no dependency" for base_jd_dependency::atom_id, matching the
 *    kbase_jm.h wrapper's convention that this port replaces, and matching
 *    kbase's own reservation of atom_number 0 as a sentinel.
 *
 *  - The device fd (dev->kmod.dev->fd) has already been through the
 *    VERSION_CHECK_JM + SET_FLAGS handshake by the time a queue is created;
 *    that handshake is device-level setup and isn't repeated here.
 *
 *  - Soft-event set/reset for vkEvent support (panvk_per_arch(event_update)()
 *    / panvk_per_arch(event_is_set)(), implemented in panvk_vX_event.c) is
 *    left untouched -- out of scope for this file.
 *
 *  - kbase's read() on the device fd is blocking and yields whole
 *    struct base_jd_event_v2 records; poll()+read() one at a time is
 *    sufficient since this queue only ever has one atom in flight.
 */
#define PANVK_KBASE_JM_ATOM_ABI_UNVERIFIED 1

/* --- base_jd_core_req (job requirement flags), JM job-slot routing bits.
 * Public kbase ABI; see the disclaimer above. */
typedef __u32 base_jd_core_req;

#define BASE_JD_REQ_FS       ((base_jd_core_req)1 << 0) /* fragment */
#define BASE_JD_REQ_CS       ((base_jd_core_req)1 << 1) /* compute */
#define BASE_JD_REQ_T        ((base_jd_core_req)1 << 2) /* tiler */
#define BASE_JD_REQ_CF       ((base_jd_core_req)1 << 3) /* cache flush only */
#define BASE_JD_REQ_V        ((base_jd_core_req)1 << 4) /* vertex/geometry */
#define BASE_JD_REQ_SOFT_JOB ((base_jd_core_req)1 << 9)

#define BASE_JD_REQ_SOFT_EVENT_WAIT  (BASE_JD_REQ_SOFT_JOB | 0x2u)
#define BASE_JD_REQ_SOFT_EVENT_SET   (BASE_JD_REQ_SOFT_JOB | 0x3u)
#define BASE_JD_REQ_SOFT_EVENT_RESET (BASE_JD_REQ_SOFT_JOB | 0x4u)

#define BASE_JD_DEP_TYPE_INVALID 0
#define BASE_JD_DEP_TYPE_DATA    (1u << 0)

struct base_jd_dependency {
   __u8 atom_id;
   __u8 dependency_type;
};

/* Deprecated even in the real ABI, but still present in the wire struct;
 * left zeroed here. */
struct base_jd_udata {
   __u64 blob[2];
};

/* This is the struct kbase_ioctl_job_submit::addr / ::stride payload:
 * one array element per atom, nr_atoms of them, each stride bytes apart
 * (== sizeof(struct base_jd_atom_v2) when tightly packed, as we do here). */
struct base_jd_atom_v2 {
   __u64 jc; /* job chain GPU VA, or soft-job payload VA */
   struct base_jd_udata udata;
   __u64 extres_list; /* unused: no external resource list here */
   __u16 nr_extres;
   __u16 compat_core_req; /* legacy alias of core_req; kept 0 */
   struct base_jd_dependency pre_dep[2];
   __u8 atom_number; /* in: 0 == "assign me one"; out: assigned id */
   __s8 prio;        /* base_jd_prio; 0 == medium */
   __u8 device_nr;
   __u8 jobslot;
   base_jd_core_req core_req;
   __u8 padding[4];
};

#define BASE_JD_PRIO_MEDIUM 0

/* struct read() back from the kbase device fd, one per completed atom. */
enum base_jd_event_code {
   BASE_JD_EVENT_DONE = 0,
   /* Any other value: some flavour of fault/timeout/removed-from-queue.
    * We don't need to distinguish further than "not DONE" here. */
};

struct base_jd_event_v2 {
   __u32 event_code;
   __u8 atom_number;
   __u8 padding[3];
   struct base_jd_udata udata;
};

/* --- JS_FEATURES bits (per hardware job-slot capability register).
 * Public ARM TRM / kbase GPU-properties ABI; see the disclaimer above. */
#define JS_FEATURE_VERTEX_JOB   (1u << 5)
#define JS_FEATURE_TILER_JOB    (1u << 7)
#define JS_FEATURE_FRAGMENT_JOB (1u << 9)

#define PANVK_KBASE_MAX_JOB_SLOTS 16

struct panvk_kbase_js_features {
   uint32_t slot_present_mask; /* raw JS_PRESENT: bit i set => slot i exists */
   uint32_t features[PANVK_KBASE_MAX_JOB_SLOTS];
};

/* Read the KBASE_IOCTL_GET_GPUPROPS blob (probe-then-fetch, per the ioctl's
 * documented two-call protocol) and pull out RAW_JS_PRESENT plus each
 * present slot's RAW_JS_FEATURES_<n> entry.
 *
 * Blob format, straight from the header: a stream of
 *   [u32 LE header = (key << 2) | size_code] [value, size_code bytes]
 * with size_code 0/1/2/3 meaning u8/u16/u32/u64 respectively.
 */
static int
panvk_kbase_get_js_features(int fd, struct panvk_kbase_js_features *out)
{
   memset(out, 0, sizeof(*out));

   union kbase_ioctl_get_gpuprops probe = {
      .buffer = 0,
      .size = 0,
      .flags = 0,
   };

   int blob_size = ioctl(fd, KBASE_IOCTL_GET_GPUPROPS, &probe);
   if (blob_size < 0)
      return -1;
   if (blob_size == 0)
      return 0;

   void *blob = malloc((size_t)blob_size);
   if (!blob) {
      errno = ENOMEM;
      return -1;
   }

   union kbase_ioctl_get_gpuprops fetch = {
      .buffer = (uintptr_t)blob,
      .size = (uint32_t)blob_size,
      .flags = 0,
   };

   int ret = ioctl(fd, KBASE_IOCTL_GET_GPUPROPS, &fetch);
   if (ret < 0) {
      int saved_errno = errno;
      free(blob);
      errno = saved_errno;
      return -1;
   }

   const uint8_t *p = blob;
   const uint8_t *end = p + blob_size;

   while (p + 4 <= end) {
      uint32_t header = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                        ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
      p += 4;

      uint32_t key = header >> 2;
      uint32_t size_code = header & 0x3;
      uint32_t value_size = 1u << size_code; /* 1, 2, 4, or 8 bytes */

      if (p + value_size > end)
         break;

      uint64_t value = 0;
      for (uint32_t i = 0; i < value_size; i++)
         value |= (uint64_t)p[i] << (8 * i);
      p += value_size;

      if (key == KBASE_GPUPROP_RAW_JS_PRESENT) {
         out->slot_present_mask = (uint32_t)value;
      } else if (key == KBASE_GPUPROP_RAW_JS_FEATURES_0) {
         out->features[0] = (uint32_t)value;
      } else if (key > KBASE_GPUPROP_RAW_JS_FEATURES_0 &&
                key <= KBASE_GPUPROP_RAW_JS_FEATURES_0 + 15) {
         out->features[key - KBASE_GPUPROP_RAW_JS_FEATURES_0] =
            (uint32_t)value;
      }
   }

   free(blob);
   return 0;
}

/* Pick the first present job slot whose advertised JS_FEATURES cover every
 * bit in want_mask. Mirrors what kbase_js_choose_affinity()/mali_kbase_js.c
 * does kernel-side, just done once up front instead of per submission. */
static int
panvk_kbase_pick_job_slot(const struct panvk_kbase_js_features *slots,
                          uint32_t want_mask)
{
   for (uint32_t i = 0; i < PANVK_KBASE_MAX_JOB_SLOTS; i++) {
      if (!(slots->slot_present_mask & (1u << i)))
         continue;

      if ((slots->features[i] & want_mask) == want_mask)
         return (int)i;
   }

   return -1;
}

/* Submit a single job chain as one kbase JM atom, chained onto the atom
 * this queue submitted last, and block until it (and therefore everything
 * submitted before it on this queue) has completed.
 *
 * See the big comment on panvk_gpu_queue::jm_last_atom for why this is
 * synchronous instead of returning a fence-like object: a kbase atom's
 * dependency can only reference one prior atom_number *on this same
 * context*, and completion is only observable by draining the shared JM
 * event stream for the device -- there's nothing DRM-syncobj-shaped to
 * export or wait on from outside this function.
 */
static bool
panvk_queue_jm_submit_atom(struct panvk_gpu_queue *queue,
                           base_jd_core_req core_req, int jobslot,
                           uint64_t jc)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);
   int fd = dev->kmod.dev->fd;

   struct base_jd_atom_v2 atom = {
      .jc = jc,
      .atom_number = 0, /* let the kernel assign one */
      .prio = BASE_JD_PRIO_MEDIUM,
      .device_nr = 0,
      .jobslot = (uint8_t)jobslot,
      .core_req = core_req,
      .pre_dep[0] =
         {
            /* 0 means "no dependency" -- see the port notes above. On this
             * queue only one atom is ever outstanding, so there's no risk
             * of this depending on an atom that already got
             * reused/recycled kernel-side. */
            .atom_id = queue->jm_last_atom,
            .dependency_type =
               queue->jm_last_atom ? BASE_JD_DEP_TYPE_DATA
                                   : BASE_JD_DEP_TYPE_INVALID,
         },
      .pre_dep[1] = {.atom_id = 0, .dependency_type = BASE_JD_DEP_TYPE_INVALID},
   };

   struct kbase_ioctl_job_submit submit = {
      .addr = (uintptr_t)&atom,
      .nr_atoms = 1,
      .stride = sizeof(atom),
   };

   int ret = ioctl(fd, KBASE_IOCTL_JOB_SUBMIT, &submit);
   if (ret < 0) {
      mesa_loge("panvk: KBASE_IOCTL_JOB_SUBMIT failed: %s", strerror(errno));
      return false;
   }

   uint8_t atom_number = atom.atom_number;
   queue->jm_last_atom = atom_number;

   /* Only one atom is ever in flight at a time in this submission model,
    * so the next completion for *this atom_number* is the one we're
    * waiting for; anything else reported first (there shouldn't be
    * anything else, but the ABI doesn't promise it) is skipped. */
   for (;;) {
      struct pollfd pfd = {.fd = fd, .events = POLLIN};

      int pret;
      do {
         pret = poll(&pfd, 1, -1);
      } while (pret < 0 && errno == EINTR);

      if (pret < 0) {
         mesa_loge("panvk: poll() on kbase fd failed: %s", strerror(errno));
         return false;
      }

      struct base_jd_event_v2 evt;
      ssize_t rret = read(fd, &evt, sizeof(evt));
      if (rret < 0) {
         if (errno == EAGAIN || errno == EINTR)
            continue;
         mesa_loge("panvk: read() on kbase fd failed: %s", strerror(errno));
         return false;
      }
      if (rret != (ssize_t)sizeof(evt)) {
         mesa_loge("panvk: short read on kbase fd (%zd of %zu bytes)", rret,
                   sizeof(evt));
         return false;
      }

      if (evt.atom_number != atom_number)
         continue;

      if (evt.event_code != BASE_JD_EVENT_DONE) {
         mesa_loge("panvk: kbase JM atom %u (job chain 0x%" PRIx64
                   ") reported failure (event_code=%u)",
                   atom_number, jc, evt.event_code);
         return false;
      }

      return true;
   }
}

static bool
panvk_queue_submit_batch(struct panvk_gpu_queue *queue,
                         struct panvk_cmd_buffer *cmdbuf,
                         struct panvk_batch *batch)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);
   struct panvk_physical_device *phys_dev =
      to_panvk_physical_device(dev->vk.physical);

   /* Reset the batch if it's already been issued */
   if (batch->issued) {
      util_dynarray_foreach(&batch->jobs, void *, job)
         memset((*job), 0, 4 * 4);

      /* Reset the tiler before re-issuing the batch */
      if (batch->tiler.ctx_descs.cpu) {
         memcpy(batch->tiler.heap_desc.cpu, &batch->tiler.heap_templ,
                sizeof(batch->tiler.heap_templ));

         struct mali_tiler_context_packed *ctxs = batch->tiler.ctx_descs.cpu;

         for (uint32_t i = 0; i < batch->fb.layer_count; i++)
            memcpy(&ctxs[i], &batch->tiler.ctx_templ, sizeof(*ctxs));
      }

      /* We don't keep track of BO <-> job relationship, so let's just flush the
       * whole desc pool for now. */
      panvk_pool_flush_maps(&cmdbuf->desc_pool);
   }

   /* Flush pending synchronization requests before submitting the job, to
    * make sure things are GPU-visible. */
   pan_kmod_flush_bo_map_syncs(dev->kmod.dev);

   if (batch->vtc_jc.first_job) {
      if (!panvk_queue_jm_submit_atom(
             queue, BASE_JD_REQ_V | BASE_JD_REQ_T | BASE_JD_REQ_CS,
             queue->jm_vt_slot, batch->vtc_jc.first_job))
         return false;

      /* Submission is always synchronous now, so the work is already done;
       * this is only about deciding whether to pay for readback/decode. */
      if (PANVK_DEBUG(TRACE) || PANVK_DEBUG(SYNC)) {
         /* If we want to read the descriptors back, we need to invalidate the
          * whole desc pool, otherwise we might end up with stale data. */
         panvk_pool_invalidate_maps(&cmdbuf->desc_pool);
         pan_kmod_flush_bo_map_syncs(dev->kmod.dev);
      }

      if (PANVK_DEBUG(TRACE))
         pandecode_jc(dev->debug.decode_ctx, batch->vtc_jc.first_job,
                      phys_dev->kmod.dev->props.gpu_id);

      if (PANVK_DEBUG(DUMP))
         pandecode_dump_mappings(dev->debug.decode_ctx);

      if (PANVK_DEBUG(SYNC))
         pandecode_abort_on_fault(dev->debug.decode_ctx,
                                  batch->vtc_jc.first_job,
                                  phys_dev->kmod.dev->props.gpu_id);
   }

   if (batch->frag_jc.first_job) {
      if (!panvk_queue_jm_submit_atom(queue, BASE_JD_REQ_FS,
                                      queue->jm_frag_slot,
                                      batch->frag_jc.first_job))
         return false;

      if (PANVK_DEBUG(TRACE) || PANVK_DEBUG(SYNC)) {
         panvk_pool_invalidate_maps(&cmdbuf->desc_pool);
         pan_kmod_flush_bo_map_syncs(dev->kmod.dev);
      }

      if (PANVK_DEBUG(TRACE))
         pandecode_jc(dev->debug.decode_ctx, batch->frag_jc.first_job,
                      phys_dev->kmod.dev->props.gpu_id);

      if (PANVK_DEBUG(DUMP))
         pandecode_dump_mappings(dev->debug.decode_ctx);

      if (PANVK_DEBUG(SYNC))
         pandecode_abort_on_fault(dev->debug.decode_ctx,
                                  batch->frag_jc.first_job,
                                  phys_dev->kmod.dev->props.gpu_id);
   }

   if (PANVK_DEBUG(TRACE))
      pandecode_next_frame(dev->debug.decode_ctx);

   batch->issued = true;
   return true;
}

/* vkCmdWaitEvents2() operations recorded on this batch. There's no GPU-side
 * soft-event-wait atom we submit here, so -- same spirit as the
 * synchronous atom submission above -- we just block the CPU on the
 * host-visible event status until it's set. Because this queue only ever
 * has one batch in flight at a time, nothing downstream can race ahead of
 * this wait. */
static VkResult
panvk_queue_wait_events(struct panvk_gpu_queue *queue,
                        struct panvk_batch *batch)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);

   util_dynarray_foreach(&batch->event_ops, struct panvk_cmd_event_op, op) {
      if (op->type != PANVK_EVENT_OP_WAIT)
         continue;

      while (!panvk_per_arch(event_is_set)(op->event)) {
         if (vk_device_is_lost(&dev->vk))
            return VK_ERROR_DEVICE_LOST;

         thrd_yield();
      }
   }

   return VK_SUCCESS;
}

/* Event set/reset itself is left going through panvk_per_arch(event_update)()
 * (implemented in panvk_vX_event.c), which is out of scope for this port --
 * see the port notes at the top of the file. */
static VkResult
panvk_queue_signal_events(struct panvk_gpu_queue *queue,
                          struct panvk_batch *batch)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);

   util_dynarray_foreach(&batch->event_ops, struct panvk_cmd_event_op, op) {
      switch (op->type) {
      case PANVK_EVENT_OP_SET:
         if (!panvk_per_arch(event_update)(dev, op->event,
                                           KBASE_JM_SOFT_EVENT_SET))
            return VK_ERROR_DEVICE_LOST;
         break;
      case PANVK_EVENT_OP_RESET:
         if (!panvk_per_arch(event_update)(dev, op->event,
                                           KBASE_JM_SOFT_EVENT_RESET))
            return VK_ERROR_DEVICE_LOST;
         break;
      case PANVK_EVENT_OP_WAIT:
         /* Handled up-front in panvk_queue_wait_events(). */
         break;
      default:
         UNREACHABLE("bad panvk_cmd_event_op type\n");
      }
   }

   return VK_SUCCESS;
}

VkResult
panvk_per_arch(gpu_queue_submit)(struct vk_queue *vk_queue,
                                 struct vk_queue_submit *submit)
{
   struct panvk_gpu_queue *queue =
      container_of(vk_queue, struct panvk_gpu_queue, vk);
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);
   VkResult result;

   /* A kbase atom's dependency can only chain onto prior atoms *on this
    * same queue*; there's no way to hand it an external semaphore as a
    * GPU-side dependency. So wait semaphores are resolved on the CPU,
    * before we submit anything. This queue's own submissions are already
    * fully synchronous (see panvk_queue_jm_submit_atom()), so this doesn't
    * give up any pipelining we'd otherwise have had. */
   for (unsigned i = 0; i < submit->wait_count; i++) {
      result = vk_sync_wait(&dev->vk, submit->waits[i].sync,
                            submit->waits[i].wait_value,
                            VK_SYNC_WAIT_COMPLETE, UINT64_MAX);
      if (result != VK_SUCCESS)
         return result;
   }

   for (uint32_t j = 0; j < submit->command_buffer_count; ++j) {
      struct panvk_cmd_buffer *cmdbuf = container_of(
         submit->command_buffers[j], struct panvk_cmd_buffer, vk);

      list_for_each_entry(struct panvk_batch, batch, &cmdbuf->batches, node) {
         result = panvk_queue_wait_events(queue, batch);
         if (result != VK_SUCCESS)
            return result;

         if (!panvk_queue_submit_batch(queue, cmdbuf, batch))
            return vk_queue_set_lost(&queue->vk,
                                     "kbase JM atom submission failed");

         result = panvk_queue_signal_events(queue, batch);
         if (result != VK_SUCCESS)
            return result;
      }
   }

   /* Every batch above already ran to completion by the time we get here,
    * so signalling is pure host-side bookkeeping. */
   for (unsigned i = 0; i < submit->signal_count; i++) {
      result = vk_sync_signal(&dev->vk, submit->signals[i].sync,
                              submit->signals[i].signal_value);
      if (result != VK_SUCCESS)
         return result;
   }

   return VK_SUCCESS;
}

VkResult
panvk_per_arch(create_gpu_queue)(struct panvk_device *device,
                                 const VkDeviceQueueCreateInfo *create_info,
                                 uint32_t queue_idx,
                                 struct vk_queue **out_queue)
{
   ASSERTED const VkDeviceQueueGlobalPriorityCreateInfoKHR *priority_info =
      vk_find_struct_const(create_info->pNext,
                           DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR);
   ASSERTED const VkQueueGlobalPriorityKHR priority =
      priority_info ? priority_info->globalPriority
                    : VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;

   /* XXX: struct base_jd_atom_v2 only carries a per-atom base_jd_prio value,
    * not a queue-wide priority negotiated at creation time, so we don't
    * plumb anything beyond MEDIUM through yet. */
   assert(priority == VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR);

   assert(kbase_gfx_dev_kind(device->kmod.dev) == KBASE_GFX_DEV_JM);

   struct panvk_gpu_queue *queue =
      vk_zalloc(&device->vk.alloc, sizeof(*queue), 8,
               VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!queue)
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result =
      vk_queue_init(&queue->vk, &device->vk, create_info, queue_idx);
   if (result != VK_SUCCESS)
      goto err_free_queue;

   /* Job-slot routing (which hardware job slot vertex/tiler vs fragment
    * atoms land on) is resolved once here against this device's
    * RAW_JS_FEATURES_<n> GPU properties, read straight off
    * KBASE_IOCTL_GET_GPUPROPS, and stashed per-atom in
    * struct base_jd_atom_v2::jobslot at submission time. */
   struct panvk_kbase_js_features slots;
   int ret = panvk_kbase_get_js_features(device->kmod.dev->fd, &slots);
   if (ret) {
      result = panvk_errorf(device, VK_ERROR_INITIALIZATION_FAILED,
                            "failed to query kbase JS_FEATURES via "
                            "KBASE_IOCTL_GET_GPUPROPS: %s",
                            strerror(errno));
      goto err_finish_queue;
   }

   queue->jm_vt_slot = panvk_kbase_pick_job_slot(
      &slots, JS_FEATURE_VERTEX_JOB | JS_FEATURE_TILER_JOB);
   queue->jm_frag_slot =
      panvk_kbase_pick_job_slot(&slots, JS_FEATURE_FRAGMENT_JOB);

   if (queue->jm_vt_slot < 0 || queue->jm_frag_slot < 0) {
      result = panvk_errorf(device, VK_ERROR_INITIALIZATION_FAILED,
                            "no kbase JM job slot advertises the required "
                            "vertex+tiler/fragment JS_FEATURES");
      goto err_finish_queue;
   }

   queue->vk.driver_submit = panvk_per_arch(gpu_queue_submit);
   *out_queue = &queue->vk;
   return VK_SUCCESS;

err_finish_queue:
   vk_queue_finish(&queue->vk);
err_free_queue:
   vk_free(&device->vk.alloc, queue);
   return result;
}

void
panvk_per_arch(destroy_gpu_queue)(struct vk_queue *vk_queue)
{
   struct panvk_gpu_queue *queue =
      container_of(vk_queue, struct panvk_gpu_queue, vk);
   struct panvk_device *dev = to_panvk_device(vk_queue->base.device);

   /* Nothing JM-specific to tear down here with the raw-ioctl backend: no
    * per-queue kernel-side context beyond what closing the device fd
    * already releases at device destruction. */

   vk_queue_finish(&queue->vk);
   vk_free(&dev->vk.alloc, queue);
}

VkResult
panvk_per_arch(gpu_queue_check_status)(struct vk_queue *vk_queue)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(QueueWaitIdle)(VkQueue _queue)
{
   VK_FROM_HANDLE(panvk_gpu_queue, queue, _queue);
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);

   /* we need to use vk_common_QueueWaitIdle if we ever go threaded */
   assert(queue->vk.submit.mode != VK_QUEUE_SUBMIT_MODE_THREADED);

   if (vk_device_is_lost(&dev->vk)) {
      /* Check printf buffer one more time before exiting */
      u_printf_with_ctx(stdout, &dev->printf.ctx);
      return VK_ERROR_DEVICE_LOST;
   }

   /* panvk_per_arch(gpu_queue_submit)() already blocks until every atom it
    * submits has completed (see panvk_queue_jm_submit_atom()), so there is
    * nothing left to wait for by the time control returns to this queue's
    * caller. */
   return VK_SUCCESS;
}
