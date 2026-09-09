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
 * NOTE ON THIS PORT (JM-only, raw kbase ioctl UAPI)
 * ===================================================
 * This talks to /dev/mali* directly through the vendored kbase uapi
 * headers: mali_kbase_ioctl.h, mali_kbase_jm_ioctl.h, mali_base_kernel.h and
 * mali_base_jm_kernel.h (pulled in transitively via panvk_kbase_uapi.h /
 * mali_base_kernel.h's own #include of mali_base_jm_kernel.h). No
 * kbase_jm.h wrapper is used anywhere in this file.
 *
 * FIX (see the big comment above panvk_kbase_pick_job_slot() and inside
 * create_gpu_queue()): the previous version of this file treated
 * "no single job slot advertises both VERTEX and TILER in its JS_FEATURES"
 * as a *fatal* vkCreateDevice() failure. That check had no functional
 * purpose to begin with: struct base_jd_atom_v2::jobslot is only honoured
 * by the kernel "when BASE_JD_REQ_JOB_SLOT is specified" in core_req (see
 * the @jobslot doc comment in mali_base_jm_kernel.h) -- and this file never
 * sets that bit, so the discovered slot was never actually consulted at
 * submission time either way. In other words: we were hard-failing device
 * creation over a value that, even when found, would have been silently
 * discarded. Real hardware doesn't need it: leaving BASE_JD_REQ_JOB_SLOT
 * unset makes the kernel pick the job slot itself from the core_req bits
 * via kbase_js_choose_affinity() (see the historical note that used to sit
 * at the top of this file, before any of this GET_GPUPROPS-based slot
 * lookup was added) -- which is exactly the "let the kernel route it"
 * behaviour every other panvk kbase backend relies on. So slot discovery
 * is now best-effort/diagnostic only: we still query and log it (useful
 * for understanding a given board's JS_FEATURES layout), but a miss no
 * longer blocks device creation, and submission only opts into an explicit
 * slot when discovery actually found one.
 */

/* -----------------------------------------------------------------------
 * Job-slot discovery via KBASE_IOCTL_GET_GPUPROPS. Best-effort/diagnostic:
 * see the FIX note above for why a miss here is not fatal.
 * ----------------------------------------------------------------------- */

/* JS_FEATURES bits (per hardware job-slot capability register). Not part of
 * the uapi headers themselves (those only give the *key* used to fetch the
 * raw register value via GET_GPUPROPS, not the register's own bit layout)
 * -- these come from the public Mali GPU_JS_FEATURES register
 * documentation, flagged here since it's the one piece not sourced
 * directly from the four headers above. */
#define JS_FEATURE_VERTEX_JOB   (1u << 5)
#define JS_FEATURE_TILER_JOB    (1u << 7)
#define JS_FEATURE_FRAGMENT_JOB (1u << 9)

#define PANVK_KBASE_MAX_JOB_SLOTS 16

struct panvk_kbase_js_features {
   uint32_t slot_present_mask; /* raw JS_PRESENT: bit i set => slot i exists */
   uint32_t features[PANVK_KBASE_MAX_JOB_SLOTS];
};

/* Probe-then-fetch per KBASE_IOCTL_GET_GPUPROPS's documented protocol:
 * call once with size=0 to learn the blob size (returned as the ioctl's
 * return value), then again with a buffer of that size. Blob format, from
 * mali_kbase_ioctl.h's own doc comment: a stream of
 *   [u32 LE header = (key << 2) | size_code] [value, size_code bytes]
 * with size_code 00/01/10/11 meaning u8/u16/u32/u64 respectively. */
static int
panvk_kbase_get_js_features(int fd, struct panvk_kbase_js_features *out)
{
   memset(out, 0, sizeof(*out));

   struct kbase_ioctl_get_gpuprops probe = {
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

   struct kbase_ioctl_get_gpuprops fetch = {
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
      } else if (key >= KBASE_GPUPROP_RAW_JS_FEATURES_0 &&
                key <= KBASE_GPUPROP_RAW_JS_FEATURES_15) {
         out->features[key - KBASE_GPUPROP_RAW_JS_FEATURES_0] =
            (uint32_t)value;
      }
   }

   free(blob);
   return 0;
}

/* Pick the first present job slot whose advertised JS_FEATURES cover every
 * bit in want_mask. Mirrors what kbase_js_choose_affinity()/mali_kbase_js.c
 * does kernel-side, just done once up front instead of per submission.
 *
 * Returns -1 if nothing matches -- which is a perfectly normal outcome (not
 * every board's JS_FEATURES layout puts vertex+tiler on the same slot, and
 * some kernels don't populate this property at all), not an error. See the
 * FIX note at the top of the file for why the caller must not treat -1 as
 * fatal. */
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

/* -----------------------------------------------------------------------
 * Atom submission via KBASE_IOCTL_JOB_SUBMIT, using the real
 * struct base_jd_atom_v2 from mali_base_jm_kernel.h.
 * ----------------------------------------------------------------------- */

/* Pick the next atom_number for this queue. base_jd_atom_v2::atom_number is
 * a userspace-owned value (see the file-level note above), out of the
 * BASE_JD_ATOM_COUNT (256) space defined in mali_base_jm_kernel.h. Cycles
 * through 1..255, skipping 0 since a pre_dep referencing atom_id 0 with
 * BASE_JD_DEP_TYPE_INVALID means "no dependency" -- this queue never has
 * more than one atom outstanding, so simple cycling can't collide with
 * anything still in flight. */
static uint8_t
panvk_kbase_next_atom_number(struct panvk_gpu_queue *queue)
{
   return (uint8_t)((queue->jm_last_atom % (BASE_JD_ATOM_COUNT - 1)) + 1);
}

/* Submit a single job chain as one kbase JM atom, chained onto the atom
 * this queue submitted last, and block until it (and therefore everything
 * submitted before it on this queue) has completed.
 *
 * jobslot may be -1, meaning "let the kernel choose" (see the FIX note at
 * the top of the file) -- BASE_JD_REQ_JOB_SLOT is only added to core_req
 * when the caller actually has a discovered slot to offer.
 *
 * See the big comment on panvk_gpu_queue::jm_last_atom for why this is
 * synchronous instead of returning a fence-like object: a kbase atom's
 * dependency can only reference prior atoms *on this same context*, and
 * completion is only observable by draining the shared JM event stream for
 * the device -- there's nothing DRM-syncobj-shaped to export or wait on
 * from outside this function.
 */
static bool
panvk_queue_jm_submit_atom(struct panvk_gpu_queue *queue,
                           base_jd_core_req core_req, int jobslot,
                           uint64_t jc)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);
   int fd = dev->kmod.dev->fd;

   uint8_t atom_number = panvk_kbase_next_atom_number(queue);

   if (jobslot >= 0)
      core_req |= BASE_JD_REQ_JOB_SLOT;

   struct base_jd_atom_v2 atom = {
      .jc = jc,
      .udata = {.blob = {0, 0}},
      .extres_list = 0,
      .nr_extres = 0,
      .jit_id = {0, 0}, /* must be zero, per mali_base_jm_kernel.h */
      .pre_dep =
         {
            {
               /* jm_last_atom == 0 means "nothing submitted on this queue
                * yet" -- see the port notes above. */
               .atom_id = queue->jm_last_atom,
               .dependency_type = queue->jm_last_atom
                                     ? BASE_JD_DEP_TYPE_DATA
                                     : BASE_JD_DEP_TYPE_INVALID,
            },
            {.atom_id = 0, .dependency_type = BASE_JD_DEP_TYPE_INVALID},
         },
      .atom_number = atom_number,
      .prio = BASE_JD_PRIO_MEDIUM,
      .device_nr = 0,
      .jobslot = jobslot >= 0 ? (uint8_t)jobslot : 0,
      .core_req = core_req,
      .padding = {0},
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
                   ") reported failure (event_code=0x%x)",
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
      /* BASE_JD_REQ_CS covers Vertex/Geometry/Compute Shader jobs;
       * BASE_JD_REQ_T is tiling -- both per their doc comments in
       * mali_base_jm_kernel.h. This job chain contains vertex and tiler
       * jobs. queue->jm_vt_slot may be -1 (no single slot advertised both
       * capabilities in JS_FEATURES, or the property wasn't available) --
       * panvk_queue_jm_submit_atom() treats that as "let the kernel pick",
       * which is the normal/safe default. */
      if (!panvk_queue_jm_submit_atom(queue, BASE_JD_REQ_CS | BASE_JD_REQ_T,
                                      queue->jm_vt_slot,
                                      batch->vtc_jc.first_job))
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
      /* BASE_JD_REQ_FS: "Requires fragment shaders". Same -1-means-auto
       * handling as above applies to queue->jm_frag_slot. */
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
 * soft-event-wait atom we submit here, so we just block the CPU on the
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

/* Event set/reset is a direct KBASE_IOCTL_SOFT_EVENT_UPDATE call (not an
 * atom -- see the file-level note), implemented in panvk_vX_event.c via
 * panvk_per_arch(event_update)(); out of scope for this file. */
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

   /* struct base_jd_atom_v2::prio (base_jd_prio) is per-atom, not
    * negotiated once at queue/context creation time, so a non-MEDIUM
    * global priority isn't plumbed through yet. base_jd_prio does have
    * BASE_JD_PRIO_HIGH/LOW/REALTIME levels available per-atom in
    * mali_base_jm_kernel.h if this needs extending later. */
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

   /* Job-slot routing is best-effort/diagnostic only -- see the FIX note
    * at the top of the file. A query or lookup failure here must NOT fail
    * device creation: struct base_jd_atom_v2::jobslot (and therefore
    * queue->jm_vt_slot/jm_frag_slot) is only ever consulted by the kernel
    * when BASE_JD_REQ_JOB_SLOT is set on the atom, which
    * panvk_queue_jm_submit_atom() only does when it has a slot >= 0 to
    * offer. Falling back to -1 ("let the kernel choose via
    * kbase_js_choose_affinity()") is always safe. */
   queue->jm_vt_slot = -1;
   queue->jm_frag_slot = -1;

   struct panvk_kbase_js_features slots;
   int ret = panvk_kbase_get_js_features(device->kmod.dev->fd, &slots);
   if (ret) {
      mesa_logw("panvk: failed to query kbase JS_FEATURES via "
               "KBASE_IOCTL_GET_GPUPROPS: %s -- falling back to "
               "kernel-chosen job-slot affinity", strerror(errno));
   } else {
      mesa_logd("panvk: kbase RAW_JS_PRESENT=0x%x", slots.slot_present_mask);
      for (uint32_t i = 0; i < PANVK_KBASE_MAX_JOB_SLOTS; i++) {
         if (slots.slot_present_mask & (1u << i))
            mesa_logd("panvk: kbase JS_FEATURES[%u]=0x%x", i,
                      slots.features[i]);
      }

      queue->jm_vt_slot = panvk_kbase_pick_job_slot(
         &slots, JS_FEATURE_VERTEX_JOB | JS_FEATURE_TILER_JOB);
      queue->jm_frag_slot =
         panvk_kbase_pick_job_slot(&slots, JS_FEATURE_FRAGMENT_JOB);

      if (queue->jm_vt_slot < 0 || queue->jm_frag_slot < 0) {
         mesa_logd("panvk: no single kbase JM job slot advertises the "
                   "requested vertex+tiler/fragment JS_FEATURES combo -- "
                   "falling back to kernel-chosen affinity for the slot(s) "
                   "that didn't match (this is normal on some boards)");
      }
   }

   queue->vk.driver_submit = panvk_per_arch(gpu_queue_submit);
   *out_queue = &queue->vk;
   return VK_SUCCESS;

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

   /* KBASE_IOCTL_POST_TERM (ioctl nr 4, _IO with no payload), from
    * mali_kbase_jm_ioctl.h. Best-effort: the queue is being torn down
    * regardless of whether this succeeds. */
   ioctl(dev->kmod.dev->fd, KBASE_IOCTL_POST_TERM);

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
