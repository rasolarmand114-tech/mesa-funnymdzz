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
#include <stdlib.h>
#include <string.h>

#include "util/log.h"
#include "util/os_misc.h"
#include "util/os_time.h"

#include "genxml/gen_macros.h"

#include "decode.h"

#include "panvk_cmd_buffer.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_event.h"
#include "panvk_image.h"
#include "panvk_image_view.h"
#include "panvk_instance.h"

#include "panvk_physical_device.h"
#include "panvk_priv_bo.h"
#include "panvk_queue.h"
#include "../../lib/kmod/kbase_jm.h"
#include "vk_framebuffer.h"
#include "vk_sync.h"

/*
 * This file is the whole of panvk_per_arch(gpu_queue_submit)() and friends
 * for the JM architectures (v6/v7/v9 -- Job Manager, arch <= 9). A CSF
 * architecture (v10+) is compiled from a different source file against
 * kbase_kmod_csf_*() in kbase_kmod.h and never reaches this one. That split
 * is why this file only ever calls kbase_jm_*() (from kbase_jm.h): the two
 * submission models are not interchangeable, and every kbase_jm_*() call
 * already refuses to run on a non-JM device on its own (see
 * KBASE_JM_REQUIRE_JM_DEVICE in kbase_jm.c). The two checks in this file
 * (panvk_kbase_require_jm() below) are a second, file-level line of
 * defence on top of that, specifically written as ordinary runtime checks
 * rather than assert() -- an assert() disappears entirely in an NDEBUG
 * (release) build, which is exactly the build where silently treating a
 * CSF device as a JM one would be most dangerous and least likely to be
 * noticed.
 */

/* Refuses to continue if @dev is not conclusively a kbase JM device.
 * Always active (no NDEBUG/assert escape hatch). Logs loudly because
 * tripping this should never be silent. */
static bool
panvk_kbase_require_jm(struct pan_kmod_dev *kdev, const char *where)
{
   if (kbase_gfx_dev_kind(kdev) == KBASE_GFX_DEV_JM)
      return true;

   mesa_loge("panvk: %s reached on a non-JM kbase device -- refusing to "
             "run the JM submission path on it", where);
   return false;
}

/* Pick the first job slot whose advertised JS_FEATURES cover every bit in
 * want_mask. Mirrors what kbase_js_choose_affinity()/mali_kbase_js.c does
 * kernel-side, just done once up front instead of per submission. */
static int
panvk_kbase_pick_job_slot(const struct kbase_jm_job_slot_info *slots,
                           uint32_t want_mask)
{
   for (uint32_t i = 0; i < slots->slot_count; i++) {
      if ((slots->features[i] & want_mask) == want_mask)
         return (int)i;
   }

   return -1;
}

/* Upper bound on how long we'll wait for a single JM atom to complete
 * before giving up and declaring the device lost, instead of blocking
 * vkQueueSubmit()/vkQueueWaitIdle() forever. Tunable via
 * PANVK_KBASE_ATOM_TIMEOUT_MS for bisecting a specific hang. */
static int64_t
panvk_kbase_atom_timeout_ns(void)
{
   static int64_t cached_ns = -2; /* -2: not yet computed */

   if (cached_ns != -2)
      return cached_ns;

   int64_t timeout_ms = 10000;
   const char *env = os_get_option("PANVK_KBASE_ATOM_TIMEOUT_MS");
   if (env && env[0]) {
      char *end = NULL;
      long parsed = strtol(env, &end, 10);
      if (end && *end == '\0' && parsed > 0)
         timeout_ms = parsed;
      else
         mesa_logw("panvk: ignoring invalid PANVK_KBASE_ATOM_TIMEOUT_MS=%s",
                   env);
   }

   cached_ns = timeout_ms * 1000000ll;
   return cached_ns;
}

/* Submit a single job chain as one kbase JM atom, chained onto the atom
 * this queue submitted last, and block until it (and therefore everything
 * submitted before it on this queue) has completed or the wait times out.
 */
static bool
panvk_queue_jm_submit_atom(struct panvk_gpu_queue *queue,
                           enum kbase_jm_atom_kind kind, uint64_t jc)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);
   struct pan_kmod_dev *kdev = dev->kmod.dev;

   if (!panvk_kbase_require_jm(kdev, "panvk_queue_jm_submit_atom"))
      return false;

   int jobslot = kind == KBASE_JM_ATOM_FRAGMENT ? queue->jm_frag_slot
                                                 : queue->jm_vt_slot;

   struct kbase_jm_atom_desc desc = {
      .jc = jc,
      .kind = kind,
      .priority = KBASE_JM_PRIO_MEDIUM,
      .jobslot = jobslot,
      .depends_on_atom = queue->jm_last_atom,
   };

   int ret = kbase_jm_atom_submit(kdev, &desc);
   if (ret < 0) {
      mesa_loge("panvk: kbase_jm_atom_submit failed: %s", strerror(errno));
      return false;
   }

   uint8_t atom_number = (uint8_t)ret;
   queue->jm_last_atom = atom_number;

   int64_t deadline_ns = os_time_get_nano() + panvk_kbase_atom_timeout_ns();

   for (;;) {
      int64_t now_ns = os_time_get_nano();
      if (now_ns >= deadline_ns) {
         mesa_loge("panvk: timed out waiting for kbase JM atom %u (job "
                   "chain 0x%" PRIx64 ") to complete -- device is either "
                   "stuck or its watchdog didn't fire; treating this queue "
                   "as lost instead of hanging vkQueueSubmit forever",
                   atom_number, jc);
         return false;
      }

      uint8_t evt_atom_number;
      bool succeeded;
      int wret = kbase_jm_wait_event(kdev, deadline_ns - now_ns,
                                     &evt_atom_number, &succeeded);
      if (wret < 0) {
         mesa_loge("panvk: kbase_jm_wait_event failed: %s", strerror(errno));
         return false;
      }
      if (wret == 0)
         continue;

      if (evt_atom_number != atom_number)
         continue;

      if (!succeeded) {
         mesa_loge("panvk: kbase JM atom %u (job chain 0x%" PRIx64
                   ") reported failure",
                   atom_number, jc);
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

   if (batch->issued) {
      util_dynarray_foreach(&batch->jobs, void *, job)
         memset((*job), 0, 4 * 4);

      if (batch->tiler.ctx_descs.cpu) {
         memcpy(batch->tiler.heap_desc.cpu, &batch->tiler.heap_templ,
                sizeof(batch->tiler.heap_templ));

         struct mali_tiler_context_packed *ctxs = batch->tiler.ctx_descs.cpu;

         for (uint32_t i = 0; i < batch->fb.layer_count; i++)
            memcpy(&ctxs[i], &batch->tiler.ctx_templ, sizeof(*ctxs));
      }

      panvk_pool_flush_maps(&cmdbuf->desc_pool);
   }

   pan_kmod_flush_bo_map_syncs(dev->kmod.dev);

   if (batch->vtc_jc.first_job) {
      if (!panvk_queue_jm_submit_atom(queue, KBASE_JM_ATOM_VERTEX_TILER,
                                      batch->vtc_jc.first_job))
         return false;

      if (PANVK_DEBUG(TRACE) || PANVK_DEBUG(SYNC)) {
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
      if (!panvk_queue_jm_submit_atom(queue, KBASE_JM_ATOM_FRAGMENT,
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

static VkResult
panvk_queue_signal_events(struct panvk_gpu_queue *queue,
                          struct panvk_batch *batch)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);

   util_dynarray_foreach(&batch->event_ops, struct panvk_cmd_event_op, op) {
      switch (op->type) {
      case PANVK_EVENT_OP_SET:
         if (!panvk_per_arch(event_update)(dev, op->event,
                                           BASE_JD_SOFT_EVENT_SET))
            return VK_ERROR_DEVICE_LOST;
         break;
      case PANVK_EVENT_OP_RESET:
         if (!panvk_per_arch(event_update)(dev, op->event,
                                           BASE_JD_SOFT_EVENT_RESET))
            return VK_ERROR_DEVICE_LOST;
         break;
      case PANVK_EVENT_OP_WAIT:
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

   assert(priority == VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR);

   if (!panvk_kbase_require_jm(device->kmod.dev,
                               "panvk_per_arch(create_gpu_queue)")) {
      return panvk_errorf(device, VK_ERROR_INITIALIZATION_FAILED,
                          "this panvk build's JM gpu_queue was asked to "
                          "create a queue on a non-JM kbase device");
   }

   struct panvk_gpu_queue *queue =
      vk_zalloc(&device->vk.alloc, sizeof(*queue), 8,
               VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!queue)
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result =
      vk_queue_init(&queue->vk, &device->vk, create_info, queue_idx);
   if (result != VK_SUCCESS)
      goto err_free_queue;

   struct kbase_jm_job_slot_info slots;
   if (kbase_jm_query_job_slots(device->kmod.dev, &slots)) {
      result = panvk_errorf(device, VK_ERROR_INITIALIZATION_FAILED,
                            "failed to query kbase JM job slots: %s",
                            strerror(errno));
      goto err_finish_queue;
   }

   queue->jm_vt_slot = panvk_kbase_pick_job_slot(
      &slots, KBASE_JM_JSn_FEATURE_VERTEX | KBASE_JM_JSn_FEATURE_TILER);
   queue->jm_frag_slot =
      panvk_kbase_pick_job_slot(&slots, KBASE_JM_JSn_FEATURE_FRAGMENT);

   if (queue->jm_vt_slot < 0 || queue->jm_frag_slot < 0) {
      result = panvk_errorf(device, VK_ERROR_INITIALIZATION_FAILED,
                            "no kbase JM job slot advertises the required "
                            "vertex+tiler/fragment JS_FEATURES");
      goto err_finish_queue;
   }

   /* This line MUST run on every successful path out of this function.
    * If it doesn't, vk_queue_submit() (generic Vulkan runtime code) is
    * left calling through a NULL driver_submit function pointer on the
    * very first vkQueueSubmit() -- which is exactly the kind of
    * "SIGSEGV right at vkQueueSubmit, before any ioctl" crash a NULL
    * driver_submit produces. */
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

   kbase_jm_post_term(dev->kmod.dev);

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

   assert(queue->vk.submit.mode != VK_QUEUE_SUBMIT_MODE_THREADED);

   if (vk_device_is_lost(&dev->vk)) {
      u_printf_with_ctx(stdout, &dev->printf.ctx);
      return VK_ERROR_DEVICE_LOST;
   }

   return VK_SUCCESS;
}
