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

#include "vk_framebuffer.h"
#include "vk_sync.h"

#include "../../lib/kmod/kbase_jm.h"

/* Submit a single job chain as one kbase JM atom, chained onto the atom
 * this queue submitted last, and block until it (and therefore everything
 * submitted before it on this queue) has completed.
 *
 * See the big comment on panvk_gpu_queue::jm_last_atom for why this is
 * synchronous instead of returning a fence-like object: kbase_jm_atom_desc
 * only lets an atom depend on one prior atom on the *same* context, and
 * completion is only observable by draining a shared poll()+read() event
 * stream on the device fd -- there's nothing DRM-syncobj-shaped to export
 * or wait on from outside this function.
 */
static bool
panvk_queue_jm_submit_atom(struct panvk_gpu_queue *queue,
                           enum kbase_jm_atom_kind kind, uint64_t jc)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);

   struct kbase_jm_atom_desc desc = {
      .jc = jc,
      .kind = kind,
      .priority = KBASE_JM_PRIO_MEDIUM,
      .jobslot = -1, /* let kbase_jm pick a slot advertising the JSn
                       * features this atom kind needs */
      .depends_on_atom = queue->jm_last_atom,
   };

   int atom_id = kbase_jm_atom_submit(dev->kmod.dev, &desc);
   if (atom_id < 0) {
      mesa_loge("panvk: kbase_jm_atom_submit() failed: %s", strerror(errno));
      return false;
   }

   queue->jm_last_atom = (uint8_t)atom_id;

   /* Only one atom is ever in flight at a time in this submission model, so
    * the next JM event this queue observes has to be the one for the atom
    * we just submitted. */
   uint8_t completed_atom;
   bool succeeded;
   int ret = kbase_jm_wait_event(dev->kmod.dev, -1, &completed_atom,
                                 &succeeded);
   if (ret < 0) {
      mesa_loge("panvk: kbase_jm_wait_event() failed: %s", strerror(errno));
      return false;
   }

   assert(completed_atom == (uint8_t)atom_id);

   if (!succeeded) {
      mesa_loge("panvk: kbase JM atom %u (job chain 0x%" PRIx64
                ") reported failure", completed_atom, jc);
      return false;
   }

   return true;
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

   /* Unlike DRM_IOCTL_PANFROST_SUBMIT, kbase JM atoms don't take an
    * explicit BO handle list: all BOs live in the single VM bound to this
    * kbase context already, so there's nothing to pass here beyond the job
    * chain pointer itself. */
   if (batch->vtc_jc.first_job) {
      if (!panvk_queue_jm_submit_atom(queue, KBASE_JM_ATOM_VERTEX_TILER,
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

/* vkCmdWaitEvents2() operations recorded on this batch. There's no GPU-side
 * soft-event-wait atom exposed by kbase_jm.h, so -- same spirit as the
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

   /* kbase_jm_atom_desc can only chain an atom onto one prior atom *on this
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

   /* XXX: kbase_jm_atom_desc only carries a per-atom BASE_JD_PRIO_* value
    * (see kbase_jm_base_prio()), not a queue-wide priority negotiated at
    * creation time, so we don't plumb anything beyond MEDIUM through yet. */
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
