/*
 * Copyright © 2021 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_QUEUE_H
#define PANVK_QUEUE_H

#ifndef PAN_ARCH
#error "PAN_ARCH must be defined"
#endif

#include <stdint.h>

#include "panvk_device.h"

#include "vk_queue.h"

struct panvk_gpu_queue {
   struct vk_queue vk;

   /* base_jd_atom_v2::atom_number of the most recently submitted kbase
    * JM atom on this queue (0 == none submitted yet). Fed into the next
    * atom's pre_dep[0] (as a BASE_JD_DEP_TYPE_DATA dependency), so
    * completion order always matches submission order.
    *
    * The real kbase ABI only lets an atom depend on up to two prior
    * atoms *by atom_number, within this same context*, and completion
    * is only ever reported asynchronously through a poll()+read() event
    * stream (struct base_jd_event_v2) on the device fd -- there is no
    * DRM syncobj/fence to hook external waiters into. Rather than run a
    * background thread to demultiplex that event stream, this queue
    * uses a fully synchronous submission model:
    * panvk_per_arch(gpu_queue_submit)() blocks until every atom it
    * submits has completed before returning. That keeps this struct
    * (and QueueWaitIdle) trivial, at the cost of not overlapping
    * consecutive vkQueueSubmit()s on this queue. If that overlap ever
    * matters, this is the field to grow into a proper in-flight
    * tracker.
    *
    * This also assumes a single panvk_gpu_queue per VkDevice: Mali JM
    * hardware only exposes one job-manager submission context per kbase
    * fd, matching the existing "no queue priorities" limitation in
    * panvk_per_arch(create_gpu_queue)().
    */
   uint8_t jm_last_atom;

   /* Job slot index (into this device's kbase JS_FEATURES table) that
    * vertex+tiler atoms are routed to. Resolved once at queue creation
    * time in panvk_per_arch(create_gpu_queue)() via
    * panvk_kbase_pick_job_slot(), since slot routing is now an explicit
    * per-atom field (struct kbase_jm_atom_desc::jobslot) instead of
    * being implied by core_req flags the way the raw kbase ioctl ABI
    * used to work. A negative value should never be observed outside of
    * queue construction: create_gpu_queue() fails device creation if no
    * slot advertising the required features is found. */
   int jm_vt_slot;

   /* Same as jm_vt_slot, but for fragment atoms (KBASE_JM_ATOM_FRAGMENT).
    * Picked independently since vertex+tiler and fragment work can (and
    * typically do) land on different job slots. */
   int jm_frag_slot;
};

VK_DEFINE_HANDLE_CASTS(panvk_gpu_queue, vk.base, VkQueue, VK_OBJECT_TYPE_QUEUE)

VkResult panvk_per_arch(create_gpu_queue)(
   struct panvk_device *device, const VkDeviceQueueCreateInfo *create_info,
   uint32_t queue_idx, struct vk_queue **out_queue);
void panvk_per_arch(destroy_gpu_queue)(struct vk_queue *vk_queue);
VkResult panvk_per_arch(gpu_queue_submit)(struct vk_queue *vk_queue,
                                          struct vk_queue_submit *vk_submit);
VkResult panvk_per_arch(gpu_queue_check_status)(struct vk_queue *vk_queue);

#endif
