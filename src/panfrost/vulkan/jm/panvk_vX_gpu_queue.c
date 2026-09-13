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

/*
 * ---------------------------------------------------------------------
 * kbase JM (Job Manager) job submission -- self-contained, no shared
 * mesa/kbase headers
 * ---------------------------------------------------------------------
 *
 * Earlier versions of this file pulled the kbase ioctl definitions in
 * from shared headers (kbase_jm.h / kbase_kmod.h / mali_base_jm_kernel.h
 * / mali_base_csf_kernel.h, ...). In this tree those headers step on each
 * other in ways that are very sensitive to include order and to exactly
 * which other file in the same binary pulled which one in first (macro
 * and struct-tag collisions between the JM and CSF variants, a header
 * that silently produced no declarations for reasons we couldn't pin
 * down, etc). Chasing that across several rounds of CI cost more time
 * than the kbase ioctl surface used here is actually worth: this file
 * only needs 2 ioctls (VERSION_CHECK, JOB_SUBMIT) and reads one small,
 * fixed-layout event struct.
 *
 * So: everything this file needs from the kbase uAPI is declared right
 * here, under a "panvk_kbase_" prefix that cannot collide with anything
 * any other header in this translation unit defines. No external kbase
 * header is included.
 *
 * IMPORTANT -- atom struct is base_jd_atom (v3), NOT base_jd_atom_v2:
 * an earlier revision of this file submitted the 56-byte
 * base_jd_atom_v2 layout (no seq_nr) and every KBASE_IOCTL_JOB_SUBMIT
 * came back EINVAL. Confirmed on-device with a minimal standalone
 * ioctl test (UAPI negotiates as 11.46 on this kernel/DDK, i.e. the
 * "R54P1"-era job manager): switching the submitted struct to the
 * 64-byte layout with `seq_nr` as its first member (`jc` moves to
 * offset 8) made JOB_SUBMIT succeed immediately. That 64-byte/seq_nr
 * layout is exactly `struct base_jd_atom` in mali_base_jm_kernel.h
 * ("Same as base_jd_atom_v2, but has an extra seq_nr at the beginning"
 * + a trailing renderpass_id byte where v2 just has more padding) --
 * so this DDK's JOB_SUBMIT only accepts the v3 shape.
 *
 * IMPORTANT #2 -- there is no per-atom job-slot override, don't invent one:
 * with the v3 atom fixed, JOB_SUBMIT started being *accepted*, but the
 * submitted atom then sat forever with no completion event and no
 * fault event either, until our own watchdog timed it out ("device is
 * either stuck or its watchdog didn't fire"). An earlier revision of
 * this file "fixed" that by adding a `jobslot` byte to the atom (right
 * after `device_nr`, before `core_req`) plus a made-up
 * BASE_JD_REQ_JOB_SLOT core_req bit (1 << 17), on the theory that
 * automatic slot routing was unreliable for a combined vertex+tiler
 * (CS|T) atom on this device/kernel, and that reading per-slot
 * JS_FEATURES via KBASE_IOCTL_GET_GPUPROPS and pinning the atom to a
 * matching slot would fix it.
 *
 * That theory doesn't hold up. Cross-checked against several real
 * kbase UAPI trees (mali_base_kernel.h / mali_base_jm_kernel.h across
 * multiple driver vintages, plus mali_kbase_js.c), there is no
 * per-atom job-slot-override field or flag anywhere in the JM ABI.
 * The byte the earlier revision repurposed as `jobslot` is, in every
 * real copy of this struct, reserved padding between `device_nr` and
 * `core_req` that must stay zero; no core_req bit at position 17 (or
 * anywhere past BASE_JD_REQ_SKIP_CACHE_END at bit 16) is defined
 * either. Job-slot assignment (JS0 = fragment, JS1/JS2 = vertex/
 * tiler/compute) is done entirely inside the kernel's own job
 * scheduler from the FS/CS/T/ONLY_COMPUTE bits of `core_req` -- that's
 * the only lever userspace actually has, and it's the one every other
 * kbase-based driver relies on. So the "pin to a slot" write was a
 * no-op at best (the kernel ignores both the unknown core_req bit and
 * the reserved byte) and an ABI violation (nonzero reserved field) at
 * worst -- either way it could never have changed which slot the
 * kernel picked. That's exactly consistent with the identical
 * silent-timeout symptom coming back unchanged after this "fix" was
 * added: it never did anything.
 *
 * The actual fix is to submit only the real FS / (CS|T) requirement
 * bits, leave the reserved byte at 0, and let the kernel's scheduler
 * route the atom -- the same as every other kbase client does. If
 * jobs still don't complete after that, the cause is elsewhere (jc
 * pointing at unmapped/incoherent memory, a job chain the shader
 * cores genuinely fault on that this DDK/kernel combo fails to
 * report, a GPU power-domain issue, etc.) and needs fresh on-device
 * triage rather than another guess at slot routing.
 *
 *   - KBASE_IOCTL_VERSION_CHECK   = _IOWR(0x80, 0, {u16 major, u16 minor})
 *   - KBASE_IOCTL_JOB_SUBMIT      = _IOW (0x80, 2, {u64 addr, u32 nr_atoms, u32 stride})
 *   - struct base_jd_atom   (v3, 64 bytes, see layout below) -- what we submit
 *   - struct base_jd_event_v2 (24 bytes: u32 event_code, u8 atom_number,
 *     u8 pad[3], u64 udata[2])
 *   - BASE_JD_EVENT_DONE = 0x01
 *
 *   GET_GPUPROPS is deliberately *not* used: see IMPORTANT #2 above --
 *   there's no per-atom job-slot override to look features up for.
 */

#include <errno.h>
#include <inttypes.h>
#include <poll.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

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

#include "vk_framebuffer.h"
#include "vk_sync.h"

#define PANVK_KBASE_IOCTL_TYPE 0x80

struct panvk_kbase_version_check {
   uint16_t major;
   uint16_t minor;
};
#define PANVK_KBASE_IOCTL_VERSION_CHECK \
   _IOWR(PANVK_KBASE_IOCTL_TYPE, 0, struct panvk_kbase_version_check)

struct panvk_kbase_job_submit {
   uint64_t addr;
   uint32_t nr_atoms;
   uint32_t stride;
};
#define PANVK_KBASE_IOCTL_JOB_SUBMIT \
   _IOW(PANVK_KBASE_IOCTL_TYPE, 2, struct panvk_kbase_job_submit)

struct panvk_kbase_dependency {
   uint8_t atom_id;
   uint8_t dependency_type;
};
#define PANVK_KBASE_DEP_TYPE_DATA 1

/* struct base_jd_atom (v3), byte-for-byte: 64 bytes total. This is
 * base_jd_atom_v2 with a `seq_nr` prepended and one byte of its trailing
 * padding repurposed as `renderpass_id` -- see the big comment at the
 * top of this file for why v3 (not v2) is what this kernel actually
 * accepts. Every field still falls on its natural alignment boundary in
 * this order, so no __attribute__((packed)) is needed on either aarch64
 * or x86_64. */
struct panvk_kbase_atom {
   uint64_t seq_nr;                        /* offset 0  */
   uint64_t jc;                            /* offset 8  */
   uint64_t udata[2];                      /* offset 16 (base_jd_udata) */
   uint64_t extres_list;                   /* offset 32 */
   uint16_t nr_extres;                     /* offset 40 */
   uint8_t jit_id[2];                      /* offset 42 */
   struct panvk_kbase_dependency pre_dep[2]; /* offset 44 */
   uint8_t atom_number;                    /* offset 48 */
   uint8_t prio;                           /* offset 49 */
   uint8_t device_nr;                      /* offset 50 */
   /* Reserved padding before core_req -- NOT a job-slot selector, and
    * not ours to repurpose. See IMPORTANT #2 at the top of this file.
    * Must always stay 0. */
   uint8_t reserved0;                      /* offset 51 */
   uint32_t core_req;                      /* offset 52 */
   uint8_t renderpass_id;                  /* offset 56 */
   uint8_t padding[7];                     /* offset 57 */
};                                          /* size 64 */
_Static_assert(sizeof(struct panvk_kbase_atom) == 64,
               "base_jd_atom (v3) must be 64 bytes");

#define PANVK_KBASE_JD_REQ_FS ((uint32_t)1 << 0) /* fragment job    */
#define PANVK_KBASE_JD_REQ_CS ((uint32_t)1 << 1) /* vertex/geom job */
#define PANVK_KBASE_JD_REQ_T  ((uint32_t)1 << 2) /* tiler job       */

#define PANVK_KBASE_JD_PRIO_MEDIUM 0

/* struct base_jd_event_v2, byte-for-byte: 24 bytes. Unlike the atom
 * struct, the event struct did NOT change shape between v2 and v3 --
 * only the submitted atom did -- so this is unaffected by that fix. */
struct panvk_kbase_event_v2 {
   uint32_t event_code;
   uint8_t atom_number;
   uint8_t padding[3];
   uint64_t udata[2];
};

#define PANVK_KBASE_JD_EVENT_DONE 0x01

/* --------------------------------------------------------------------- */

enum panvk_kbase_atom_kind {
   PANVK_KBASE_ATOM_VERTEX_TILER,
   PANVK_KBASE_ATOM_FRAGMENT,
};

/* Upper bound on how long we'll wait for a single JM atom to complete
 * before giving up and declaring the device lost, instead of blocking
 * vkQueueSubmit()/vkQueueWaitIdle() forever if a completion event never
 * shows up. Tunable via PANVK_KBASE_ATOM_TIMEOUT_MS for bisecting a
 * specific hang. */
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
 *
 * See the big comment on panvk_gpu_queue::jm_last_atom for why this is
 * synchronous instead of returning a fence-like object: a kbase atom's
 * pre_dep can only reference one prior atom_number *on this same
 * context*, and completion is only observable by draining a shared
 * poll()+read() base_jd_event_v2 stream on the device fd -- there's
 * nothing DRM-syncobj-shaped to export or wait on from outside this
 * function.
 */
static bool
panvk_queue_jm_submit_atom(struct panvk_gpu_queue *queue,
                           enum panvk_kbase_atom_kind kind, uint64_t jc)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);
   int fd = dev->kmod.dev->fd;

   /* Atom numbers are a u8 (BASE_JD_ATOM_COUNT == 256) and 0 is reserved
    * to mean "no dependency" in pre_dep, so cycle through 1..255. Since
    * this queue only ever has one atom in flight, there's no risk of
    * colliding with an atom that's still outstanding. */
   uint8_t atom_number = queue->jm_last_atom + 1;
   if (atom_number == 0)
      atom_number = 1;

   /* Which job slot this atom lands on (JS0 = fragment, JS1/JS2 =
    * vertex/tiler/compute) is entirely the kernel job scheduler's
    * decision, made from these two requirement bits alone -- see
    * IMPORTANT #2 at the top of this file. We don't, and can't, pick
    * the slot ourselves. */
   uint32_t core_req = kind == PANVK_KBASE_ATOM_FRAGMENT
                           ? PANVK_KBASE_JD_REQ_FS
                           : (PANVK_KBASE_JD_REQ_CS | PANVK_KBASE_JD_REQ_T);

   struct panvk_kbase_atom atom = {
      /* No logical grouping of atoms beyond ordinary pre_dep chaining,
       * so seq_nr just mirrors atom_number. */
      .seq_nr = atom_number,
      .jc = jc,
      .core_req = core_req,
      .atom_number = atom_number,
      .prio = PANVK_KBASE_JD_PRIO_MEDIUM,
      /* .reserved0 is left at 0 -- see the struct definition above. */
      /* renderpass_id is only meaningful with BASE_JD_REQ_START/END_RENDERPASS,
       * which we don't use (no JM incremental rendering here) -- 0 is
       * "not part of a renderpass". */
      .renderpass_id = 0,
   };

   if (queue->jm_last_atom != 0) {
      atom.pre_dep[0].atom_id = queue->jm_last_atom;
      atom.pre_dep[0].dependency_type = PANVK_KBASE_DEP_TYPE_DATA;
   }

   struct panvk_kbase_job_submit submit = {
      .addr = (uintptr_t)&atom,
      .nr_atoms = 1,
      .stride = sizeof(atom),
   };

   if (ioctl(fd, PANVK_KBASE_IOCTL_JOB_SUBMIT, &submit)) {
      mesa_loge("panvk: KBASE_IOCTL_JOB_SUBMIT failed: %s", strerror(errno));
      return false;
   }

   queue->jm_last_atom = atom_number;

   /* Only one atom is ever in flight at a time in this submission model,
    * so the next event for *this atom_number* is the one we're waiting
    * for; anything else read off the stream first (there shouldn't be
    * anything else, but the ABI doesn't promise it) is skipped. The whole
    * loop is bounded by an absolute deadline instead of giving poll() a
    * fresh full timeout every round, so a device that keeps handing us
    * events for other atom numbers can't turn this into an unbounded
    * wait -- this is what fixes vkQueueSubmit()/vkQueueWaitIdle() hanging
    * forever when a completion event never arrives. */
   int64_t deadline_ns = os_time_get_nano() + panvk_kbase_atom_timeout_ns();

   for (;;) {
      int64_t now_ns = os_time_get_nano();
      int64_t remaining_ns = deadline_ns - now_ns;
      if (remaining_ns <= 0) {
         mesa_loge("panvk: timed out waiting for kbase JM atom %u (job "
                   "chain 0x%" PRIx64 ") to complete -- device is either "
                   "stuck or its watchdog didn't fire; treating this "
                   "queue as lost instead of hanging forever",
                   atom_number, jc);
         return false;
      }

      struct pollfd pfd = {.fd = fd, .events = POLLIN};
      int pret = poll(&pfd, 1, (int)(remaining_ns / 1000000));
      if (pret < 0) {
         if (errno == EINTR)
            continue;
         mesa_loge("panvk: poll() on kbase fd failed: %s", strerror(errno));
         return false;
      }
      if (pret == 0)
         continue; /* timed out this round; deadline check above catches
                     * the overall timeout on the next iteration */

      struct panvk_kbase_event_v2 evt;
      ssize_t n = read(fd, &evt, sizeof(evt));
      if (n < 0) {
         if (errno == EINTR || errno == EAGAIN)
            continue;
         mesa_loge("panvk: read() on kbase fd failed: %s", strerror(errno));
         return false;
      }
      if (n == 0 || (size_t)n < sizeof(evt))
         continue;

      if (evt.atom_number != atom_number)
         continue;

      if (evt.event_code != PANVK_KBASE_JD_EVENT_DONE) {
         mesa_loge("panvk: kbase JM atom %u (job chain 0x%" PRIx64
                   ") reported failure (event_code 0x%x)",
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
      if (!panvk_queue_jm_submit_atom(queue, PANVK_KBASE_ATOM_VERTEX_TILER,
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
      if (!panvk_queue_jm_submit_atom(queue, PANVK_KBASE_ATOM_FRAGMENT,
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

/* TODO(verify): struct panvk_event (panvk_event.h) only carries a plain
 * uint32_t `syncobj` field -- there is no panvk_per_arch(event_is_set)/
 * (event_update)() declared anywhere in this codebase, so a version of
 * this file calling them could never have linked. Until
 * panvk_vX_event.c's real mechanism for that field is confirmed, this
 * treats `syncobj` directly as a plain 0/1 flag via atomic builtins, so
 * at least sets/waits within this process are data-race-free. This does
 * NOT yet know how panvk_vX_event.c / panvk_vX_cmd_event.c expect
 * `syncobj` to be used (it may need to be a real kernel syncobj handle
 * instead), so treat vkSetEvent/vkResetEvent/vkCmdWaitEvents as
 * unverified until that file is checked against this. */
static bool
panvk_kbase_event_is_set(struct panvk_event *event)
{
   return __atomic_load_n(&event->syncobj, __ATOMIC_ACQUIRE) != 0;
}

static void
panvk_kbase_event_set(struct panvk_event *event, bool set)
{
   __atomic_store_n(&event->syncobj, set ? 1u : 0u, __ATOMIC_RELEASE);
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

      while (!panvk_kbase_event_is_set(op->event)) {
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
   util_dynarray_foreach(&batch->event_ops, struct panvk_cmd_event_op, op) {
      switch (op->type) {
      case PANVK_EVENT_OP_SET:
         panvk_kbase_event_set(op->event, true);
         break;
      case PANVK_EVENT_OP_RESET:
         panvk_kbase_event_set(op->event, false);
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

   /* A kbase atom's pre_dep can only chain onto prior atoms *on this same
    * queue*; there's no way to hand it an external semaphore as a
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

   /* XXX: struct base_jd_atom only carries a per-atom BASE_JD_PRIO_*
    * value, not a queue-wide priority negotiated at creation time, so we
    * don't plumb anything beyond MEDIUM through yet. */
   assert(priority == VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR);

   struct panvk_gpu_queue *queue =
      vk_zalloc(&device->vk.alloc, sizeof(*queue), 8,
               VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!queue)
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result =
      vk_queue_init(&queue->vk, &device->vk, create_info, queue_idx);
   if (result != VK_SUCCESS)
      goto err_free_queue;

   /* This MUST run on every successful path out of this function: it's
    * what wires panvk_per_arch(gpu_queue_submit)() into the generic
    * Vulkan-runtime queue-submission dispatcher. If it's ever skipped,
    * the very first vkQueueSubmit() calls through a NULL function
    * pointer and crashes before issuing a single ioctl. */
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
