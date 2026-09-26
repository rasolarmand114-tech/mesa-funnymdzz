/*
 * Copyright © 2024 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "bifrost/bifrost_compile.h"
#include "pan_desc.h"
#include "pan_encoder.h"
#include "panvk_cmd_alloc.h"
#include "panvk_cmd_buffer.h"
#include "panvk_cmd_precomp.h"
#include "panvk_macros.h"
#include "panvk_mempool.h"
#include "panvk_precomp_cache.h"

#if PAN_ARCH < 9
void
panvk_per_arch(dispatch_precomp)(struct panvk_precomp_ctx *ctx,
                                 struct panlib_precomp_grid grid,
                                 enum panlib_barrier barrier,
                                 enum libpan_shaders_program idx, void *data,
                                 size_t data_size)
{
   ASSERTED enum panlib_barrier supported_barriers =
      PANLIB_BARRIER_JM_BARRIER | PANLIB_BARRIER_JM_SUPPRESS_PREFETCH;
   assert(!(barrier & ~supported_barriers) && "Unsupported barrier flags");

   struct panvk_cmd_buffer *cmdbuf = ctx->cmdbuf;

   /* Make sure we have a batch opened to queue our COMPUTE job to. */
   if (!cmdbuf->cur_batch)
      panvk_per_arch(cmd_open_batch)(cmdbuf);

   struct panvk_batch *batch = cmdbuf->cur_batch;
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   const struct panvk_shader_variant *shader =
      panvk_per_arch(precomp_cache_get)(dev->precomp_cache, idx);

   assert(shader);
   assert(batch && "Need current batch to be present!");

   struct pan_ptr push_uniforms = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, BIFROST_PRECOMPILED_KERNEL_SYSVALS_SIZE + data_size, 16);

   assert(push_uniforms.gpu);

   struct bifrost_precompiled_kernel_sysvals sysvals;
   sysvals.num_workgroups.x = grid.count[0];
   sysvals.num_workgroups.y = grid.count[1];
   sysvals.num_workgroups.z = grid.count[2];
   sysvals.printf_buffer_address = dev->printf.bo->addr.dev;

   bifrost_precompiled_kernel_prepare_push_uniforms(push_uniforms.cpu, data,
                                                    data_size, &sysvals);

   struct pan_ptr job = panvk_cmd_alloc_desc(cmdbuf, COMPUTE_JOB);
   assert(job.gpu);

   pan_pack_work_groups_compute(
      pan_section_ptr(job.cpu, COMPUTE_JOB, INVOCATION), grid.count[0],
      grid.count[1], grid.count[2], shader->cs.local_size.x,
      shader->cs.local_size.y, shader->cs.local_size.z, false, false);

   pan_section_pack(job.cpu, COMPUTE_JOB, PARAMETERS, cfg) {
      cfg.job_task_split = util_logbase2_ceil(shader->cs.local_size.x + 1) +
                           util_logbase2_ceil(shader->cs.local_size.y + 1) +
                           util_logbase2_ceil(shader->cs.local_size.z + 1);
   }

   struct pan_compute_dim dim = {.x = grid.count[0],
                                 .y = grid.count[1],
                                 .z = grid.count[2]};
   uint64_t tld =
      panvk_per_arch(cmd_dispatch_prepare_tls)(cmdbuf, shader, &dim, false);
   assert(tld);

   pan_section_pack(job.cpu, COMPUTE_JOB, DRAW, cfg) {
      cfg.state = panvk_priv_mem_dev_addr(shader->rsd),
      cfg.push_uniforms = push_uniforms.gpu;
      cfg.thread_storage = tld;
   }

   util_dynarray_append(&batch->jobs, job.cpu);

   bool job_barrier = (barrier & PANLIB_BARRIER_JM_BARRIER) != 0;
   bool suppress_prefetch =
      (barrier & PANLIB_BARRIER_JM_SUPPRESS_PREFETCH) != 0;

   pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, job_barrier,
                  suppress_prefetch, grid.jm.local_dep, grid.jm.global_dep,
                  &job, false);
}
#else
/* TODO v9: precompiled-kernel dispatch not yet ported.
 *
 * Same root cause as jm/panvk_vX_cmd_dispatch.c: this packs COMPUTE_JOB's
 * old INVOCATION/PARAMETERS/DRAW sections directly, which don't exist in
 * that shape at PAN_ARCH >= 9 (confirmed by this exact build error).
 *
 * This one is NOT handled the same way as cmd_dispatch(), though:
 *   - csf/panvk_vX_cmd_precomp.c is not a usable reference here. It
 *     doesn't pack a job-descriptor struct at all -- CSF has no job
 *     descriptors, it writes cs_update_compute_ctx()/cs_move*_to()
 *     register-move instructions straight into a command stream. So it
 *     can't reveal what v9's actual JM "Compute Job/Compute Payload"
 *     section layout is; that's JM-only and still needs the real v9
 *     genxml, which isn't available in this source subset.
 *   - dispatch_precomp() is shared panlib infrastructure (see
 *     panvk_cmd_precomp.h's MESA_DISPATCH_PRECOMP), called generically
 *     for precompiled meta kernels (clears/copies/etc.) from panlib code
 *     that isn't part of this source subset either, so unlike a single
 *     VkCmd* entry point its v9 call sites can't be fully enumerated
 *     here. A silent no-op risks some operation completing as if its
 *     GPU work ran when it didn't -- worse than a compile error. This
 *     asserts instead, so any v9 path that still needs it fails loudly
 *     and points straight back here instead of producing quietly wrong
 *     results. If that's not what you want while bringing more of v9 up
 *     (e.g. you'd rather keep going and treat this as inert for now),
 *     drop the assert() and leave the function body empty -- same
 *     no-op convention as CmdDrawIndirect/cmd_dispatch()'s v9 stub.
 *
 * To actually port this: get the real v9 COMPUTE_JOB section names,
 * e.g. `grep -A30 'struct name="Compute Job"' src/panfrost/lib/genxml/v9.xml`
 * or `grep COMPUTE_JOB build/.../genxml/genxml/v9_pack.h` in your tree
 * (the exact generated-header path depends on your build dir layout) --
 * that tells us what sections/fields replace INVOCATION/PARAMETERS/DRAW
 * for v9, and this can be written for real instead of stubbed. */
void
panvk_per_arch(dispatch_precomp)(struct panvk_precomp_ctx *ctx,
                                 struct panlib_precomp_grid grid,
                                 enum panlib_barrier barrier,
                                 enum libpan_shaders_program idx, void *data,
                                 size_t data_size)
{
   assert(!"panvk_v9_dispatch_precomp: not yet implemented (see comment above)");
}
#endif
