/*
 * SPDX-License-Identifier: MIT
 *
 * kbase_jm.h — Public interface for the Job Manager (JM) command-submission
 * add-on to the kbase kmod backend (kbase_kmod.c / kbase_kmod.h).
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct pan_kmod_dev;

enum kbase_gfx_dev_kind {
   KBASE_GFX_DEV_UNKNOWN = 0,
   KBASE_GFX_DEV_CSF,
   KBASE_GFX_DEV_JM,
};

enum kbase_gfx_dev_kind kbase_gfx_dev_kind(const struct pan_kmod_dev *dev);

#define KBASE_JM_MAX_JOB_SLOTS 16

#define KBASE_JM_JSn_FEATURE_NULL           (1u << 1)
#define KBASE_JM_JSn_FEATURE_VERTEX         (1u << 2)
#define KBASE_JM_JSn_FEATURE_TILER          (1u << 7)
#define KBASE_JM_JSn_FEATURE_FRAGMENT       (1u << 12)
#define KBASE_JM_JSn_FEATURE_COMPUTE        (1u << 10)

struct kbase_jm_job_slot_info {
   uint32_t slot_count;
   uint32_t features[KBASE_JM_MAX_JOB_SLOTS];
};

int kbase_jm_query_job_slots(struct pan_kmod_dev *dev,
                              struct kbase_jm_job_slot_info *out);

enum kbase_jm_atom_kind {
   KBASE_JM_ATOM_VERTEX_TILER,
   KBASE_JM_ATOM_FRAGMENT,
   KBASE_JM_ATOM_COMPUTE,
};

enum kbase_jm_atom_priority {
   KBASE_JM_PRIO_REALTIME = 0,
   KBASE_JM_PRIO_HIGH,
   KBASE_JM_PRIO_MEDIUM,
   KBASE_JM_PRIO_LOW,
};

struct kbase_jm_atom_desc {
   uint64_t jc;
   enum kbase_jm_atom_kind kind;
   enum kbase_jm_atom_priority priority;
   int jobslot;
   uint8_t depends_on_atom;
};

int kbase_jm_atom_submit(struct pan_kmod_dev *dev,
                          const struct kbase_jm_atom_desc *desc);

/* Blocks (up to timeout_ns, or forever if negative) until at least one JM
 * completion event is available, then reads and reports it. Returns 1 and
 * fills out_atom_number / out_succeeded when an event was consumed, 0 on
 * timeout, -1 (errno set) on error / non-JM device. */
int kbase_jm_wait_event(struct pan_kmod_dev *dev, int64_t timeout_ns,
                         uint8_t *out_atom_number, bool *out_succeeded);

enum kbase_jm_soft_event_status {
   KBASE_JM_SOFT_EVENT_RESET = 0,
   KBASE_JM_SOFT_EVENT_SET = 1,
};

int kbase_jm_soft_event_update(struct pan_kmod_dev *dev,
                                uint64_t event_gpu_va,
                                enum kbase_jm_soft_event_status status);

int kbase_jm_post_term(struct pan_kmod_dev *dev);

#if defined(__cplusplus)
}
#endif
