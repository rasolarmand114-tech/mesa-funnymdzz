/*
 * SPDX-License-Identifier: MIT
 *
 * kbase_jm.h — Public interface for the Job Manager (JM) command-submission
 * add-on to the kbase kmod backend (kbase_kmod.c / kbase_kmod.h).
 *
 * Scope
 * -----
 * This header/module ONLY concerns the JM flavour of kbase (arch <= 9,
 * Midgard/Bifrost GPUs, uAPI major version 11.x — G31/G51/G52/G72/G76 and
 * older). It never calls into, nor duplicates, the CSF (arch >= 10, uAPI
 * major version 1.x) code paths that already live in kbase_kmod.c
 * (kbase_kmod_csf_group_create / _queue_bind / _queue_kick / ...).
 *
 * Every function here re-validates that the device it is handed is really a
 * JM device before doing anything (see kbase_jm_probe_dev_kind() in
 * kbase_jm.c) and fails safely with -1/ENOTSUP on a CSF device instead of
 * ever issuing a JM-only ioctl or reading a JM-only register/property on
 * hardware that doesn't have it.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct pan_kmod_dev;

/* -------------------------------------------------------------------
 * Device-kind probe (shared concept between CSF and JM)
 * ------------------------------------------------------------------- */

enum kbase_gfx_dev_kind {
   KBASE_GFX_DEV_UNKNOWN = 0,
   KBASE_GFX_DEV_CSF,
   KBASE_GFX_DEV_JM,
};

/* Cheap, side-effect-free probe: returns which command-submission model
 * this kbase device implements, based on the uAPI major version recorded
 * at kbase_kmod_dev_create() time (dev->driver.version.major: 1.x == CSF,
 * 11.x == JM). Callers should branch on this exactly once and then only
 * ever call the matching family of functions (kbase_kmod_csf_* declared in
 * kbase_kmod.h for CSF, kbase_jm_* declared below for JM) for that device's
 * whole lifetime. */
enum kbase_gfx_dev_kind kbase_gfx_dev_kind(const struct pan_kmod_dev *dev);

/* -------------------------------------------------------------------
 * JM job-slot introspection (JS_PRESENT / JS_FEATURES_n registers)
 * ------------------------------------------------------------------- */

#define KBASE_JM_MAX_JOB_SLOTS 16

/* Per-job-slot capability bits, taken verbatim from the hardware
 * JS_FEATURES_<n> register (as exposed by KBASE_GPUPROP_RAW_JS_FEATURES_n).
 * Only the bits kbase_jm.c actually acts on are named; the raw value is
 * always kept too. */
#define KBASE_JM_JSn_FEATURE_NULL           (1u << 1)
#define KBASE_JM_JSn_FEATURE_VERTEX         (1u << 2)
#define KBASE_JM_JSn_FEATURE_TILER          (1u << 7)
#define KBASE_JM_JSn_FEATURE_FRAGMENT       (1u << 12)
#define KBASE_JM_JSn_FEATURE_COMPUTE        (1u << 10)

struct kbase_jm_job_slot_info {
   /* Number of job slots actually present on this GPU (from JS_PRESENT,
    * i.e. popcount of KBASE_GPUPROP_RAW_JS_PRESENT). Always <=
    * KBASE_JM_MAX_JOB_SLOTS. */
   uint32_t slot_count;

   /* Raw JS_FEATURES_<n> register value for each present slot. */
   uint32_t features[KBASE_JM_MAX_JOB_SLOTS];
};

/* Reads JS_PRESENT + JS_FEATURES_0..15 through KBASE_IOCTL_GET_GPUPROPS and
 * fills @out. Returns 0 on success, -1 (errno set) on failure or if @dev is
 * not a JM device. */
int kbase_jm_query_job_slots(struct pan_kmod_dev *dev,
                              struct kbase_jm_job_slot_info *out);

/* -------------------------------------------------------------------
 * JM atom submission
 * ------------------------------------------------------------------- */

enum kbase_jm_atom_kind {
   KBASE_JM_ATOM_VERTEX_TILER, /* BASE_JD_REQ_FS | BASE_JD_REQ_T (needs a tiler-capable slot) */
   KBASE_JM_ATOM_FRAGMENT,     /* BASE_JD_REQ_FS                                              */
   KBASE_JM_ATOM_COMPUTE,      /* BASE_JD_REQ_CS | BASE_JD_REQ_ONLY_COMPUTE                    */
};

enum kbase_jm_atom_priority {
   KBASE_JM_PRIO_REALTIME = 0,
   KBASE_JM_PRIO_HIGH,
   KBASE_JM_PRIO_MEDIUM,
   KBASE_JM_PRIO_LOW,
};

struct kbase_jm_atom_desc {
   /* GPU address of the job chain head (first job descriptor). */
   uint64_t jc;

   enum kbase_jm_atom_kind kind;
   enum kbase_jm_atom_priority priority;

   /* Optional: force a specific job slot instead of auto-picking one from
    * kbase_jm_query_job_slots() capability bits. -1 = auto. */
   int jobslot;

   /* Optional data-dependency: wait for this previously-submitted atom id
    * to complete first. 0 = no dependency (atom id 0 is reserved/unused by
    * this module for that reason — real atom ids start at 1). */
   uint8_t depends_on_atom;
};

/* Submits one job atom through KBASE_IOCTL_JOB_SUBMIT. Automatically picks
 * a job slot whose JS_FEATURES register advertises support for @desc->kind
 * unless desc->jobslot >= 0. Returns the atom id (> 0) on success, or -1
 * (errno set) on failure / if @dev is not a JM device. */
int kbase_jm_atom_submit(struct pan_kmod_dev *dev,
                          const struct kbase_jm_atom_desc *desc);

/* Blocks (up to timeout_ns, or forever if negative) until at least one JM
 * completion event is available, then reads and reports it. Returns 1 and
 * fills *out_atom_number/*out_succeeded when an event was consumed, 0 on
 * timeout, -1 (errno set) on error / non-JM device. This is the JM
 * equivalent of kbase_kmod_csf_wait_event(); it is NOT interchangeable with
 * it, because JM and CSF use different notification payloads. */
int kbase_jm_wait_event(struct pan_kmod_dev *dev, int64_t timeout_ns,
                         uint8_t *out_atom_number, bool *out_succeeded);

/* -------------------------------------------------------------------
 * JM soft-events (BASE_JD_REQ_SOFT_EVENT_WAIT/SET/RESET)
 *
 * A soft-event is a GPU-visible boolean living in a BASE_JD_REQ_SOFT_JOB
 * atom's memory: one atom chain can BASE_JD_REQ_SOFT_EVENT_WAIT on it while
 * a completely independent submission SETs or RESETs it later from the CPU
 * via KBASE_IOCTL_SOFT_EVENT_UPDATE, without needing a full atom
 * round-trip. Useful for cheap cross-queue signalling on JM hardware,
 * which has no CSF-style sync objects.
 * ------------------------------------------------------------------- */

enum kbase_jm_soft_event_status {
   KBASE_JM_SOFT_EVENT_RESET = 0,
   KBASE_JM_SOFT_EVENT_SET = 1,
};

/* @event_gpu_va must be the same GPU address a waiting atom's .jc soft-job
 * descriptor points at. Returns 0 on success, -1 (errno set) on failure /
 * non-JM device. */
int kbase_jm_soft_event_update(struct pan_kmod_dev *dev,
                                uint64_t event_gpu_va,
                                enum kbase_jm_soft_event_status status);

/* -------------------------------------------------------------------
 * JM context teardown notification
 * ------------------------------------------------------------------- */

/* Wakes up anyone blocked in kbase_jm_wait_event() on this fd with a
 * terminating condition, via KBASE_IOCTL_POST_TERM. Call this before
 * tearing down a JM pan_kmod_dev so a concurrent event-wait thread doesn't
 * block forever on a fd that is about to be closed. Returns 0 on success,
 * -1 (errno set) on failure / non-JM device. */
int kbase_jm_post_term(struct pan_kmod_dev *dev);

#if defined(__cplusplus)
} /* extern "C" */
#endif
