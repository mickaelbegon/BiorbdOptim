# FHO3 post-shake registry benchmark

Date: 2026-09-27.  This is an evaluation-only benchmark on CPUs 12--23,
with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.  It builds the
historical reduced 0.30 Nm MX FHO3, then stops before IPOPT iterates.

## Baseline

| quantity | value |
|---|---:|
| decision variables | 12,263 |
| constraints | 11,911 |
| upper-triangular `nlp_hess_l` nnz | 49,077 |
| native full-Hessian evaluation | 0.1580 s |

## Real repeated post-shake group

The registry materialized 123 penalty contributions in 16.83 s (offline
introspection, not a solver cost).  With a 128-active-variable cap, 29
nonlinear constraints remained.  They form one algebraically validated group:
`reduced_internal_crank_velocity_constraint`, 29 occurrences, 6 local
decision variables, and 12 local Hessian nnz per occurrence.

| operation over 29 occurrences | median time |
|---|---:|
| VM mapped local Hessian (12 map workers) | 0.422 ms |
| generated-C mapped local Hessian (12 map workers) | 0.379 ms |
| C generation + compile (one kernel) | 0.232 s |
| value difference VM/C | 0 |

The C kernel is a modest **1.11x** faster than the VM map for this group.
Crucially, compacting the selected local function before code generation
reduced generated C from 2.46 MB to 80 kB (shared object: 41 kB).  A raw
post-shake substitution still carries the full zero-filled NLP embedding and
must never be sent directly to C generation.

## Interpretation

This is a correctness and feasibility signal, not a projected FHO speedup.
The measured group represents at most 348 local Hessian entries before sparse
overlap, versus 49,077 in the full FHO Hessian.  Replacing only this group
cannot materially lower the 0.158 s native full-Hessian cost.

The important limitation is `ThreadMap`: generic dispatch currently records a
multi-thread penalty after all stage inputs have been packed into one CasADi
call.  Its expensive dynamics/continuity packets therefore appear as a large
aggregate instead of one local record per stage.  A useful production API must
record the stage fragments *before* that aggregation, while separately keeping
their canonical final `g` row ranges.

## Issue proposal (do not publish automatically)

**Title:** Expose post-shake penalty provenance and per-stage ThreadMap
fragments for exact sparse derivative callbacks

**Body:**

Bioptim's final MX NLP provides aggregate `f(x), g(x)` but loses the mapping
from repeated penalty/ThreadMap stage to canonical `x` columns and `g` rows.
This blocks safe packet-level exact derivative code generation.  Branch
`codex/postshake-penalty-registry` (`01308717`) prototypes an opt-in registry
with exact localized value/Jacobian/Hessian functions and audited canonical
rows.  On a 12,263-variable / 11,911-constraint cycling FHO3, it finds a real
29-stage, 6-variable group whose generated-C Hessian map matches VM exactly
and is 1.11x faster.  However, the expensive ThreadMap terms are still
recorded only after aggregation, so this does not yet improve the full
Hessian.

Proposed API: at penalty dispatch, expose immutable records containing
`penalty type`, `phase`, semantic stage, canonical `g` row range, active `x`
indices, local sparsity, and a per-stage fragment for ThreadMap penalties.
Keep it opt-in and solver-neutral.  Acceptance tests should reconstruct
canonical `g`, Jacobian triplets, and Hessian-Lagrangian triplets from records
to <=1e-10 on SX and MX, including collocation, multinode, and ThreadMap
constraints.  No IPOPT callback change is requested in the first PR.
