# Post-shake penalty registry (prototype)

## Why

Large direct-collocation NLPs repeat the same dynamics and penalty structures
at many stages.  After Bioptim's `shake` pass, however, the solver sees only a
single canonical `f(x), g(x)` graph.  The original penalty provenance and the
final NLP row/column locations are no longer available together.  This makes
it impossible to safely replace repeated local Hessian work by one generated
kernel per term type.

`SolverInterface.build_post_shake_penalty_registry()` is an opt-in
introspection API.  It does not alter a solve, callbacks, or the canonical
NLP.  It reconstructs the same generic dispatch and applies the same `shake`
operation term by term.

## Minimal interface

```python
registry = interface.build_post_shake_penalty_registry(expand=False)
for term in registry.terms:
    print(term.metadata.penalty_name, term.metadata.stage)
    print(term.metadata.g_row_start, term.metadata.g_row_stop)
    print(term.decision_indices)
    value = term.value(local_x)
    jacobian = term.jacobian(local_x)
    hessian = term.lagrangian_hessian(local_x, local_lambda)
```

Every term exposes:

| field | meaning |
|---|---|
| `metadata.kind`, `scope`, `penalty_name` | stable provenance for grouping candidate kernels |
| `metadata.phase`, `stage`, `occurrence`, `multi_thread` | location in the OCP / ThreadMap call |
| `g_row_start`, `g_row_stop` | final canonical `g` rows (global, post-dispatch) for constraints |
| `decision_indices` | active columns in the final NLP vector `x` |
| `value`, `jacobian`, `lagrangian_hessian` | exact functions restricted to active `x` columns |
| `jacobian_sparsity`, `hessian_sparsity` | local sparse structures |

The Hessian function takes the local constraint multipliers.  For scalar
objective terms it also takes a one-element multiplier, allowing a common
kernel API.

## Expected compilation pipeline

1. Build and materialize the registry after the final symbolic problem has
   been dispatched and shaken.
2. Group only terms whose *provenance, local dimensions, active-index pattern
   modulo stage translation, and sparsity* match.
3. Generate C once per verified group for value/Jacobian/Hessian.
4. Map each generated kernel across its registered stages, then scatter its
   local values into the canonical sparse callback buffers.
5. Compare the assembled callback against the original MX Hessian and
   Jacobian before enabling it for IPOPT.

The registry intentionally stops before step 4.  CasADi's `nlpsol` callback
contract needs a solver-specific sparse scatter implementation; providing one
inside this prototype would change the canonical solver path and make the
equivalence audit harder to trust.

Before C generation, expand the selected *local* function.  The exact
full-vector substitution used to localize MX expressions contains a
zero-filled global embedding; expanding only the selected kernel removes that
embedding without paying the cost for every registry entry.

## Expected benefit and limitations

The benefit comes from amortizing code generation and crossing the Python/C
ABI once per *packet/type*, not once per row.  It should therefore be used for
large homogeneous groups such as collocation dynamics or ThreadMap penalties.
Small, heterogeneous, or cross-phase terms should remain in the MX graph.

This API does not by itself make a solve faster.  Materialization builds
individual exact derivative graphs and is deliberately diagnostic/offline.
It must be cached and invalidated whenever the symbolic NLP changes (for
example, horizon length, constraints, or a RHO initial-condition rebuild).

`shake` can eliminate fixed time variables.  Consequently all exposed
indices are meaningful only for the exact final decision vector supplied to
the registry.  Consumers must never reuse a registry across changed NLP
layouts.
