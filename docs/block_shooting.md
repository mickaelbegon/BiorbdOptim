# Block shooting

Block shooting is an explicit direct-shooting transcription between direct single shooting (DSS) and Bioptim's
historical direct multiple shooting (DMS). It partially condenses the state trajectory while leaving controls,
parameters, objectives, and ordinary constraints in the generic CasADi NLP.

For a phase with `N` shooting intervals, `B` blocks define boundaries
`0 = b_0 < b_1 < ... < b_B = N`. Only the state at each non-terminal block boundary is an independent variable.
States inside a block and the terminal state are produced by the existing RK integrator. Therefore, a phase has
`B * nx` independent state values and `(B - 1) * nx` block-continuity constraints.

- `BlockShooting(n_blocks=1)` is DSS: only the initial state is independent.
- `BlockShooting(n_blocks=B)` is a partially condensed multiple-shooting transcription.
- `BlockShooting(n_blocks=N)` uses one block per interval. It has `N` independent state vectors; unlike historical
  DMS, its terminal state remains integrated rather than independent.
- Omitting `block_shooting` preserves historical DMS with `N + 1` independent state vectors.

## Configuration

```python
from bioptim import BlockShooting, OptimalControlProgram

ocp = OptimalControlProgram(
    bio_model=model,
    dynamics=dynamics,
    n_shooting=100,
    phase_time=1.0,
    x_bounds=x_bounds,
    u_bounds=u_bounds,
    x_init=x_init,
    u_init=u_init,
    objective_functions=objectives,
    block_shooting=BlockShooting(n_blocks=5),
)
```

Equivalent convenience constructors are available:

```python
BlockShooting.single()
BlockShooting.from_number_of_blocks(5)
BlockShooting.from_block_size(10)
```

`n_blocks` creates blocks whose lengths differ by at most one, with larger blocks first. `block_size` creates full
blocks of that size followed by a shorter final block when needed. The two options are mutually exclusive.

For a multiphase OCP, pass one configuration per phase:

```python
block_shooting = [
    BlockShooting.single(),
    BlockShooting(n_blocks=4),
    BlockShooting(block_size=10),
]
```

## State bounds and initial guesses

Bounds at block starts remain variable bounds (`lbx` and `ubx`). Finite bounds at integrated internal and terminal
nodes are converted into general `BOUND_STATE` constraints; rows with two infinite bounds are omitted. This keeps the
usual first, intermediate, last, linear, and per-frame bound interpolation semantics.

Only initial guesses at block starts enter the NLP vector. Initial guesses supplied at eliminated nodes are currently
ignored. When a complete `Solution` is used as a primal warm start, Bioptim samples its states at block starts and
retains controls at every control node. Dual variables are reused only when the solution belongs to the exact same OCP;
they are deliberately omitted when warming a block-shooting OCP from another transcription or OCP instance.

`solution.decision_states()` deliberately preserves the historical API and returns the reconstructed `N + 1` nodal
states. Internally, `nlp.X_decision` contains block-start variables, while `nlp.X` contains the full symbolic
trajectory.

## Current compatibility

The initial implementation supports MX graphs, RK1, RK2, RK4, RK8, fixed or optimized phase time, optimized dynamic
parameters, numerical time series, multiple phases, and control types `CONSTANT`, `CONSTANT_WITH_LAST_NODE`,
`LINEAR_CONTINUOUS`, and `NONE`. It uses the solver-independent CasADi NLP path and is exercised through IPOPT and the
generic FATROP dispatch.

Direct collocation, IRK, trapezoidal and variational integrators, SX graphs, stochastic OCPs, and non-empty algebraic
states are rejected explicitly. Continuity-as-an-objective is also not implemented for block shooting yet.

Long blocks reduce NLP dimension and continuity constraints but produce denser, deeper symbolic dependencies and can
make single-shooting sensitivities harder to optimize. More blocks improve locality at the cost of additional state
variables and matching constraints.

See [`bioptim/examples/getting_started/block_shooting.py`](../bioptim/examples/getting_started/block_shooting.py) for
an executable pendulum example.
