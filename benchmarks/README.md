# Solver benchmarks

The solver benchmark runs seven Bioptim problems with IPOPT, FATROP, ACADOS, and
alpaqa:

- `pendulum`: torque-driven swing-up with endpoint constraints;
- `cube`: torque-driven marker-target motion with Mayer and Lagrange costs;
- `static_arm`: muscle-driven reaching with residual torques;
- `free_time`: pendulum swing-up with optimized final time;
- `multiphase`: three linked cube motions with one free time per phase;
- `contact_inequality`: jump with unilateral contact and non-slipping inequalities;
- `holonomic_muscle`: muscle-driven arm/pendulum swing-up with dependent
  coordinates, an iterative holonomic reconstruction, and direct collocation.

It records OCP construction time, wall-clock solve time, solver time, iterations,
cost, status, constraint violation, the native alpaqa status, and available
CasADi function-evaluation counters in JSON and CSV files. CasADi's alpaqa
plugin does not expose ALM/PANOC iteration counts, so that field is empty for
alpaqa.

Run a small comparison from the repository root:

```bash
python -m benchmarks.solver_benchmark \
  --cases pendulum cube static_arm free_time multiphase contact_inequality \
  --solvers ipopt fatrop alpaqa \
  --sizes 20 50 \
  --warmups 1 \
  --repetitions 3
```

All solvers use a tolerance of `1e-6`, at most 500 iterations, one thread, and
`OrderingStrategy.TIME_MAJOR`. The latter is required by FATROP and ensures that
the compared solvers receive the same variable ordering.

Unavailable solvers and runtime failures are written to the result files instead
of aborting the remaining matrix. For ACADOS installed outside its default
location, pass `--acados-dir /path/to/acados`.

Run the long holonomic stress case separately:

```bash
python -m benchmarks.solver_benchmark \
  --cases holonomic_muscle \
  --solvers ipopt fatrop alpaqa \
  --sizes 5 \
  --warmups 0 \
  --repetitions 1
```

## Interpreting results

Alpaqa uses the same initial guess as the other solvers. Some benchmark cases
were originally tuned for interior-point methods and may return
`SOLVER_RET_NAN` from a poor initial guess. Such runs are recorded as solver
failures and must not be included in timing medians.

A one-shot smoke run on macOS with CasADi 3.8.0, five shooting intervals,
`1e-6` tolerances and at most 200 ALM/PANOC iterations produced:

| Case | Outcome | Build (s) | `solve()` (s) | Solver (s) | Cost | Max. violation | ψ evaluations |
|---|---|---:|---:|---:|---:|---:|---:|
| Cube | success | 0.143 | 0.141 | 0.089 | 1117.237760 | 6.08e-7 | 2181 |
| Pendulum | limited/failure | 0.248 | 0.773 | 0.394 | 4.344239 | non-finite | 143 |

These cold, single-run figures validate the benchmark but are insufficient for
performance conclusions. The pendulum's zero initial trajectory is particularly
poor for PANOC and generates non-finite dynamics evaluations.

Constraint violation is computed against each constraint's lower and upper
bounds, rather than as `max(abs(g))`; this is essential for time, contact, and
other inequality constraints whose feasible values are not zero.

The holonomic-muscle case is the long stress benchmark. Its collocation
formulation and implicit reconstruction of dependent coordinates are expensive,
so it should be run separately from routine smoke benchmarks.
