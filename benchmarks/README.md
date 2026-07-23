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

### Alpaqa versus IPOPT

The cube case was then run with one warm-up and three measured repetitions.
Alpaqa used Python 3.11/CasADi 3.8.0; IPOPT used Python 3.14/CasADi 3.7.2 because
the available CasADi 3.8 environment contains an ABI-incompatible IPOPT plugin.
The figures therefore compare the solver integrations on the same machine and
problem definition, but not within an identical runtime environment.

| Shooting | Solver | Build median (s) | `solve()` median (s) | Solver median (s) | Cost | Max. violation |
|---:|---|---:|---:|---:|---:|---:|
| 5 | IPOPT | 0.058 | 0.102 | 0.079 | 1117.237740 | 1.33e-15 |
| 5 | alpaqa | 0.092 | 0.172 | 0.126 | 1117.237758 | 2.36e-7 |
| 10 | IPOPT | 0.082 | 0.116 | 0.075 | 1117.997079 | 1.33e-15 |
| 10 | alpaqa | 0.182 | 0.720 | 0.656 | 1117.997099 | 6.24e-7 |

On this small problem, alpaqa is 1.68 times slower than IPOPT in `solve()` at
five shooting intervals and 6.21 times slower at ten intervals. Its solution
cost differs by less than `2e-5`, and its final constraint violation remains
below the requested `1e-6` tolerance. These results do not yet support a speed
advantage for alpaqa; NMPC-style warm starts and time-limited repeated solves
should be benchmarked separately.

Constraint violation is computed against each constraint's lower and upper
bounds, rather than as `max(abs(g))`; this is essential for time, contact, and
other inequality constraints whose feasible values are not zero.

## Cyclic NMPC benchmark

`nmpc_solver_benchmark.py` uses the repository's simple two-degree-of-freedom
cyclic arm NMPC example. Generate one converged IPOPT horizon first, then give
that exact primal/dual seed to both solvers. The seed file makes it possible to
compare installations where IPOPT and alpaqa are provided by different CasADi
environments:

```bash
python -m benchmarks.nmpc_solver_benchmark \
  --solver ipopt --seed-output /tmp/bioptim_nmpc_seed.npz

python -m benchmarks.nmpc_solver_benchmark \
  --solver ipopt --seed-input /tmp/bioptim_nmpc_seed.npz \
  --output benchmarks/results/nmpc_ipopt

python -m benchmarks.nmpc_solver_benchmark \
  --solver alpaqa --seed-input /tmp/bioptim_nmpc_seed.npz \
  --output benchmarks/results/nmpc_alpaqa
```

The default problem has 20 shooting intervals, five consecutive windows, a
`1e-4` tolerance, and a 0.5 s alpaqa deadline per window. After the common first
window, Bioptim advances the horizon and shifts the previous trajectory to form
the next initial guess. Failures do not stop the run: meeting the deadline is a
measured outcome. JSON and CSV outputs contain status, cost, solver time,
average wall time per window, and equality-constraint violation for every
window. Use the same options and seed when comparing result files.

A three-repetition smoke comparison (three windows per repetition) gave the
following median solver times. As above, IPOPT used CasADi 3.7.2 and alpaqa used
CasADi 3.8.0 on the same machine, so these figures validate the protocol but
are not a controlled solver-only comparison.

| Window | IPOPT status | IPOPT solver (s) | alpaqa status | alpaqa solver (s) | alpaqa violation |
|---:|---:|---:|---:|---:|---:|
| 0 (IPOPT seed) | 3/3 | 0.011 | 3/3 | 0.028 | 2.90e-5 |
| 1 (shifted) | 3/3 | 0.161 | 0/3 | 0.504 | 3.87e-3 |
| 2 (shifted) | 3/3 | 0.177 | 0/3 | 0.503 | 3.89e-3 |

The first alpaqa solve is feasible and close to the IPOPT cost (a difference
of about `1.1e-5`). With the 0.5 s deadline, however, alpaqa does not make the
shifted warm starts feasible while IPOPT solves all windows. Improving how
primal and dual warm starts are shifted for alpaqa is therefore the next useful
optimization target; simply supplying an excellent first-window solution is
not enough for this NMPC case.

The holonomic-muscle case is the long stress benchmark. Its collocation
formulation and implicit reconstruction of dependent coordinates are expensive,
so it should be run separately from routine smoke benchmarks.
