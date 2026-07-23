# Solver benchmarks

The solver benchmark runs seven Bioptim problems with IPOPT, FATROP, ACADOS, and
MadNLP:

- `pendulum`: torque-driven swing-up with endpoint constraints;
- `cube`: torque-driven marker-target motion with Mayer and Lagrange costs;
- `static_arm`: muscle-driven reaching with residual torques;
- `free_time`: pendulum swing-up with optimized final time;
- `multiphase`: three linked cube motions with one free time per phase;
- `contact_inequality`: jump with unilateral contact and non-slipping inequalities;
- `holonomic_muscle`: muscle-driven arm/pendulum swing-up with dependent
  coordinates, an iterative holonomic reconstruction, and direct collocation.

It records OCP construction time, wall-clock solve time, solver time, iterations,
cost, status, constraint violation, and whether each run is cold or hot in JSON
and CSV files.

Run a small comparison from the repository root:

```bash
python -m benchmarks.solver_benchmark \
  --cases pendulum cube static_arm free_time multiphase contact_inequality \
  --solvers ipopt fatrop madnlp \
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

Run a larger multiple-shooting stress case with 500 pendulum intervals:

```bash
python -m benchmarks.solver_benchmark \
  --cases pendulum \
  --solvers ipopt madnlp \
  --sizes 500 \
  --warmups 1 \
  --repetitions 1
```

Run the long holonomic stress case separately:

```bash
python -m benchmarks.solver_benchmark \
  --cases holonomic_muscle \
  --solvers ipopt fatrop madnlp \
  --sizes 5 \
  --warmups 0 \
  --repetitions 1
```

## Preliminary results

The following smoke benchmark was run on macOS on 2026-07-22. Each cell is one
cold execution without a warm-up, with a tolerance of `1e-6` and at most 200
iterations. These measurements validate the benchmark and solver integrations;
use multiple warm-ups and repetitions for performance conclusions.

### Cold versus hot: IPOPT and MadNLP

These paired measurements were run on 2026-07-23. A **cold** measurement is the
first solve in a fresh Python process. A **hot** measurement rebuilds both the OCP
and the solver after one successful solve in the same process; it therefore
amortizes global plugin and Julia initialization, but it is not an OCP warm start
and does not reuse the previous solution.

| Case | Shooting | IPOPT cold (s) | IPOPT hot (s) | MadNLP cold (s) | MadNLP hot (s) | Hot comparison |
|---|---:|---:|---:|---:|---:|---|
| Pendulum | 20 | 0.805 | 0.686 | 7.960 | **0.396** | MadNLP 42% faster |
| Pendulum | 500 | **18.266** | **18.945** | 27.239 | 23.292 | IPOPT 19% faster |
| Cube | 10 | 0.178 | 0.211 | 8.521 | **0.018** | MadNLP about 12x faster |
| Static arm | 10 | **20.364** | 31.497 | 27.483 | **23.320** | MadNLP 26% faster hot |
| Free time | 10 | **0.680** | **0.759** | failure | failure | IPOPT only successful solver |
| Multiphase | 10/phase | 0.298 | 0.341 | 7.024 | **0.275** | MadNLP 19% faster |
| Contact inequalities | 10 | 4.566 | 4.180 | 9.287 | **4.089** | MadNLP 2% faster |
| Holonomic muscle | 5 | **22.065** | **20.013** | 26.888 | 20.360 | IPOPT 2% faster hot |

MadNLP's cold penalty is 4–8 seconds on the smaller cases because Julia is
initialized on the first solve. Once hot, MadNLP is competitive or faster on
most successful cases. The free-time failure remains a robustness gap. Each
column currently contains one paired observation, so small differences—especially
the 2% contact and holonomic gaps—should be confirmed with repeated measurements.
On the 500-interval pendulum, both solvers converge to cost `35.664490` with a
maximum constraint violation below `1.86e-9`; IPOPT remains faster both cold and
hot, while MadNLP reduces its startup-inclusive time by 3.95 seconds once hot.

| Case | Shooting | Solver | `solve()` (s) | Solver (s) | Iter. | Cost | Max. constraint violation |
|---|---:|---|---:|---:|---:|---:|---:|
| Pendulum | 20 | IPOPT | 0.790 | 0.340 | 24 | 91.835622 | 6.96e-13 |
| Pendulum | 20 | FATROP | 0.426 | 0.053 | 28 | 79.609055 | 2.31e-14 |
| Pendulum | 20 | MadNLP | 6.715 | 3.808 | 24 | 68.046875 | 3.85e-13 |
| Cube | 10 | IPOPT | 0.249 | 0.198 | 13 | 1117.997079 | 1.33e-15 |
| Cube | 10 | FATROP | 0.070 | 0.002 | 15 | 1117.997079 | 1.78e-15 |
| Cube | 10 | MadNLP | 11.340 | 7.570 | 11 | 1117.997245 | 7.55e-15 |
| Static arm | 10 | IPOPT | 38.415 | 13.304 | 69 | 330.979798 | 9.67e-10 |
| Static arm | 10 | FATROP | 31.133 | 9.649 | 75 | 382.661494 | 8.86e-10 |
| Static arm | 10 | MadNLP | 32.367 | 11.342 | 67 | 330.979936 | 2.24e-08 |
| Free time | 10 | IPOPT | 1.248 | 1.118 | 54 | 220.463496 | 9.96e-09 |
| Multiphase | 10/phase | IPOPT | 0.423 | 0.191 | 11 | 33846.126974 | 4.88e-15 |
| Multiphase | 10/phase | MadNLP | 0.256 | 0.067 | 10 | 33846.126974 | 4.88e-15 |
| Contact inequalities | 10 | IPOPT | 4.273 | 3.929 | 57 | 0.151329 | 1.27e-10 |
| Contact inequalities | 10 | FATROP | 7.596 | 7.171 | 114 | 0.151329 | 4.49e-14 |
| Contact inequalities | 10 | MadNLP | 4.304 | 3.905 | 71 | 0.151329 | 2.72e-12 |
| Holonomic muscle | 5 | IPOPT | 34.192 | 31.606 | 27 | 0.01351647 | 7.15e-07 |
| Holonomic muscle | 5 | FATROP | 62.248 | 59.536 | 29 | 0.01351661 | 5.38e-07 |
| Holonomic muscle | 5 | MadNLP | 101.477 | 70.904 | 26 | 0.01351647 | 7.15e-07 |

ACADOS was unavailable in the tested environments. IPOPT and FATROP used CasADi
3.7.2; MadNLP used the CasADi 3.8.0 MadNLP build. MadNLP's cold times include
Julia startup. The different costs on the pendulum and static-arm cases indicate
different local optima from the same initial guess, so speed must be interpreted
together with objective value and feasibility.

The same smoke run also records unsupported or failing combinations:

| Case | Solver | Outcome |
|---|---|---|
| Free time | FATROP | CasADi FATROP structure detection rejects the NLP layout |
| Free time | MadNLP | line search evaluates a constraint at NaN and aborts |
| Multiphase | FATROP | CasADi FATROP structure detection rejects the NLP layout |

Constraint violation is computed against each constraint's lower and upper
bounds, rather than as `max(abs(g))`; this is essential for time, contact, and
other inequality constraints whose feasible values are not zero.

The holonomic-muscle case is the long stress benchmark. Even at only five
shooting intervals, its collocation formulation and implicit reconstruction of
dependent coordinates require 34–101 seconds for one cold `solve()` call on the
test machine. Increasing `--sizes` therefore scales this case quickly and should
be done separately from routine smoke benchmarks.
