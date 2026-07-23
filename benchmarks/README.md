# Solver benchmarks

The solver benchmark runs six Bioptim problems with IPOPT, FATROP, ACADOS, and
MadNLP:

- `pendulum`: torque-driven swing-up with endpoint constraints;
- `cube`: torque-driven marker-target motion with Mayer and Lagrange costs;
- `static_arm`: muscle-driven reaching with residual torques.
- `free_time`: pendulum swing-up with optimized final time;
- `multiphase`: three linked cube motions with one free time per phase;
- `contact_inequality`: jump with unilateral contact and non-slipping inequalities.

It records OCP construction time, wall-clock solve time, solver time, iterations,
cost, status, and constraint violation in JSON and CSV files.

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

## Preliminary results

The following smoke benchmark was run on macOS on 2026-07-22. Each cell is one
cold execution without a warm-up, with a tolerance of `1e-6` and at most 200
iterations. These measurements validate the benchmark and solver integrations;
use multiple warm-ups and repetitions for performance conclusions.

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
