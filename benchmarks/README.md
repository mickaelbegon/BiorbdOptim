# Solver benchmarks

The solver benchmark runs the same Bioptim problem with IPOPT, FATROP, ACADOS,
and MadNLP. It records OCP construction time, wall-clock solve time, solver time,
iterations, cost, status, and constraint violation in JSON and CSV files.

Run a small comparison from the repository root:

```bash
python -m benchmarks.solver_benchmark \
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
