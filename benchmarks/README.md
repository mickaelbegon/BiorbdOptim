# Solver benchmarks

The solver benchmark runs eight Bioptim problems with IPOPT, FATROP, ACADOS, and
MadNLP:

- `pendulum`: torque-driven swing-up with endpoint constraints;
- `cube`: torque-driven marker-target motion with Mayer and Lagrange costs;
- `static_arm`: muscle-driven reaching with residual torques;
- `free_time`: pendulum swing-up with optimized final time;
- `multiphase`: three linked cube motions with one free time per phase;
- `contact_inequality`: jump with unilateral contact and non-slipping inequalities;
- `holonomic_muscle`: muscle-driven arm/pendulum swing-up with dependent
  coordinates, an iterative holonomic reconstruction, and direct collocation;
- `muscle_fatigue`: muscle-driven reaching with Xia fatigue states, residual
  torques, direct collocation, and an exact Hessian.

It records OCP construction time, wall-clock solve time, solver time, iterations,
cost, status, constraint violation, objective/constraint derivative evaluation
times, and whether each run is cold or hot in JSON and CSV files.

All solvers use a tolerance of `1e-6`, at most 500 iterations, one thread, and
`OrderingStrategy.TIME_MAJOR`. The latter is required by FATROP and ensures that
the compared nonlinear solvers receive the same variable ordering.

## Run locally

Run a small comparison from the repository root:

```bash
python -m benchmarks.solver_benchmark \
  --cases pendulum cube static_arm free_time multiphase contact_inequality \
  --solvers ipopt fatrop madnlp \
  --sizes 20 50 \
  --warmups 1 \
  --repetitions 3
```

Unavailable solvers and runtime failures are written to the result files instead
of aborting the remaining matrix. For ACADOS installed outside its default
location, pass `--acados-dir /path/to/acados`.

Run the 500-interval pendulum stress case:

```bash
python -m benchmarks.solver_benchmark \
  --cases pendulum \
  --solvers ipopt fatrop madnlp \
  --sizes 500 \
  --warmups 1 \
  --repetitions 1
```

Run the Hessian-heavy muscle-fatigue benchmark:

```bash
python -m benchmarks.solver_benchmark \
  --cases muscle_fatigue \
  --solvers ipopt fatrop madnlp \
  --sizes 50 \
  --warmups 1 \
  --repetitions 1
```

Compare MadNLP's CPU linear solvers. Run each command in a fresh process so
that every backend has its own cold initialization measurement:

```bash
for linear_solver in mumps umfpack; do
  python -m benchmarks.solver_benchmark \
    --cases cube \
    --solvers madnlp \
    --sizes 10 \
    --madnlp-linear-solver "$linear_solver" \
    --warmups 1 \
    --repetitions 1 \
    --output "benchmarks/results/madnlp-linear-cube-$linear_solver"
  python -m benchmarks.solver_benchmark \
    --cases pendulum \
    --solvers madnlp \
    --sizes 100 \
    --madnlp-linear-solver "$linear_solver" \
    --warmups 1 \
    --repetitions 1 \
    --output "benchmarks/results/madnlp-linear-pendulum-$linear_solver"
done
```

With an x86-64 `libMad` runtime compiled with `MadNLPPardiso`, benchmark
PARDISO MKL in a fresh process:

```bash
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m benchmarks.solver_benchmark \
  --cases pendulum muscle_fatigue \
  --solvers madnlp \
  --sizes 100 \
  --madnlp-linear-solver pardiso_mkl \
  --warmups 1 \
  --repetitions 3
```

The
[`MadNLP PARDISO benchmark`](../.github/workflows/madnlp_pardiso_benchmark.yml)
workflow makes this comparison reproducible on Linux. It compiles the pinned
`mickaelbegon/libMad` PARDISO branch, builds CasADi 3.8 against that runtime,
and runs both MUMPS and PARDISO MKL on the same nine-case matrix used for the
complete solver benchmark. Each case/backend pair runs in a fresh process, with
one cold solve followed by three measured hot solves. MKL, OpenMP, and OpenBLAS
are restricted to one thread.

CasADi 3.7.2's older MadNLP plugin uses the `madnlp_c_*` API and cannot load
this `libMad` runtime by merely changing `LD_LIBRARY_PATH`. CasADi must be
compiled against the newer `libmad_*` interface, which is why the workflow
builds the pinned CasADi 3.8 source instead of using the earlier benchmark
wheel.

IPOPT supports two distinct PARDISO interfaces. A build linked against Intel
oneMKL accepts `pardisomkl`:

```bash
python -m benchmarks.solver_benchmark \
  --cases pendulum \
  --solvers ipopt \
  --sizes 100 \
  --ipopt-linear-solver pardisomkl
```

IPOPT 3.14 can load Panua PARDISO dynamically without recompiling IPOPT:

```bash
python -m benchmarks.solver_benchmark \
  --cases pendulum \
  --solvers ipopt \
  --sizes 100 \
  --ipopt-linear-solver pardiso \
  --ipopt-pardiso-library /absolute/path/to/libpardiso.so
```

Similarly, a compatible HSL library can be loaded at runtime:

```bash
python -m benchmarks.solver_benchmark \
  --cases pendulum \
  --solvers ipopt \
  --sizes 100 \
  --ipopt-linear-solver ma57 \
  --ipopt-hsl-library /absolute/path/to/libcoinhsl.so
```

The IPOPT build still determines which names are valid. In particular, the
macOS arm64 Conda build tested here is linked against OpenBLAS and accepts
`pardiso` through the runtime loader, but not `pardisomkl`.

MUMPS and UMFPACK are sparse solvers. Panua PARDISO, HSL, and GPU backends need
a custom `libMad` runtime. PARDISO MKL is included only in the experimental
x86-64 runtime variant.

### Why LAPACK CPU is excluded

MadNLP's `LapackCPUSolver` remains useful for unit tests and very small dense
NLPs, but it is deliberately excluded from this optimal-control benchmark.
Direct multiple shooting and direct collocation produce large, sparse,
block-structured KKT systems whose dimension grows with the number of shooting
intervals, states, controls, algebraic variables, and constraints.

Converting such a KKT system to a dense matrix discards this structure. Dense
storage grows as \(O(n_\mathrm{KKT}^2)\), while a dense factorization grows as
\(O(n_\mathrm{KKT}^3)\). Increasing the mesh can therefore turn a modest
optimal-control problem into one that is limited by memory or factorization
time, even when the dynamics and Hessian evaluations remain inexpensive.

LAPACK CPU should consequently be restricted to tiny feasibility checks,
backend validation, or problems known to have a genuinely dense and small KKT
matrix. It should not be used to draw performance conclusions for realistic
biomechanics or long-horizon optimal control. The official
[MadNLP linear-solver documentation](https://madsuite.org/MadNLP.jl/stable/)
recommends a sparse solver once the NLP exceeds 1,000 variables; in optimal
control, exploiting sparsity is often worthwhile well before that threshold.

### Exploratory macOS results

These measurements were collected on 2026-07-27 on macOS 26.5.1 arm64 with
Python 3.11.15, CasADi 3.8.0, MadNLP 0.9.1, and Bioptim 3.5.0. Each backend ran
in a fresh process with one cold solve followed by one hot solve. There is only
one observation per cell, so small differences should be treated as ties.

| Case | Shooting | Linear solver | Cold solve (s) | Hot solve (s) | Hot solver (s) | Iter. | Cost | Max violation |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Cube | 10 | MUMPS | 6.754 | **0.018** | 0.00162 | 11 | 1117.997245 | 7.55e-15 |
| Cube | 10 | UMFPACK | **4.523** | **0.018** | **0.00148** | 11 | 1117.997245 | 1.78e-15 |
| Pendulum | 100 | MUMPS | 6.644 | **1.964** | **0.425** | **48** | 36.096161 | 3.85e-13 |
| Pendulum | 100 | UMFPACK | **6.620** | 2.384 | 0.760 | 71 | 36.096161 | 2.60e-09 |
| Muscle fatigue | 20 | MUMPS | **14.417** | **5.846** | **5.200** | **94** | 17.326970 | 7.60e-10 |
| Muscle fatigue | 20 | UMFPACK | 15.326 | 8.593 | 7.985 | 133 | 17.326969 | 3.55e-12 |

On the two nontrivial sparse cases, MUMPS is the clear default: UMFPACK's
inertia-free path needs 48% more iterations on the pendulum and 41% more on
muscle fatigue.

## Reproducible Linux environment

The official CasADi 3.7.2 Linux wheel contains the IPOPT, FATROP, and MadNLP
plugins. It uses the old libstdc++ C++11 ABI, so a Conda biorbd build cannot be
mixed with it. The CI compiles the pinned RBDL-CasADi commit and biorbd 1.12.2
against that same wheel:

```bash
conda env create -f .github/madnlp-linux-environment.yml
conda activate bioptim-madnlp-linux
python -m pip install --no-deps casadi==3.7.2
.github/scripts/install_biorbd_casadi_linux.sh
python -m pip install --no-deps -e .

mkdir -p .cache/madnlp
curl --fail --location --retry 3 \
  https://github.com/tmmsartor/madnlp_c/releases/download/nightly-cpu_only/madnlp-jl1.10.4-ubuntu-20.04-x64.zip \
  --output /tmp/madnlp-jl1.10.4-ubuntu-20.04-x64.zip
echo "333c42a1beb04fdba84410cc861927a74c3591c3b9cffd15323cbfeb44fbc8a0  /tmp/madnlp-jl1.10.4-ubuntu-20.04-x64.zip" \
  | sha256sum --check
unzip -q /tmp/madnlp-jl1.10.4-ubuntu-20.04-x64.zip -d .cache/madnlp
export LD_LIBRARY_PATH="$PWD/.cache/madnlp/foo/lib:${LD_LIBRARY_PATH:-}"
```

The workflow downloads the pinned MadNLP C/Julia runtime, verifies its SHA-256,
adds its `lib` directory to `LD_LIBRARY_PATH`, and checks all three CasADi
plugins before running tests. The exact download and environment setup are in
[`madnlp_linux.yml`](../.github/workflows/madnlp_linux.yml).

Pull requests that modify the integration, benchmark, or solver interfaces run
the MadNLP tests and a three-solver smoke benchmark. A manual
`workflow_dispatch` runs the full four-solver matrix and uploads JSON, CSV, and
Markdown artifacts.

## Linux CI results

These measurements come from
[GitHub Actions run 30048124748](https://github.com/mickaelbegon/BiorbdOptim/actions/runs/30048124748),
executed on 2026-07-23:

- Ubuntu 22.04 runner, Linux `6.8.0-1062-azure`, x86-64, glibc 2.35;
- Python 3.11.15, Bioptim 3.5.0, CasADi 3.7.2;
- one warm-up and one measured hot solve;
- tolerance `1e-6`, at most 500 iterations, one thread.

A **cold** measurement is the first solve in a fresh Python process. A **hot**
measurement rebuilds both the OCP and solver after one successful solve in the
same process. It therefore amortizes plugin and Julia initialization, but it is
not an OCP warm start and does not reuse the previous solution. Each
case/solver combination runs in its own process. Since each column contains one
observation, small differences should be treated as ties until repeated.

### Complete table

| Case | Shooting | Solver | Cold solve (s) | Hot solve (s) | Hot solver (s) | Hot iter. | Cost | Max violation | Outcome |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Pendulum | 20 | IPOPT | 0.727 | 0.724 | 0.159 | 24 | 91.835622 | 6.95e-13 | success |
| Pendulum | 20 | FATROP | 0.778 | 0.771 | 0.167 | 28 | 79.609055 | 2.66e-14 | success |
| Pendulum | 20 | ACADOS | — | — | — | — | — | — | solver failure |
| Pendulum | 20 | MadNLP | 1.584 | 0.751 | 0.188 | 26 | 68.046607 | 1.24e-14 | success |
| Pendulum | 500 | IPOPT | 40.871 | 40.398 | 25.992 | 176 | 35.664490 | 1.85e-09 | success |
| Pendulum | 500 | FATROP | 39.047 | 38.899 | 22.681 | 152 | 35.664490 | 1.83e-09 | success |
| Pendulum | 500 | ACADOS | — | — | — | — | — | — | solver failure |
| Pendulum | 500 | MadNLP | **34.312** | **33.430** | **18.881** | **126** | 35.664458 | 1.42e-14 | success |
| Cube | 10 | IPOPT | 0.044 | 0.042 | 0.010 | 13 | 1117.997079 | 1.33e-15 | success |
| Cube | 10 | FATROP | **0.039** | **0.038** | **0.002** | 15 | 1117.997079 | 1.78e-15 | success |
| Cube | 10 | ACADOS | 2.181 | 2.158 | 0.001 | 1 | 180370.953774 | — | success, non-equivalent cost |
| Cube | 10 | MadNLP | 0.867 | 0.045 | 0.013 | 12 | 1117.995088 | 1.33e-15 | success |
| Static arm | 10 | IPOPT | **35.344** | 35.284 | 15.414 | 69 | 330.979798 | 9.67e-10 | success |
| Static arm | 10 | FATROP | 37.234 | 37.975 | 17.012 | 75 | 382.661494 | 8.86e-10 | success, different local minimum |
| Static arm | 10 | ACADOS | — | — | — | — | — | — | solver failure |
| Static arm | 10 | MadNLP | 35.448 | **34.590** | **15.048** | **67** | 330.979327 | 2.57e-08 | success |
| Free time | 10 | IPOPT | 0.771 | 0.756 | 0.598 | 54 | 220.463496 | 9.96e-09 | success |
| Free time | 10 | FATROP | — | — | — | — | — | — | unsupported structure |
| Free time | 10 | ACADOS | — | — | — | — | — | — | requires SX graph |
| Free time | 10 | MadNLP | — | — | — | — | — | — | runtime failure |
| Multiphase | 10/phase | IPOPT | **0.453** | 0.457 | 0.141 | 11 | 33846.126974 | 4.88e-15 | success |
| Multiphase | 10/phase | FATROP | — | — | — | — | — | — | unsupported structure |
| Multiphase | 10/phase | ACADOS | — | — | — | — | — | — | requires SX graph |
| Multiphase | 10/phase | MadNLP | 1.321 | **0.449** | **0.132** | **10** | 33846.126974 | 3.11e-15 | success |
| Contact inequalities | 10 | IPOPT | **7.073** | **7.041** | **6.499** | **57** | 0.151329095 | 1.27e-10 | success |
| Contact inequalities | 10 | FATROP | 15.917 | 16.081 | 15.529 | 118 | 0.151329186 | 4.13e-14 | success |
| Contact inequalities | 10 | ACADOS | — | — | — | — | — | — | requires SX graph |
| Contact inequalities | 10 | MadNLP | 6.564 | — | — | — | 0.151328262 | 2.84e-05 | infeasible |
| Holonomic muscle | 5 | IPOPT | **32.652** | **33.325** | **31.945** | 27 | 0.013516468 | 7.15e-07 | success |
| Holonomic muscle | 5 | FATROP | 36.879 | 37.075 | 35.680 | 29 | 0.013516606 | 5.38e-07 | success |
| Holonomic muscle | 5 | ACADOS | — | — | — | — | — | — | requires SX graph |
| Holonomic muscle | 5 | MadNLP | 34.382 | 33.401 | 32.010 | 27 | 0.013515161 | 2.28e-07 | success |
| Muscle fatigue | 50 | IPOPT | 22.938 | 23.002 | 21.788 | 67 | 17.288825 | 2.37e-08 | success |
| Muscle fatigue | 50 | FATROP | 64.233 | 64.353 | 62.411 | 168 | 17.288832 | 5.47e-08 | success |
| Muscle fatigue | 50 | ACADOS | — | — | — | — | — | — | requires SX graph |
| Muscle fatigue | 50 | MadNLP | **18.348** | **17.397** | **16.171** | **57** | 17.288543 | 3.46e-12 | success |

The benchmark classifies a solve as infeasible when the maximum bound violation
exceeds `10 × tolerance`. This is why the MadNLP contact result is reported as
infeasible despite returning solver status zero. Constraint violation is
computed against each constraint's lower and upper bounds, rather than as
`max(abs(g))`.

### IPOPT versus MadNLP

MadNLP and IPOPT both succeed on seven of the nine problem sizes. Their main
Linux results are:

- MadNLP is 17% faster hot on the 500-interval pendulum (`33.430` versus
  `40.398` s) and uses 28% fewer iterations.
- MadNLP is 24% faster hot on the Hessian-heavy muscle-fatigue problem
  (`17.397` versus `23.002` s).
- Static arm and multiphase favor MadNLP by 2%, while the cube and holonomic
  cases are within 7% and should be treated as ties with only one observation.
- IPOPT is the only successful solver on `free_time`. On contact inequalities,
  MadNLP returns a nearly identical cost but misses the benchmark feasibility
  threshold.
- The 20-interval pendulum converges to different local minima, so its timings
  do not compare equivalent solutions.

For the six shared-success cases other than the 20-interval pendulum, the
largest IPOPT/MadNLP relative cost difference is below `0.01%`. The cold
MadNLP overhead on small Linux cases is about 0.8 s, substantially smaller than
the earlier macOS measurements.

### Hessian-heavy biomechanics case

The `muscle_fatigue` case uses 50 direct-collocation intervals, six muscle
actuators with Xia fatigue states, residual joint torques, and exact second
derivatives. Derivative times are CasADi's cumulative hot-run measurements
inside the nonlinear solver.

| Solver | Cold solve (s) | Hot solve (s) | Hot solver (s) | Iter. | Solver/iter. (ms) | Hessian (s) | Constraint Jacobian (s) | Cost | Max violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IPOPT | 22.938 | 23.002 | 21.788 | 67 | 325.2 | 7.978 | 10.277 | 17.288825 | 2.37e-08 |
| FATROP | 64.233 | 64.353 | 62.411 | 168 | 371.5 | 23.147 | 22.836 | 17.288832 | 5.47e-08 |
| MadNLP | **18.348** | **17.397** | **16.171** | **57** | **283.7** | **6.838** | **7.539** | 17.288543 | 3.46e-12 |
| ACADOS | — | — | — | — | — | — | — | — | requires SX graph |

MadNLP's hot solver time is 26% lower than IPOPT's. It takes 15% fewer
iterations, and each iteration is 13% faster. FATROP takes 2.5 times as many
iterations as IPOPT and 14% longer per iteration, producing a solver time
2.9 times as long. Hessian and constraint-Jacobian evaluation together account
for 84% of IPOPT's hot solver time, 74% of FATROP's, and 89% of MadNLP's.

All three nonlinear solvers converge to costs within `0.002%` of one another.
This case confirms that both iteration count and per-iteration derivative cost
matter when biomechanical dynamics make exact Hessian and constraint-Jacobian
evaluations expensive.
