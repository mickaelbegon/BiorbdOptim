# Alpaqa through CasADi

Bioptim exposes alpaqa through CasADi's NLP solver plugin. It consumes the same
scaled CasADi NLP as IPOPT and FATROP, so it does not introduce a separate OCP
transcription. Alpaqa combines an augmented Lagrangian method (ALM) for general
constraints with PANOC and an L-BFGS direction for the box-constrained inner
problems. This can be attractive for repeated NMPC solves, but real-time
performance must be established on the actual model and hardware.

## Installation and availability

The standard CasADi 3.8.0 macOS Python wheel tested for this integration does
not contain the alpaqa plugin. The native `alpaqa` Python package is not used by
Bioptim and is not a dependency. CasADi must instead be compiled with its
alpaqa interface enabled and with a compatible alpaqa C++ library. Verify the
resulting build before running Bioptim:

```python
import casadi as cas

print(cas.__version__)
print(cas.has_nlpsol("alpaqa"))
print(cas.nlpsol_options("alpaqa"))
```

The minimum version verified from CasADi's published source is 3.7.2; earlier
versions are not claimed as supported. Build CasADi 3.7.2 or newer with
`WITH_ALPAQA=ON`. Consult the CasADi build instructions for the platform.

## Usage

```python
from bioptim import Solver

solver = Solver.ALPAQA()
solver.set_convergence_tolerance(1e-6)   # alm.tolerance
solver.set_constraint_tolerance(1e-6)    # alm.dual_tolerance
solver.set_alm_maximum_iterations(100)
solver.set_panoc_maximum_iterations(1000)
solver.set_lbfgs_memory(20)

solution = ocp.solve(solver)
```

Options are passed in the exact nested form expected by CasADi:

```python
{"alpaqa": {"alm": {...}, "panoc": {...}, "lbfgs": {...}}}
```

Advanced supported parameters can be set with, for example,
`solver.set_option_unsafe(10.0, "alm.initial_penalty")`. Unknown groups and
names raise an error rather than being silently ignored.

The primal initial guess is always passed as `x0`. Constraint multipliers are
warm-started through `lam_g0` when a previous Bioptim solution is supplied.
The CasADi plugin does not use `lam_x0`, so bound multipliers are not
warm-started. The plugin returns the standard CasADi outputs `x`, `f`, `g`,
`lam_x`, `lam_g`, and `lam_p`.

Bioptim computes the final maximum constraint-bound violation and exposes it as
`solution.inf_pr`. CasADi 3.7.2's plugin does not export the detailed ALM/PANOC
statistics shown by native alpaqa, so outer/inner iteration counts and the dual
residual are not available. CasADi does expose `success`,
`unified_return_status`, evaluation counters, and evaluation/total timings.

Online iteration callbacks are not invoked by this plugin. Generating `nlp.c`
and loading it with `casadi.Importer(..., "shell")` succeeds with CasADi 3.8.0,
but reconstructing the alpaqa solver fails because the imported NLP does not
provide the derivatives required by alpaqa. Bioptim therefore rejects
`c_compile=True`. Alpaqa can be sensitive to scaling: Bioptim passes its scaled
decision variables, but users should also keep constraint residuals at
comparable orders of magnitude.

The current `origin/master` uses direct multiple shooting and has no block
shooting/DSS NLP transcription. Because this interface only consumes the final
NLP vectors, it contains no DMS-specific indexing assumption, but block
shooting and DSS remain untested until those transcriptions are present.
