# MadNLP solver

Bioptim can use the optional `madnlp` plugin of CasADi's `nlpsol` API. Check the
installed build with:

```python
import casadi as cas
print(cas.has_nlpsol("madnlp"))
```

The standard CasADi 3.7.2 Python wheels do not include this plugin. MadNLP is
implemented in Julia, but users do not install a separate Julia environment at
runtime: CasADi must instead be built with its MadNLP/Julia C interface enabled.
Follow the CasADi build instructions and verify the command above before use.
Bioptim remains importable when the plugin is absent and raises an installation
error only when `Solver.MADNLP()` is constructed.

```python
from bioptim import Solver

solver = Solver.MADNLP()
solver.set_convergence_tolerance(1e-6)
solver.set_constraint_tolerance(1e-6)
solver.set_maximum_iterations(500)
solver.set_print_level("INFO")
solver.set_linear_solver("umfpack")
solution = ocp.solve(solver)
```

CasADi accepts MadNLP-specific values in a nested `madnlp` dictionary. Bioptim
maps convergence and constraint tolerance to MadNLP's single `tol` option,
maximum iterations to `max_iter`, and output verbosity to `print_level`. The
public log-level names are converted to the C interface values `TRACE=1`,
`DEBUG=2`, `INFO=3`, `NOTICE=4`, `WARN=5`, and `ERROR=6`.
`set_option_unsafe(value, name)` adds another value to that nested dictionary;
the CasADi plugin validates its name and type when it constructs the solver.

## Linear solvers

The CPU `libMad` runtime used by CasADi exposes three relevant backends:

- `mumps` (default): sparse symmetric-indefinite factorization with inertia;
- `umfpack`: sparse LU factorization, used by MadNLP in inertia-free mode;
- `lapack_cpu`: dense Bunch-Kaufman factorization, intended only for small
  dense KKT systems.

Select one with `solver.set_linear_solver(...)`. Bioptim translates these
lowercase aliases to the Julia type names required by `libMad`
(`MumpsSolver`, `UmfpackSolver`, and `LapackCPUSolver`). Passing the lowercase
values directly through `set_option_unsafe` is not equivalent: current
`libMad` builds warn and silently retain MUMPS.

MadNLP also supports HSL, Pardiso, and GPU linear solvers in its Julia
packages, but they are not exposed by the CPU `libMad` runtime validated here.
They require a separately compiled runtime before Bioptim can select them.

Warm-start primal values and multipliers are passed through the standard
`x0`, `lam_x0`, and `lam_g0` nlpsol inputs, with `dual_initialized` enabled.
The plugin returns `iter_count` plus `madnlp.primal_feas`,
`madnlp.dual_feas`, and `madnlp.status`; Bioptim normalizes these fields.

Bioptim online optimization callbacks are not enabled because the MadNLP
plugin does not document an iteration callback. Bioptim's C compilation path is
also disabled until its `generate_dependencies`/Importer workflow is covered by
a supported-build integration test, even though current CasADi sources contain
MadNLP code-generation hooks.

Run the unit and optional plugin tests with:

```bash
pytest tests/shard4/test_solver_options.py tests/shard4/test_madnlp.py
```
