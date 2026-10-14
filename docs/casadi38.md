# CasADi 3.8 experimental support

CasADi 3.8 introduces `Function.transform()` for explicit graph simplification,
including common-subexpression elimination and constant folding. This branch
provides an isolated Linux installation path to evaluate these passes and SX
code generation on repeated optimal-control solves. A version upgrade alone
does not establish a solve-time improvement.

Official release notes: <https://web.casadi.org/get/> (3.8.0, August 22, 2026).

## Opt-in IPOPT callback simplification

```python
solver = Solver.IPOPT()
solver.set_function_transform(True)
# Optional native compilation, paid during preparation:
solver.set_c_compile(True, compiler_flags=["-O1"], cache_dir="/private/path/callback-cache")
```

`set_function_transform(True)` requires CasADi >= 3.8 and its experimental
`Function.transform` API. It builds the IPOPT callbacks, then applies the
explicit pipeline `[["simplify", "cse", "ref_count", "const_folding"]]` to each
callback, including the primal functions, Jacobian and exact Lagrangian
Hessian. Limited-memory Hessian mode is also supported. SX and MX functions
retain their symbolic representation. The pipeline never uses `empty_inputs`
or the API's default implicit simplification flow. Each transformed callback
must preserve its input/output names, shapes and sparsities; otherwise solver
construction raises an error. The default remains disabled and works with
CasADi 3.7.2. Explicit activation on older versions raises a clear error.

The transformed callbacks are installed through CasADi's function `cache`
option. This is distinct from Bioptim's persistent native compilation cache.
Changing the activation flag rebuilds the in-memory solver. When compilation
is requested, code generation uses the transformed callbacks, while native
reload uses the original solver options so it actually evaluates compiled C.
The persistent cache hashes the generated source, so a changed transformation
result invalidates the native entry. A native cache hit still pays solver
construction, transformation and code-generation costs; it skips compilation.
An unchanged graph and configuration reuse the existing in-memory solver.

`ocp`'s solver interface records `function_transform_info` with the explicit
passes, total preparation time and per-callback transformation time/instruction
counts. Preparation should be reported separately from warm optimization
time. Instruction reductions do not by themselves establish faster RHO cycles;
end-to-end certification and timing are still necessary. The feature uses an
API marked internal by CasADi and remains experimental.

## Separate native callback compilation

The opt-in separate mode emits one C translation unit and shared library per
selected IPOPT evaluation callback. It retains the symbolic NLP oracle and the
prepared `nlp_grad` callback in the VM, including their multiplier/sensitivity
behavior, and can be combined with `set_function_transform(True)`:

```python
solver.set_c_compile_callbacks(
    True,
    cache_dir="/private/path/callback-cache",
    compiler_flags=["-O1"],
    callbacks=["nlp_f", "nlp_g", "nlp_grad_f"],
)
```

The explicit subset above leaves Jacobian and Hessian evaluations in the VM.
Omit `callbacks` to compile all evaluation callbacks created by IPOPT, also
including `nlp_jac_g` and `nlp_hess_l` when present. Limited-memory mode does not
create an exact Hessian callback. `nlp_grad` is not a selectable native callback.
It can be needed after the solve even when its reported call count is zero.

This mode requires a private persistent cache directory and Linux/GCC; its cache
ownership, checksum, toolchain and permitted compiler-flag checks match
`set_c_compile`. Each callback is independently content-addressed and atomically
published, including across processes. A failed later callback build leaves any
already completed, valid callback entries reusable; no solver is returned until
all requested callbacks are ready. Code generation still runs on cache hits.
The interface's `c_compile_cache_info` records `mode="callbacks"`, the actual
`native_callbacks` and individual cache-hit/library details. It does not expose
one monolithic `library` field in this mode.

Call `set_c_compile` again to select the historical monolithic mode, or
`set_c_compile_callbacks(False)` to disable compilation. Changing the subset
rebuilds the solver. Solver-level CasADi `jit=True` is rejected in separate mode.
Smaller translation units avoid generating the oracle and `nlp_grad` redundantly,
but a large individual Jacobian or Hessian can still take substantial compiler
time. This API has small-NLP equivalence and fresh-process reload coverage; it
does not establish an end-to-end speedup on the real RHO model.

## Installation

```bash
conda env create -f environment-casadi38.yml
conda activate bioptim-casadi38
.github/scripts/install_biorbd_casadi_linux.sh
python -m pip install --no-deps -e .
python -c 'import casadi, biorbd_casadi, bioptim; print(casadi.__version__)'
pytest -q tests/shard4/test_biorbd_model.py tests/shard5/test_solver_options.py
```

The environment pins the official 3.8.0 wheel. The optional packaging extra
`casadi38` records the supported experimental version range `>=3.8,<3.9`;
it does not build biorbd. Install that extra only in an environment where
RBDL and biorbd will be rebuilt against the chosen wheel. The Linux MadNLP
workflow now exercises 3.8.0 with these native libraries rebuilt from source.

The historical conda environment remains available because conda-forge's
Linux/noarch channels returned no `casadi=3.8` package on September 12, 2026.
Changing that environment's requirement to 3.8 would currently make it
unsatisfiable. macOS and Windows support has not been validated here.

## Native-library compatibility

Replacing CasADi 3.7.2 with 3.8.0 while reusing the installed biorbd/RBDL
binaries fails at import in the tested Linux environment:

```text
librbdl-casadi.so.3.3.1: undefined symbol: _ZN6casadi2MX6binaryExRKS0_S2_
```

RBDL and biorbd must both be rebuilt against the same CasADi headers/library.
The existing installer builds RBDL commit
`93475e2ea9bc87f37709a2312533ce3187f054b9` and biorbd `Release_1.12.2`,
detecting the wheel's C++ ABI from the exported `casadi::MX::_sym` symbol.
The official Linux 3.7.2 wheel requires `_GLIBCXX_USE_CXX11_ABI=0`; the
3.8.0 wheel requires `=1`. Rebuilding against 3.8.0 with the former hard-coded
ABI setting also fails at import, with an unresolved `MX::mtimes` symbol.

Python bindings have a separate compatibility requirement: the 3.8.0 wheel
uses `swig_runtime_data5`, while SWIG 4.3.1 generates runtime 4. With the C++
ABI corrected but SWIG 4.3.1 retained, imports succeed, yet converting a CasADi
`MX` into `biorbd.GeneralizedCoordinates` fails. The initial integration run
found 30 passing and eight failing model/options tests for this reason; the
pendulum could not be constructed. Both wheel-build environments now pin
SWIG 4.4.1 (runtime 5), and the installer verifies an MX round trip instead
of only importing biorbd. CMake explicitly selects the active `swig` executable.
Because biorbd 1.12.2 rejects SWIG >=4.4 at configuration time, the installer
applies the checked-in `.github/patches/biorbd-casadi38.patch` and explicitly
enables its experimental `BIORBD_ALLOW_SWIG4_4` option. This is a local
compatibility patch, not a claim of upstream biorbd support for SWIG 4.4.

With GCC 15.2, biorbd's `src/Utils/Path.cpp` additionally fails because it uses
`SIZE_MAX` without including `<cstdint>`. The same checked-in patch adds
the missing header directly to `Path.cpp`.
It also keeps the unused Eigen RBDL headers in its temporary build directory,
so the install can be repeated without nesting stale header directories.

## Local validation

An isolated venv at `/tmp/bioptim-casadi38-venv` installs CasADi 3.8.0 and
shares Python-only dependencies with the existing Python 3.11 environment.
The original environment retains CasADi 3.7.2. Before rebuilding biorbd,
CasADi-only checks pass: IPOPT and FATROP plugin discovery, an SX
`Function.transform()` call, and a quadratic IPOPT solve returning `x=2`.
Native rebuild and Bioptim integration results follow.

Final local outcome (September 12, 2026):

- RBDL and patched biorbd build and install successfully with ABI 1 and
  SWIG 4.4.1. The MX -> GeneralizedCoordinates -> MX round trip passes.
- With the corrected bindings, the model/options test run aborts (exit 134)
  when iterating `model.segments()`. The pendulum smoke run reports
  `PyThreadState_Get: the function must be called with the GIL held`, from
  `BiorbdModel.ranges_from_model`. No pendulum solve was reached, so neither
  SX/MX solve compatibility nor a speed improvement is established.
- The original CasADi 3.7.2 environment also aborts on the combined model
  test run. Its standalone `test_function_cached` passes, as do all 13
  solver-options tests. This iteration failure is therefore not proven to
  be a 3.8 regression. No existing segments-iteration workaround was found
  in the application source.
- Shell syntax, YAML/TOML parsing, patch applicability and `git diff --check`
  pass. CasADi-only IPOPT/FATROP/transform checks pass as noted above.

The branch remains experimental. Native binary and SWIG incompatibilities
have concrete fixes, but the Python iterator/GIL failure must be resolved
before claiming complete Bioptim compatibility or running end-to-end RHO
performance tests. No further builds or benchmark solves are running.
