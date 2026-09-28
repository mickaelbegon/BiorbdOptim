"""Opt-in C ``External`` kernels for repeated mapped penalty stages.

The normal Bioptim path keeps the original CasADi ``Function.map`` graph.  A
caller can explicitly request this helper for a selected penalty family to
replace the *scalar* function below that map by a generated C external.  The
generated library contains the primal and the first-/second-order AD helper
functions CasADi needs to differentiate an ``External`` call exactly.  CasADi
therefore still builds the global MX Jacobian and Lagrangian Hessian itself.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess

from casadi import CodeGenerator, Function, external


def _safe_stem(function: Function) -> str:
    """A deterministic name which cannot collide between scalar functions."""

    digest = hashlib.sha256(function.serialize().encode()).hexdigest()[:16]
    return f"bioptim_thread_map_{digest}"


def compile_external_with_exact_derivatives(function: Function, cache_dir: str | Path) -> Function:
    """Return a C ``External`` equivalent to ``function`` with exact AD support.

    CasADi discovers derivative helpers by their standard generated names
    (``jac_*``, ``fwd1_*``, ``adj1_*`` and their mixed first derivatives).  It
    is essential to generate all of them: compiling only the primal makes an
    ``External`` unusable in the global NLP Jacobian/Hessian.
    """

    compiler = shutil.which("gcc")
    if compiler is None:
        raise RuntimeError("Compiled ThreadMap externals require gcc on PATH.")
    destination = Path(cache_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(function)
    source = destination / f"{stem}.c"
    library = destination / f"{stem}.so"
    if not library.exists():
        generator = CodeGenerator(source.name)
        # The sequence intentionally mirrors CasADi's exact forward/reverse
        # chain for a scalar function embedded in an MX NLP.
        for derivative in (
            function,
            function.jacobian(),
            function.forward(1),
            function.reverse(1),
            function.forward(1).reverse(1),
            function.reverse(1).forward(1),
        ):
            generator.add(derivative)
        generator.generate(str(destination) + "/")
        temporary = destination / f".{stem}.tmp.so"
        try:
            subprocess.run(
                [compiler, "-fPIC", "-shared", "-O3", str(source), "-o", str(temporary)],
                check=True,
                capture_output=True,
                text=True,
            )
            temporary.replace(library)
        except subprocess.CalledProcessError as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to compile {function.name()} as an External C kernel: {error.stderr}") from error
    return external(function.name(), str(library))
