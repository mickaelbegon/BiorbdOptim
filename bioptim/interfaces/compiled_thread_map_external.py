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
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from contextlib import contextmanager

import casadi as ca
from casadi import CodeGenerator, Function, MX, external, vertcat


_COMPILER_FLAGS = ("-fPIC", "-shared", "-O3")
_CACHE_SCHEMA = 1


def _compiler_identity(compiler: str) -> dict[str, str]:
    """Return the parts of a compiler identity which affect generated ABI/code."""

    try:
        version = subprocess.run(
            [compiler, "--version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        version = "unavailable"
    return {"path": str(Path(compiler).resolve()), "version": version}


def _cache_metadata(function: Function, compiler: str) -> dict[str, object]:
    """Describe every environment input that may make an External stale."""

    return {
        "schema": _CACHE_SCHEMA,
        "function_sha256": hashlib.sha256(function.serialize().encode()).hexdigest(),
        "function_name": function.name(),
        "casadi_version": ca.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "compiler": _compiler_identity(compiler),
        "compiler_flags": list(_COMPILER_FLAGS),
    }


def _safe_stem(metadata: dict[str, object]) -> str:
    """A deterministic, environment-specific name for an External artifact."""

    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return f"bioptim_thread_map_{hashlib.sha256(encoded).hexdigest()[:20]}"


@contextmanager
def _artifact_lock(path: Path):
    """Serialize cache fills across independent FHO processes on POSIX hosts."""

    # ``flock`` is deliberately held across code generation too: two workers
    # generating the same CasADi helper names in one cache directory would
    # otherwise race before the atomic .so rename.
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_if_valid(function: Function, library: Path, manifest: Path, expected: dict[str, object]) -> Function | None:
    """Accept an artifact only after its metadata and CasADi loader agree."""

    if not library.is_file() or library.stat().st_size == 0 or not manifest.is_file():
        return None
    try:
        observed = json.loads(manifest.read_text(encoding="utf-8"))
        if observed != expected:
            return None
        # Constructing the External verifies that the expected CasADi symbol is
        # exported by the shared object, rather than trusting file existence.
        return external(function.name(), str(library))
    except (OSError, RuntimeError, ValueError):
        return None


def _write_external(function: Function, destination: Path, stem: str, compiler: str) -> Path:
    """Build an External in a private directory, returning its temporary .so."""

    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{stem}-", dir=destination))
    source = temporary_dir / f"{stem}.c"
    library = temporary_dir / f"{stem}.so"
    try:
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
        generator.generate(str(temporary_dir) + "/")
        subprocess.run(
            [compiler, *_COMPILER_FLAGS, str(source), "-o", str(library)],
            check=True,
            capture_output=True,
            text=True,
        )
        return library
    except subprocess.CalledProcessError as error:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to compile {function.name()} as an External C kernel: {error.stderr}") from error
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


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
    metadata = _cache_metadata(function, compiler)
    stem = _safe_stem(metadata)
    library = destination / f"{stem}.so"
    manifest = destination / f"{stem}.json"
    cached = _load_if_valid(function, library, manifest, metadata)
    if cached is not None:
        return cached

    with _artifact_lock(destination / f"{stem}.lock"):
        cached = _load_if_valid(function, library, manifest, metadata)
        if cached is not None:
            return cached
        temporary_library = _write_external(function, destination, stem, compiler)
        try:
            # Both artifacts become visible only after the compiler succeeded.
            # A stale manifest is rejected on the next load rather than used.
            os.replace(temporary_library, library)
            temporary_manifest = destination / f".{stem}.{os.getpid()}.json"
            temporary_manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary_manifest, manifest)
        finally:
            shutil.rmtree(temporary_library.parent, ignore_errors=True)
        loaded = _load_if_valid(function, library, manifest, metadata)
        if loaded is None:
            raise RuntimeError(f"Compiled External cache validation failed for {function.name()}.")
        return loaded


def compile_external_output_packets(
    function: Function, cache_dir: str | Path, max_output_rows: int | None
) -> Function:
    """Compile a mapped scalar function as bounded exact-output packets.

    A large ``ThreadMap`` stage is still much smaller than the complete FHO,
    but its primal plus all second-order helpers can be too large for one C
    compiler translation unit.  Splitting its *outputs* preserves the same
    input signature and exact CasADi AD graph while bounding every generated
    source file.  The returned MX wrapper is deliberately tiny: it only
    concatenates ``External`` calls, so the global FHO remains MX and
    non-compiled.

    ``None`` keeps the historical one-library behaviour.  This helper only
    supports the single-output functions used below Bioptim ``Function.map``.
    """

    if max_output_rows is None:
        return compile_external_with_exact_derivatives(function, cache_dir)
    if not isinstance(max_output_rows, int) or max_output_rows < 1:
        raise ValueError("max_output_rows must be a positive integer or None")
    if function.n_out() != 1:
        raise ValueError("Packetized ThreadMap externals require one function output")

    output_rows = function.size1_out(0)
    if output_rows <= max_output_rows:
        return compile_external_with_exact_derivatives(function, cache_dir)

    # Use MX symbols even for an SX source: the wrapper must compose into the
    # FHO's MX graph and CasADi retains the source function's exact AD helpers.
    inputs = [MX.sym(function.name_in(index), *function.size_in(index)) for index in range(function.n_in())]
    output = function(*inputs)
    if isinstance(output, (tuple, list)):
        output = output[0]
    externals = []
    for start in range(0, output_rows, max_output_rows):
        stop = min(start + max_output_rows, output_rows)
        packet = Function(
            f"{function.name()}_packet_{start:04d}_{stop:04d}",
            inputs,
            [output[start:stop]],
            [function.name_in(index) for index in range(function.n_in())],
            [function.name_out(0)],
        )
        externals.append(compile_external_with_exact_derivatives(packet, cache_dir))

    packetized = vertcat(*[kernel(*inputs) for kernel in externals])
    return Function(
        f"{function.name()}_packetized_{max_output_rows}",
        inputs,
        [packetized],
        [function.name_in(index) for index in range(function.n_in())],
        [function.name_out(0)],
    )
