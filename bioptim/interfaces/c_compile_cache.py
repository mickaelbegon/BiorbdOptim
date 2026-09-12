"""Private, content-addressed cache for CasADi NLP callback libraries.

Only compilation is cached. Solver construction and code generation still run
so the cache key describes the actual callbacks selected by CasADi/IPOPT.
"""

import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import tempfile

import casadi


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_entry(entry, signature):
    """Reject incomplete/tampered entries instead of silently loading code."""
    manifest_path = entry / "manifest.json"
    library = entry / "nlp.so"
    try:
        if entry.is_symlink() or manifest_path.is_symlink() or library.is_symlink():
            raise ValueError("symbolic links are not valid cache entries")
        manifest = json.loads(manifest_path.read_text())
        if manifest["signature"] != signature or manifest["library_sha256"] != _sha256(library):
            raise ValueError("signature or library checksum mismatch")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(
            f"Invalid compiled NLP cache entry at {entry}; choose a new cache_name or remove this entry."
        ) from error
    return library


def cached_nlpsol(plugin_name, nlp, options, *, compiler_flags, cache_dir, cache_name):
    """Build/load callbacks with a Linux GCC toolchain; return solver and metadata.

    The caller owns the cache. Checksums detect accidental corruption, not a
    malicious writer with access to that user's directory. Staging and atomic
    directory publication prevent readers from observing a partial build.
    """
    if platform.system() != "Linux":
        raise NotImplementedError("Persistent NLP compilation currently supports Linux with GCC only")
    if any(not re.fullmatch(r"-O[0-3sg]|-g[0-3]?", flag) for flag in compiler_flags):
        raise ValueError(
            "Persistent NLP compilation currently accepts only -O0/-O1/-O2/-O3/-Os/-Og and -g flags; "
            "flags introducing external dependencies need a broader cache signature"
        )
    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    permissions = root.stat()
    if permissions.st_uid != os.getuid() or permissions.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("cache_dir must be owned by the current user and not writable by group or others")
    compiler = shutil.which("gcc")
    if compiler is None:
        raise RuntimeError("Persistent NLP compilation requires gcc on PATH")
    compiler = str(Path(compiler).resolve())
    compiler_version = subprocess.check_output([compiler, "--version"], text=True)
    compiler_target = subprocess.check_output([compiler, "-dumpmachine"], text=True).strip()
    toolchain = {}
    for tool in ("cc1", "as", "ld"):
        executable = subprocess.check_output([compiler, f"-print-prog-name={tool}"], text=True).strip()
        resolved = shutil.which(executable)
        if resolved is None:
            raise RuntimeError(f"Cannot identify GCC toolchain component {tool}: {executable}")
        resolved = Path(resolved).resolve()
        toolchain[tool] = {"path": str(resolved), "sha256": _sha256(resolved)}
    vm_solver = casadi.nlpsol("nlpsol", plugin_name, nlp, options)
    with tempfile.TemporaryDirectory(prefix=f".{cache_name}-", dir=root) as temporary:
        staging = Path(temporary)
        # Equivalent to generate_dependencies(), using the directory-prefix API
        # so concurrent callers never change the process working directory.
        generator = casadi.CodeGenerator("nlp.c")
        generator.add(vm_solver.oracle())
        for name in vm_solver.get_function():
            generator.add(vm_solver.get_function(name))
        generator.generate(str(staging) + os.sep)
        source = staging / "nlp.c"
        signature = {
            "format": 1,
            "source_sha256": _sha256(source),
            "casadi_version": casadi.__version__,
            "casadi_git_revision": casadi.CasadiMeta.git_revision(),
            "compiler": compiler,
            "compiler_sha256": _sha256(Path(compiler)),
            "compiler_version": compiler_version,
            "compiler_target": compiler_target,
            "toolchain": toolchain,
            "compiler_flags": list(compiler_flags),
            "platform": platform.platform(),
            "plugin": plugin_name,
            "compiler_environment": {
                name: os.environ.get(name)
                for name in (
                    "GCC_EXEC_PREFIX", "COMPILER_PATH", "LIBRARY_PATH", "CPATH", "C_INCLUDE_PATH", "SOURCE_DATE_EPOCH"
                )
            },
        }
        # Stable CPU identity/feature lines (the full file includes varying MHz).
        cpu_lines = Path("/proc/cpuinfo").read_text().splitlines()
        identity = sorted(
            {
                line
                for line in cpu_lines
                if line.split(":", 1)[0].strip()
                in {"vendor_id", "cpu family", "model", "model name", "stepping", "flags", "Features", "CPU architecture"}
            }
        )
        signature["cpu_sha256"] = hashlib.sha256("\n".join(identity).encode()).hexdigest()
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        entry = root / f"{cache_name}-{key}"
        hit = entry.exists()
        if not hit:
            library = staging / "nlp.so"
            command = [compiler, "-fPIC", "-shared", *compiler_flags, str(source), "-o", str(library), "-lm"]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(f"NLP callback compilation failed:\n{completed.stdout}{completed.stderr}")
            # Verify loadability before publishing the artifact.
            check_solver = casadi.nlpsol("nlpsol", plugin_name, str(library), options)
            del check_solver
            (staging / "manifest.json").write_text(
                json.dumps({"signature": signature, "library_sha256": _sha256(library)}, indent=2) + "\n"
            )
            try:
                os.rename(staging, entry)
            except OSError as error:
                # An identical concurrent build may have published first.
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                hit = True
        library = _validate_entry(entry, signature)
        solver = casadi.nlpsol("nlpsol", plugin_name, str(library), options)
        return solver, {"hit": hit, "key": key, "library": str(library)}
