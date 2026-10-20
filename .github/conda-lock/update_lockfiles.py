#!/usr/bin/env python3
"""Regenerate and validate the three Conda lockfiles in one command."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

from check_lockfiles import LOCKFILES, REPOSITORY_ROOT, SOURCE_FILES, SOURCE_HASH_FILE, check, compute_source_hash

CONDA_LOCK_VERSION = "4.0.2"
LOCK_PATHS = (*LOCKFILES.values(), SOURCE_HASH_FILE)


def run(*command: str, cwd: Path = REPOSITORY_ROOT, capture: bool = False) -> str:
    """Run a command and propagate its failure without masking its diagnostics."""
    result = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def git(*args: str) -> str:
    """Return Git output from the repository being updated."""
    return run("git", *args, capture=True)


def tool_cache() -> Path:
    """Keep the pinned generator outside the checkout and reuse it on later runs."""
    override = os.environ.get("BIOPTIM_CONDA_LOCK_CACHE")
    return Path(override) if override else Path.home() / ".cache" / "bioptim" / f"conda-lock-{CONDA_LOCK_VERSION}"


def conda_lock_executable() -> Path:
    """Find the pinned generator, installing it in an isolated cache if needed."""
    installed = shutil.which("conda-lock")
    if installed:
        version = run(installed, "--version", capture=True)
        if re.search(rf"\b{re.escape(CONDA_LOCK_VERSION)}\b", version):
            return Path(installed)

    cache = tool_cache()
    executable = cache / ("Scripts/conda-lock.exe" if os.name == "nt" else "bin/conda-lock")
    if not executable.is_file():
        print(f"Installing conda-lock {CONDA_LOCK_VERSION} in {cache}")
        venv.create(cache, with_pip=True)
        python = cache / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run(str(python), "-m", "pip", "install", "--disable-pip-version-check", f"conda-lock=={CONDA_LOCK_VERSION}")
    version = run(str(executable), "--version", capture=True)
    if not re.search(rf"\b{re.escape(CONDA_LOCK_VERSION)}\b", version):
        raise RuntimeError(f"Expected conda-lock {CONDA_LOCK_VERSION}, got {version!r}")
    return executable


def conda_solver() -> tuple[Path, bool]:
    """Prefer an installed Mamba solver and fall back to an existing Conda."""
    requested = os.environ.get("BIOPTIM_CONDA_SOLVER")
    candidates = [requested] if requested else ["micromamba", "mamba", "conda"]
    conda_executable = os.environ.get("CONDA_EXE")
    if not requested and conda_executable:
        sibling = Path(conda_executable).parent
        candidates.extend(str(sibling / name) for name in ("micromamba", "mamba", "conda"))
    if not requested:
        for installation in ("miniconda3", "miniforge3", "anaconda3", "mambaforge"):
            bin_directory = Path.home() / installation / ("Scripts" if os.name == "nt" else "bin")
            for name in ("micromamba", "mamba", "conda"):
                candidates.append(str(bin_directory / name))
    for candidate in candidates:
        found = shutil.which(candidate) if candidate else None
        if found:
            path = Path(found)
            return path, path.stem in {"micromamba", "mamba"}
    raise RuntimeError("No Conda solver found. Activate a Conda environment with mamba, micromamba or conda.")


def guard_existing_work(commit_and_push: bool) -> None:
    """Avoid replacing an existing lock edit or pushing unrelated local commits."""
    if git("status", "--porcelain", "--", *(str(path) for path in LOCK_PATHS)):
        raise RuntimeError("Lockfiles already have local changes; commit or review them before regenerating.")
    if not commit_and_push:
        return
    if git("diff", "--cached", "--name-only"):
        raise RuntimeError("Staged changes exist; commit or unstage them before --commit-and-push.")
    if git("status", "--porcelain", "--", *(str(path) for path in SOURCE_FILES)):
        raise RuntimeError("Dependency inputs have local changes; commit and push those before --commit-and-push.")
    branch = git("symbolic-ref", "--quiet", "--short", "HEAD")
    upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    remote = git("config", "--get", f"branch.{branch}.remote")
    if upstream != f"{remote}/{branch}":
        raise RuntimeError(f"Branch {branch!r} does not track the same branch name on {upstream!r}.")
    if git("rev-parse", "HEAD") != git("rev-parse", "@{u}"):
        raise RuntimeError("Local branch differs from its tracked remote. Synchronize it before --commit-and-push.")


def regenerate() -> bool:
    """Resolve once into a temporary tree and only copy validated results."""
    if not check(REPOSITORY_ROOT):
        print("Conda lockfiles are current; nothing to update.")
        return False

    solver, use_micromamba = conda_solver()
    executable = conda_lock_executable()
    with tempfile.TemporaryDirectory(prefix="bioptim-lockfiles-") as directory:
        staging = Path(directory)
        output = staging / "generated"
        output.mkdir()
        command = [
            str(executable),
            "lock",
            "--conda",
            str(solver),
            "--kind",
            "explicit",
            "--lockfile",
            str(staging / "conda-lock.yml"),
            "--virtual-package-spec",
            str(REPOSITORY_ROOT / ".github/conda-lock/virtual-packages.yml"),
        ]
        if use_micromamba:
            command.append("--micromamba")
        for platform in LOCKFILES:
            command.extend(("--platform", platform))
        for source in SOURCE_FILES[:2]:
            command.extend(("--file", str(REPOSITORY_ROOT / source)))
        command.extend(("--filename-template", str(output / "conda-{platform}.lock")))
        run(*command)

        verified = staging / "verified"
        for source in SOURCE_FILES:
            destination = verified / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPOSITORY_ROOT / source, destination)
        for platform, relative_path in LOCKFILES.items():
            destination = verified / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output / f"conda-{platform}.lock", destination)
        fingerprint = verified / SOURCE_HASH_FILE
        fingerprint.write_text(f"{compute_source_hash(verified)}\n")
        errors = check(verified)
        if errors:
            raise RuntimeError("Generated lockfiles failed validation: " + "; ".join(errors))

        for relative_path in LOCK_PATHS:
            shutil.copyfile(verified / relative_path, REPOSITORY_ROOT / relative_path)

    print("Updated and validated three Conda lockfiles and source-files.sha256.")
    return True


def main() -> int:
    """Update locks and optionally commit and push only the generated files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit-and-push", action="store_true", help="commit the generated files and push the tracked PR branch"
    )
    args = parser.parse_args()
    try:
        if not args.commit_and_push and not check(REPOSITORY_ROOT):
            print("Conda lockfiles are current; nothing to update.")
            return 0
        guard_existing_work(args.commit_and_push)
        changed = regenerate()
        if changed and args.commit_and_push:
            run("git", "add", "--", *(str(path) for path in LOCK_PATHS))
            run("git", "commit", "-m", "ci: refresh Conda lockfiles")
            branch = git("symbolic-ref", "--quiet", "--short", "HEAD")
            remote = git("config", "--get", f"branch.{branch}.remote")
            run("git", "push", remote, f"HEAD:refs/heads/{branch}")
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
