"""Benchmark post-shake penalty grouping on an uncompiled MX NLP.

The command is normally recovered from a cycling FHO audit log.  It builds the
same final NLP but deliberately stops before IPOPT iterates.  No solver
callback is replaced or used by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import statistics
import subprocess
import sys
import time

import casadi as ca
import numpy as np


class _CapturedNlp(BaseException):
    pass


class _StoppingSolver:
    def __init__(self, solver):
        self.solver = solver

    def call(self, _limits):
        raise _CapturedNlp()

    def stats(self):  # pragma: no cover - generic_solve never reaches it
        return self.solver.stats()


def _command_from_log(path: Path) -> list[str]:
    content = json.loads(path.read_text()) if path.suffix == ".json" else None
    if content is not None:
        return list(content)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("command: "):
            return shlex.split(line.removeprefix("command: "))
    raise ValueError(f"No command found in {path}")


def _replace_or_append(cli: list[str], flag: str, value: str) -> None:
    if flag in cli:
        cli[cli.index(flag) + 1] = value
    else:
        cli.extend((flag, value))


def _time_call(function, *inputs, repeats: int) -> tuple[float, object]:
    samples = []
    output = None
    for _ in range(repeats):
        tic = time.perf_counter()
        output = function(*inputs)
        samples.append(time.perf_counter() - tic)
    return statistics.median(samples), output


def _equivalent_to_representative(candidate, representative, rng: np.random.Generator) -> bool:
    if candidate.decision_indices.__len__() != representative.decision_indices.__len__():
        return False
    if candidate.value.size1_out(0) != representative.value.size1_out(0):
        return False
    z = ca.DM(rng.normal(size=(len(candidate.decision_indices), 1)))
    lam = ca.DM(rng.normal(size=(candidate.value.size1_out(0), 1)))
    for left, right, args in (
        (candidate.value, representative.value, (z,)),
        (candidate.jacobian, representative.jacobian, (z,)),
        (candidate.lagrangian_hessian, representative.lagrangian_hessian, (z, lam)),
    ):
        if np.max(np.abs(np.asarray(left(*args)) - np.asarray(right(*args))), initial=0.0) > 1e-12:
            return False
    return True


def _compile(function: ca.Function, directory: Path) -> tuple[ca.Function, float]:
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / f"{function.name()}.c"
    library = directory / f"{function.name()}.so"
    tic = time.perf_counter()
    generator = ca.CodeGenerator(source.name)
    generator.add(function)
    generator.generate(str(directory) + "/")
    subprocess.run(["gcc", "-O3", "-fPIC", "-shared", str(source), "-o", str(library), "-lm"], check=True)
    return ca.external(function.name(), str(library)), time.perf_counter() - tic


def _sparsity_signature(sparsity: ca.Sparsity) -> tuple:
    rows, columns = sparsity.get_triplet()
    return sparsity.size1(), sparsity.size2(), tuple(int(row) for row in rows), tuple(int(column) for column in columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hsl-library", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--map-threads", type=int, default=12)
    parser.add_argument("--max-local-x", type=int, default=128)
    parser.add_argument("--inspect-only", action="store_true", help="Report real groups without code generation")
    args = parser.parse_args()
    if args.repeats < 1 or args.map_threads < 1 or args.max_local_x < 1:
        parser.error("repeats, map-threads and max-local-x must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    command = _command_from_log(args.command_log.resolve())
    script, cli = Path(command[1]).resolve(), command[2:]
    _replace_or_append(cli, "--ipopt-hsl-library", str(args.hsl_library.resolve()))
    _replace_or_append(cli, "--output-json", str(output / "unused-result.json"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import bioptim.interfaces.interface_utils as iu
    import bioptim.interfaces.ipopt_interface as ii

    original_nlpsol = iu.nlpsol
    original_generic_solve = ii.generic_solve
    captured: dict[str, object] = {}

    def nlpsol(name, plugin, nlp, options):
        solver = original_nlpsol(name, plugin, nlp, options)
        captured["solver"] = solver
        return _StoppingSolver(solver)

    def generic_solve(interface, expand_during_shake_tree=False):
        captured["interface"] = interface
        return original_generic_solve(interface, expand_during_shake_tree)

    iu.nlpsol = nlpsol
    ii.generic_solve = generic_solve
    sys.argv = [str(script), *cli]
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    except _CapturedNlp:
        pass
    finally:
        iu.nlpsol = original_nlpsol
        ii.generic_solve = original_generic_solve

    solver = captured["solver"]
    nlp = solver.get_function("nlp_hess_l")
    x0 = ca.DM.zeros(nlp.size1_in(0), 1)
    sigma = 1.0
    lam = ca.DM.ones(nlp.size1_in(3), 1)
    native_time, native_value = _time_call(nlp, x0, ca.DM.zeros(nlp.size1_in(1), 1), sigma, lam, repeats=args.repeats)

    # The interface is captured by wrapping generic_solve in the same way as
    # the normal audit. It is set by the command's IpoptInterface solve.
    owner = captured.get("interface")
    if owner is None:
        raise RuntimeError("No IpoptInterface was captured")
    registry_tic = time.perf_counter()
    registry = owner.build_post_shake_penalty_registry()
    registry_time = time.perf_counter() - registry_tic
    print(f"REGISTRY_BUILT seconds={registry_time:.3f} terms={len(registry.terms)}", flush=True)
    terms = [
        term
        for term in registry.terms
        if term.metadata.kind in ("constraint", "thread_map_fragment")
        and term.hessian_sparsity.nnz()
        and len(term.decision_indices) <= args.max_local_x
    ]
    buckets: dict[tuple, list] = {}
    for term in terms:
        signature = (
            term.metadata.scope,
            term.metadata.penalty_name,
            len(term.decision_indices),
            term.value.size1_out(0),
            _sparsity_signature(term.jacobian_sparsity),
            _sparsity_signature(term.hessian_sparsity),
        )
        buckets.setdefault(signature, []).append(term)
    rng = np.random.default_rng(42)
    validated = []
    # A full FHO can contain thousands of structurally distinct singleton
    # terms.  Inspect the largest buckets first and validate a bounded sample:
    # this is enough to select a real repeated candidate without turning a
    # preparation benchmark into an O(N) Hessian campaign.
    for group in sorted(buckets.values(), key=len, reverse=True):
        if len(group) < 2:
            break
        representative = group[0]
        equal = [term for term in group[: min(len(group), 32)] if _equivalent_to_representative(term, representative, rng)]
        if len(equal) > 1:
            validated.append(equal)
    if not validated:
        raise RuntimeError("No repeated algebraically equivalent nonlinear post-shake term was found")
    group_summary = [
        {
            "name": group[0].metadata.penalty_name,
            "scope": group[0].metadata.scope,
            "thread_map_fragment": group[0].metadata.thread_map_fragment,
            "structural_count": len(group),
            "local_x": len(group[0].decision_indices),
            "local_hessian_nnz": group[0].hessian_sparsity.nnz(),
            "raw_hessian_entries_covered": len(group) * group[0].hessian_sparsity.nnz(),
            "raw_hessian_fraction_of_native": (
                len(group) * group[0].hessian_sparsity.nnz() / max(nlp.sparsity_out(0).nnz(), 1)
            ),
        }
        for group in sorted(buckets.values(), key=len, reverse=True)[:20]
    ]
    group = max(validated, key=len)
    representative = group[0]
    count = len(group)
    print(f"GROUP_SELECTED name={representative.metadata.penalty_name} count={count}", flush=True)
    if args.inspect_only:
        result = {
            "command": [command[0], str(script), *cli],
            "native_hessian_median_s": native_time,
            "native_hessian_upper_nnz": nlp.sparsity_out(0).nnz(),
            "registry_build_s": registry_time,
            "registry_terms": len(registry.terms),
            "nonlinear_constraint_terms": len(terms),
            "thread_map_fragment_terms": sum(term.metadata.thread_map_fragment for term in terms),
            "maximum_local_x": args.max_local_x,
            "signature_bucket_count": len(buckets),
            "algebraically_validated_repeated_group_count": len(validated),
            "candidate_groups": group_summary,
            "note": "Inspection only: no local C compilation or global callback was used.",
        }
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    local_x = ca.horzcat(*[ca.DM(rng.normal(size=(len(term.decision_indices), 1))) for term in group])
    local_lam = ca.horzcat(*[ca.DM.ones(term.value.size1_out(0), 1) for term in group])
    print("COMPACTING_SELECTED_KERNEL", flush=True)
    compact_hessian = representative.lagrangian_hessian.expand()
    vm_map = compact_hessian.map(count, "thread", args.map_threads)
    vm_time, vm_value = _time_call(vm_map, local_x, local_lam, repeats=args.repeats)
    compiled, compile_time = _compile(compact_hessian, output / "compiled")
    print(f"KERNEL_COMPILED seconds={compile_time:.3f}", flush=True)
    c_map = compiled.map(count, "thread", args.map_threads)
    print("COMPILED_MAP_BUILT", flush=True)
    c_time, c_value = _time_call(c_map, local_x, local_lam, repeats=args.repeats)
    result = {
        "command": [command[0], str(script), *cli],
        "native_hessian_median_s": native_time,
        "native_hessian_upper_nnz": nlp.sparsity_out(0).nnz(),
        "registry_build_s": registry_time,
        "registry_terms": len(registry.terms),
        "nonlinear_constraint_terms": len(terms),
        "maximum_local_x": args.max_local_x,
        "signature_bucket_count": len(buckets),
        "algebraically_validated_repeated_group_count": len(validated),
        "largest_group": {
            "name": representative.metadata.penalty_name,
            "scope": representative.metadata.scope,
            "count": count,
            "local_x": len(representative.decision_indices),
            "local_hessian_nnz": representative.hessian_sparsity.nnz(),
            "raw_hessian_entries_covered": count * representative.hessian_sparsity.nnz(),
            "raw_hessian_fraction_of_native": (
                count * representative.hessian_sparsity.nnz() / max(nlp.sparsity_out(0).nnz(), 1)
            ),
            "vm_map_median_s": vm_time,
            "compiled_map_median_s": c_time,
            "compile_s": compile_time,
            "compiled_speedup": vm_time / max(c_time, 1e-15),
            "max_abs_vm_vs_compiled": float(np.max(np.abs(np.asarray(vm_value) - np.asarray(c_value)), initial=0.0)),
        },
        "native_hessian_sha256": hashlib.sha256(np.asarray(native_value).tobytes()).hexdigest(),
        "note": "No global callback was changed; packet values have not been scattered into nlp_hess_l.",
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
