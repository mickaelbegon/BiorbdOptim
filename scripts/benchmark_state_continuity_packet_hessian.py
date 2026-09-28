"""Prototype exact STATE_CONTINUITY Hessian packets without changing IPOPT.

The script rebuilds the historical cycling FHO3 MX NLP, captures its normal
``nlp_hess_l`` callback, then independently evaluates the pre-``Function.map``
STATE_CONTINUITY fragments exposed by :class:`PostShakePenaltyRegistry`.
Their local exact Lagrangian Hessians are scatter-added into the *existing*
global Hessian sparsity pattern.  This is deliberately a measurement tool:
it never installs a callback or runs an IPOPT iteration.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import runpy
import shlex
import statistics
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
    if path.suffix == ".json":
        return list(json.loads(path.read_text()))
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("command: "):
            return shlex.split(line.removeprefix("command: "))
    raise ValueError(f"No command found in {path}")


def _replace_or_append(cli: list[str], flag: str, value: str) -> None:
    if flag in cli:
        cli[cli.index(flag) + 1] = value
    else:
        cli.extend((flag, value))


def _time(function, *args, repeats: int):
    values = []
    output = None
    for _ in range(repeats):
        tic = time.perf_counter()
        output = function(*args)
        values.append(time.perf_counter() - tic)
    return statistics.median(values), output


def _equivalent(left, right, rng: np.random.Generator) -> bool:
    if len(left.decision_indices) != len(right.decision_indices):
        return False
    if left.value.size1_out(0) != right.value.size1_out(0):
        return False
    x = ca.DM(rng.normal(size=(len(left.decision_indices), 1)))
    lam = ca.DM(rng.normal(size=(left.value.size1_out(0), 1)))
    for a, b, args in (
        (left.value, right.value, (x,)),
        (left.jacobian, right.jacobian, (x,)),
        (left.lagrangian_hessian, right.lagrangian_hessian, (x, lam)),
    ):
        if np.max(np.abs(np.asarray(a(*args)) - np.asarray(b(*args))), initial=0.0) > 1e-11:
            return False
    return True


def _capture(command: list[str], output: Path, hsl_library: Path):
    script, cli = Path(command[1]).resolve(), command[2:]
    _replace_or_append(cli, "--ipopt-hsl-library", str(hsl_library.resolve()))
    _replace_or_append(cli, "--output-json", str(output / "unused-result.json"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import bioptim.interfaces.interface_utils as iu
    import bioptim.interfaces.ipopt_interface as ii

    original_nlpsol, original_generic_solve = iu.nlpsol, ii.generic_solve
    captured: dict[str, object] = {}

    def nlpsol(name, plugin, nlp, options):
        solver = original_nlpsol(name, plugin, nlp, options)
        captured["solver"] = solver
        return _StoppingSolver(solver)

    def generic_solve(interface, expand_during_shake_tree=False):
        captured["interface"] = interface
        return original_generic_solve(interface, expand_during_shake_tree)

    iu.nlpsol, ii.generic_solve = nlpsol, generic_solve
    sys.argv = [str(script), *cli]
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    except _CapturedNlp:
        pass
    finally:
        iu.nlpsol, ii.generic_solve = original_nlpsol, original_generic_solve
    return captured["solver"], captured["interface"], [command[0], str(script), *cli]


def _gather(metadata, decision_indices, x: ca.DM, lam_g: ca.DM) -> tuple[ca.DM, ca.DM]:
    rows = slice(metadata.g_row_start, metadata.g_row_stop)
    return x[list(decision_indices)], lam_g[rows]


def _scatter(packet, local_hessians: list[np.ndarray], global_slots: dict[tuple[int, int], int], global_nnz: int) -> np.ndarray:
    """Scatter lower/upper local sparse triplets into canonical global slots."""
    values = np.zeros(global_nnz)
    for metadata, decision_indices, local_value in zip(packet.metadata, packet.decision_indices, local_hessians):
        local_rows, local_cols = packet.hessian_sparsity.get_triplet()
        # ``np.asarray`` materializes CasADi's sparse output densely, whereas
        # ``get_triplet`` is column-major sparse order. Index the dense view
        # explicitly instead of zipping it with the sparse triplet.
        for local_row, local_col in zip(local_rows, local_cols):
            # A Hessian is symmetric. The IPOPT callback stores one triangle,
            # so scatter exactly one local triangle to avoid double counting.
            if local_row < local_col:
                continue
            value = local_value[local_row, local_col]
            i, j = decision_indices[local_row], decision_indices[local_col]
            slot = global_slots.get((i, j))
            if slot is None:
                slot = global_slots.get((j, i))
            if slot is None:
                raise RuntimeError(f"Local entry ({i}, {j}) not in global Hessian sparsity")
            values[slot] += value
    return values


def _eval_individual(packet, x: ca.DM, lam_g: ca.DM, workers: int) -> list[np.ndarray]:
    inputs = [_gather(metadata, indices, x, lam_g) for metadata, indices in zip(packet.metadata, packet.decision_indices)]
    def call(item):
        _, (local_x, local_lam) = item
        return np.asarray(packet.lagrangian_hessian(local_x, local_lam))
    # CasADi Function calls are independent, but do not use a Python pool for
    # the main reported path: CasADi's native Function.map owns thread safety
    # and scheduling. This fallback exists only if a cycling instance is not
    # algebraically mappable.
    if workers == 1:
        return [call(item) for item in zip(packet.metadata, inputs)]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(call, zip(packet.metadata, inputs)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hsl-library", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3, 12])
    args = parser.parse_args()
    if args.repeats < 1 or any(worker < 1 for worker in args.workers):
        parser.error("repeats and workers must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    solver, interface, command = _capture(_command_from_log(args.command_log.resolve()), output, args.hsl_library)
    hess = solver.get_function("nlp_hess_l")
    # The zero vector is outside the FES model's physical domain and can
    # legitimately produce NaNs in second derivatives.  IPOPT's submitted
    # x0 is finite and lets this prototype make a meaningful exact comparison.
    x = ca.DM(interface.ocp.init_vector)
    p = ca.DM.zeros(hess.size1_in(1), 1)
    ng = hess.size1_in(3)
    registry_tic = time.perf_counter()
    registry = interface.build_post_shake_penalty_registry(materialize=False)
    raw_registry_s = time.perf_counter() - registry_tic
    import bioptim.interfaces.interface_utils as iu
    packet_tic = time.perf_counter()
    packet = registry.build_pre_shake_thread_map_packet(
        interface.ocp.variables_vector,
        lambda expression: iu._normalize_pre_shake_thread_map_fragment(
            interface.ocp, expression, interface.ocp.variables_vector, interface.ocp.bounds_vectors
        ),
        "STATE_CONTINUITY",
    )
    packet_build_s = time.perf_counter() - packet_tic
    if any(metadata.g_row_start is None for metadata in packet.metadata):
        raise RuntimeError("STATE_CONTINUITY packet has unresolved canonical g rows")
    rng = np.random.default_rng(20260928)
    # The packet's one representative comes from the non-threaded source.
    # Exact equality is assessed below by scattering all its mapped outputs
    # into the native FHO Hessian, rather than by materializing 91 MX terms.
    algebraically_mappable = True
    # A nontrivial multiplier field makes equality test both the local
    # Lagrangian Hessian and its scatter locations meaningful.
    lam_g = ca.DM(rng.normal(size=(ng, 1)))
    lam_only_continuity = ca.DM.zeros(ng, 1)
    for metadata in packet.metadata:
        start, stop = metadata.g_row_start, metadata.g_row_stop
        lam_only_continuity[start:stop] = lam_g[start:stop]
    native_s, native = _time(hess, x, p, 0.0, lam_only_continuity, repeats=args.repeats)
    global_rows, global_cols = hess.sparsity_out(0).get_triplet()
    global_slots = {(int(row), int(col)): k for k, (row, col) in enumerate(zip(global_rows, global_cols))}
    global_nnz = len(global_rows)
    gathered_x = ca.horzcat(*[_gather(metadata, indices, x, lam_g)[0] for metadata, indices in zip(packet.metadata, packet.decision_indices)])
    gathered_lam = ca.horzcat(*[_gather(metadata, indices, x, lam_g)[1] for metadata, indices in zip(packet.metadata, packet.decision_indices)])
    measurements = []
    for workers in dict.fromkeys(args.workers):
        if algebraically_mappable:
            mapped = packet.lagrangian_hessian.map(len(packet.metadata), "thread", workers)
            build_tic = time.perf_counter()
            # Map construction was deliberately measured separately.
            mapped = packet.lagrangian_hessian.map(len(packet.metadata), "thread", workers)
            map_build_s = time.perf_counter() - build_tic
            eval_s, mapped_value = _time(mapped, gathered_x, gathered_lam, repeats=args.repeats)
            # CasADi returns horizontally concatenated sparse matrices.
            local_hessians = [np.asarray(mapped_value[:, index * len(packet.decision_indices[0]):(index + 1) * len(packet.decision_indices[0])]) for index in range(len(packet.metadata))]
            evaluator = "casadi_map_thread"
        else:
            map_build_s = 0.0
            def run_individual():
                return _eval_individual(packet, x, lam_g, workers)
            eval_s, local_hessians = _time(run_individual, repeats=args.repeats)
            evaluator = "python_pool_fallback"
        scatter_s, scattered = _time(_scatter, packet, local_hessians, global_slots, global_nnz, repeats=args.repeats)
        # ``nlp_hess_l`` is sparse (49k stored triangular entries on FHO3).
        # Do not densify its 12k × 12k DM merely to compare the packet.
        native_values = np.asarray(native.nonzeros(), dtype=float)
        delta = scattered - native_values
        finite = bool(np.all(np.isfinite(native_values)) and np.all(np.isfinite(scattered)))
        max_abs = float(np.max(np.abs(delta), initial=0.0)) if finite else None
        nonzero_mismatch = int(np.count_nonzero(np.abs(delta) > 1e-10)) if finite else None
        measurements.append({
            "workers": workers,
            "evaluator": evaluator,
            "map_build_s": map_build_s,
            "local_hessian_eval_median_s": eval_s,
            "scatter_add_median_s": scatter_s,
            "packet_total_median_s": eval_s + scatter_s,
            "native_continuity_hessian_median_s": native_s,
            "speedup_vs_native_continuity": native_s / max(eval_s + scatter_s, 1e-15),
            "max_abs_packet_minus_native": max_abs,
            "entries_above_1e-10": nonzero_mismatch,
            "comparison_finite": finite,
        })
    result = {
        "mode": "prototype_only_no_ipopt_callback_no_solve",
        "command": command,
        "nx": int(hess.size1_in(0)), "ng": int(ng),
        "global_hessian_triangular_nnz": global_nnz,
        "raw_registry_build_s": raw_registry_s,
        "packet_build_s": packet_build_s,
        "state_continuity_fragments": len(packet.metadata),
        "fragment_local_x": len(packet.decision_indices[0]),
        "fragment_g_rows": packet.value.size1_out(0),
        "fragment_local_hessian_nnz": packet.hessian_sparsity.nnz(),
        "algebraically_mappable": algebraically_mappable,
        "native_continuity_hessian_median_s": native_s,
        "measurements": measurements,
        "interpretation": "Exact equivalence is assessed against nlp_hess_l with sigma=0 and multipliers nonzero only on canonical STATE_CONTINUITY rows. A product callback is not installed by this script.",
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
