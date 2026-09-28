"""Offline benchmark for cached compiled exact STATE_CONTINUITY Hessian packets.

It rebuilds the historical FHO3 MX NLP but never replaces ``nlp_hess_l`` or
calls IPOPT. It compares the existing VM local Hessian kernel with generated C
for that same exact Hessian and uses a cached ``numpy.bincount`` scatter plan.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import os
import runpy
import shlex
import shutil
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

    def stats(self):
        return self.solver.stats()


@dataclass(frozen=True)
class ScatterPlan:
    """All callback-invariant indexing and sparse structure."""

    x_indices: np.ndarray
    lambda_indices: np.ndarray
    source_nonzero_offsets: np.ndarray
    destination_slots: np.ndarray
    global_nnz: int


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
    values, output = [], None
    for _ in range(repeats):
        tic = time.perf_counter()
        output = function(*args)
        values.append(time.perf_counter() - tic)
    return statistics.median(values), output


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


def _make_plan(packet, hess) -> ScatterPlan:
    """Resolve gathers and lower-triangle global sparse slots once per NLP."""
    n_fragments = len(packet.metadata)
    local_size = len(packet.decision_indices[0])
    if any(len(indices) != local_size for indices in packet.decision_indices):
        raise ValueError("STATE_CONTINUITY packet is not homogeneous")
    x_indices = np.asarray(packet.decision_indices, dtype=np.int64).T
    lambda_indices = np.asarray(
        [np.arange(item.g_row_start, item.g_row_stop, dtype=np.int64) for item in packet.metadata], dtype=np.int64
    ).T
    local_rows, local_cols = packet.hessian_sparsity.get_triplet()
    local_rows, local_cols = np.asarray(local_rows), np.asarray(local_cols)
    keep = local_rows >= local_cols
    local_sparse_offsets = np.flatnonzero(keep)
    local_rows, local_cols = local_rows[keep], local_cols[keep]
    global_rows, global_cols = hess.sparsity_out(0).get_triplet()
    slots = {(int(row), int(col)): index for index, (row, col) in enumerate(zip(global_rows, global_cols))}
    destinations = np.empty((n_fragments, len(local_rows)), dtype=np.int64)
    for stage, indices in enumerate(packet.decision_indices):
        for entry, (local_row, local_col) in enumerate(zip(local_rows, local_cols)):
            pair = (indices[local_row], indices[local_col])
            destinations[stage, entry] = slots.get(pair, slots.get((pair[1], pair[0]), -1))
    if np.any(destinations < 0):
        raise RuntimeError("Packet local sparsity is absent from nlp_hess_l sparsity")
    # ``Function.map`` horizontally concatenates sparse outputs, so its
    # ``nonzeros()`` are one local sparse block after the other. Retain only
    # the lower triangle directly in sparse storage; materialising the dense
    # 158 x (158*90) map output would otherwise dominate the assembly time.
    stage_offsets = packet.hessian_sparsity.nnz() * np.arange(n_fragments, dtype=np.int64)[:, None]
    return ScatterPlan(
        x_indices=x_indices,
        lambda_indices=lambda_indices,
        source_nonzero_offsets=(stage_offsets + local_sparse_offsets[None, :]).reshape(-1),
        destination_slots=destinations.reshape(-1),
        global_nnz=len(global_rows),
    )


def _gather_plan(plan: ScatterPlan, x: ca.DM, lam_g: ca.DM) -> tuple[ca.DM, ca.DM]:
    x_values = np.asarray(x, dtype=float).reshape(-1)
    lambda_values = np.asarray(lam_g, dtype=float).reshape(-1)
    return ca.DM(x_values[plan.x_indices]), ca.DM(lambda_values[plan.lambda_indices])


def _scatter_bincount(mapped_value: ca.DM, plan: ScatterPlan) -> np.ndarray:
    """Vectorised exact sparse scatter-add: no Python per-entry loop."""
    values = np.asarray(mapped_value.nonzeros(), dtype=float).reshape(-1)
    return np.bincount(
        plan.destination_slots,
        weights=values[plan.source_nonzero_offsets],
        minlength=plan.global_nnz,
    )


def _compile_exact_hessian(function: ca.Function, destination: Path, optimization: str) -> tuple[ca.Function, float]:
    """Generate C for the already exact local Lagrangian Hessian Function."""
    compiler = shutil.which("gcc")
    if compiler is None:
        raise RuntimeError("gcc is required for --compile-local-kernel")
    destination.mkdir(parents=True, exist_ok=True)
    stem = "state_continuity_exact_hessian"
    source, shared = destination / f"{stem}.c", destination / f"{stem}.so"
    tic, old_cwd = time.perf_counter(), Path.cwd()
    try:
        os.chdir(destination)
        # CasADi 3.7 accepts a basename, not an absolute generation path.
        # Keep the function's own MX output expression: reconstructing a new
        # Function from ``mx_out`` would treat its output symbol as free.
        function.generate(f"{stem}.c", {"with_header": True})
    finally:
        os.chdir(old_cwd)
    completed = subprocess.run(
        [compiler, optimization, "-fPIC", "-shared", str(source), "-o", str(shared)],
        text=True, capture_output=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"C compilation failed:\n{completed.stderr}")
    return ca.external(function.name(), str(shared)), time.perf_counter() - tic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hsl-library", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3, 12])
    parser.add_argument("--compile-local-kernel", action="store_true")
    parser.add_argument(
        "--compiler-optimization", choices=["-O0", "-O1", "-O2", "-O3"], default="-O0",
        help="C optimisation for the large exact Hessian kernel; O0 is the robust baseline.",
    )
    args = parser.parse_args()
    if args.repeats < 1 or any(worker < 1 for worker in args.workers):
        parser.error("repeats and workers must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    solver, interface, command = _capture(_command_from_log(args.command_log.resolve()), output, args.hsl_library)
    hess, x = solver.get_function("nlp_hess_l"), ca.DM(interface.ocp.init_vector)
    p, ng = ca.DM.zeros(hess.size1_in(1), 1), hess.size1_in(3)
    import bioptim.interfaces.interface_utils as iu

    tic = time.perf_counter()
    registry = interface.build_post_shake_penalty_registry(materialize=False)
    raw_registry_s = time.perf_counter() - tic
    tic = time.perf_counter()
    packet = registry.build_pre_shake_thread_map_packet(
        interface.ocp.variables_vector,
        lambda expression: iu._normalize_pre_shake_thread_map_fragment(
            interface.ocp, expression, interface.ocp.variables_vector, interface.ocp.bounds_vectors
        ), "STATE_CONTINUITY",
    )
    packet_build_s = time.perf_counter() - tic
    tic = time.perf_counter()
    plan = _make_plan(packet, hess)
    scatter_plan_build_s = time.perf_counter() - tic
    rng, lam_g = np.random.default_rng(20260928), None
    lam_g = ca.DM(rng.normal(size=(ng, 1)))
    lam_only_continuity = ca.DM.zeros(ng, 1)
    for item in packet.metadata:
        lam_only_continuity[item.g_row_start : item.g_row_stop] = lam_g[item.g_row_start : item.g_row_stop]
    native_s, native = _time(hess, x, p, 0.0, lam_only_continuity, repeats=args.repeats)
    gather_s, (gathered_x, gathered_lam) = _time(_gather_plan, plan, x, lam_g, repeats=args.repeats)
    evaluators: list[tuple[str, ca.Function, float]] = [("casadi_vm_exact_hessian", packet.lagrangian_hessian, 0.0)]
    max_kernel_error = None
    if args.compile_local_kernel:
        compiled, compile_s = _compile_exact_hessian(
            packet.lagrangian_hessian, output / "generated", args.compiler_optimization
        )
        reference = np.asarray(packet.lagrangian_hessian(gathered_x[:, 0], gathered_lam[:, 0]))
        compiled_value = np.asarray(compiled(gathered_x[:, 0], gathered_lam[:, 0]))
        max_kernel_error = float(np.max(np.abs(reference - compiled_value), initial=0.0))
        if max_kernel_error > 1e-10:
            raise RuntimeError(f"Compiled local Hessian differs from VM: {max_kernel_error}")
        evaluators.append(("generated_c_exact_hessian", compiled, compile_s))
    native_values, measurements = np.asarray(native.nonzeros(), dtype=float), []
    for evaluator_name, evaluator, compile_s in evaluators:
        for workers in dict.fromkeys(args.workers):
            tic = time.perf_counter()
            mapped = evaluator.map(len(packet.metadata), "thread", workers)
            map_build_s = time.perf_counter() - tic
            eval_s, mapped_value = _time(mapped, gathered_x, gathered_lam, repeats=args.repeats)
            scatter_s, scattered = _time(_scatter_bincount, mapped_value, plan, repeats=args.repeats)
            delta = scattered - native_values
            max_abs = float(np.max(np.abs(delta), initial=0.0))
            above = int(np.count_nonzero(np.abs(delta) > 1e-10))
            if max_abs > 1e-10:
                raise RuntimeError(f"{evaluator_name}/{workers}: packet mismatch {max_abs}")
            measurements.append({
                "evaluator": evaluator_name, "workers": workers,
                "one_time_codegen_compile_s": compile_s, "one_time_map_build_s": map_build_s,
                "gather_median_s": gather_s, "local_hessian_eval_median_s": eval_s,
                "vectorized_scatter_median_s": scatter_s,
                "cached_packet_callback_median_s": gather_s + eval_s + scatter_s,
                "native_nlp_hess_l_median_s": native_s,
                "speedup_vs_native": native_s / max(gather_s + eval_s + scatter_s, 1e-15),
                "max_abs_packet_minus_native": max_abs, "entries_above_1e-10": above,
            })
    result = {
        "mode": "offline_prototype_no_ipopt_callback_no_solve", "command": command,
        "nx": int(hess.size1_in(0)), "ng": int(ng), "global_hessian_triangular_nnz": plan.global_nnz,
        "one_time_raw_registry_build_s": raw_registry_s, "one_time_packet_build_s": packet_build_s,
        "one_time_scatter_plan_build_s": scatter_plan_build_s,
        "state_continuity_fragments": len(packet.metadata), "fragment_local_x": len(packet.decision_indices[0]),
        "fragment_g_rows": packet.value.size1_out(0), "fragment_local_hessian_nnz": packet.hessian_sparsity.nnz(),
        "compiled_kernel_max_abs_error": max_kernel_error, "measurements": measurements,
        "compiler_optimization": args.compiler_optimization if args.compile_local_kernel else None,
        "interpretation": "Exact packet reconstruction of nlp_hess_l with sigma=0 and multipliers only on STATE_CONTINUITY. The generated C contains the already-derived local Lagrangian Hessian; no callback is installed and IPOPT is never run.",
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
