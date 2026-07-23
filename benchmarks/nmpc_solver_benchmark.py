"""Benchmark IPOPT and alpaqa on the cyclic NMPC example.

The IPOPT solution of one horizon can be saved and then reused as the common
initial point in another Python environment::

    python -m benchmarks.nmpc_solver_benchmark --solver ipopt --seed-output /tmp/nmpc_seed.npz
    python -m benchmarks.nmpc_solver_benchmark --solver ipopt --seed-input /tmp/nmpc_seed.npz
    python -m benchmarks.nmpc_solver_benchmark --solver alpaqa --seed-input /tmp/nmpc_seed.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import casadi as cas
import numpy as np

import bioptim
from bioptim import Solution, Solver
from bioptim.examples.toy_examples.moving_horizon_estimation.cyclic_nmpc import prepare_nmpc
from bioptim.examples.utils import ExampleUtils
from bioptim.misc.enums import SolverType


@dataclass
class WindowResult:
    solver: str
    repetition: int
    window: int
    status: int
    cost: float
    mean_wall_s: float
    solver_s: float
    max_constraint_violation: float
    iterations: int | None


def make_nmpc(cycle_len: int, cycle_duration: float, max_torque: float):
    return prepare_nmpc(
        ExampleUtils.folder + "/models/arm2.bioMod",
        cycle_len=cycle_len,
        cycle_duration=cycle_duration,
        max_torque=max_torque,
    )


def make_solver(name: str, tolerance: float, max_iterations: int, max_wall_time: float | None):
    if not cas.has_nlpsol(name):
        raise RuntimeError(f"CasADi nlpsol plugin '{name}' is unavailable")
    solver = Solver.IPOPT() if name == "ipopt" else Solver.ALPAQA()
    solver.set_convergence_tolerance(tolerance)
    solver.set_constraint_tolerance(tolerance)
    solver.set_maximum_iterations(max_iterations)
    solver.set_print_level(0)
    if name == "alpaqa":
        solver.set_alm_maximum_iterations(max_iterations)
        solver.set_lbfgs_memory(20)
        if max_wall_time is not None:
            solver.set_maximum_wall_time(max_wall_time)
    return solver


def solution_from_seed(nmpc, seed_path: Path) -> Solution:
    seed = np.load(seed_path)
    vector_key = "x" if "x" in seed else "vector"
    return Solution.from_dict(
        nmpc,
        {
            "x": cas.DM(seed[vector_key]),
            "f": cas.DM(0),
            "g": cas.DM.zeros(seed["lam_g"].size, 1),
            "lam_g": cas.DM(seed["lam_g"]),
            "lam_x": cas.DM(seed["lam_x"]),
            "lam_p": cas.DM.zeros(0, 1),
            "inf_pr": None,
            "inf_du": None,
            "solver_time_to_optimize": 0.0,
            "real_time_to_optimize": 0.0,
            "iter": 0,
            "status": 0,
            "solver": SolverType.IPOPT.value,
        },
    )


def solve_windows(nmpc, solver, warm_start: Solution | None, n_windows: int):
    def keep_going(_nmpc, window_index, _solution):
        return window_index < n_windows

    _, windows, _ = nmpc.solve(
        keep_going,
        solver=solver,
        warm_start=warm_start,
        # A missed real-time deadline is part of the benchmark result. Keep
        # advancing so later windows are measured as well.
        max_consecutive_failing=n_windows + 1,
        get_all_iterations=True,
    )
    return windows


def save_ipopt_seed(args, path: Path) -> None:
    if args.solver != "ipopt":
        raise ValueError("--seed-output requires --solver ipopt")
    nmpc = make_nmpc(args.cycle_len, args.cycle_duration, args.max_torque)
    solver = make_solver("ipopt", args.tolerance, args.max_iterations, None)
    solution = solve_windows(nmpc, solver, None, 1)[0]
    if solution.status != 0:
        raise RuntimeError(f"IPOPT seed solve failed with status {solution.status}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        x=np.asarray(solution.vector),
        lam_g=np.asarray(solution.lam_g),
        lam_x=np.asarray(solution.lam_x),
    )
    print(f"Saved IPOPT seed to {path}")


def iteration_count(solution) -> int | None:
    value = solution.iterations
    if isinstance(value, int):
        return value
    return None


def run(args) -> list[WindowResult]:
    results = []
    seed_path = Path(args.seed_input) if args.seed_input else None
    for repetition in range(args.repetitions):
        nmpc = make_nmpc(args.cycle_len, args.cycle_duration, args.max_torque)
        solver = make_solver(args.solver, args.tolerance, args.max_iterations, args.max_wall_time)
        warm_start = solution_from_seed(nmpc, seed_path) if seed_path else None
        wall_start = time.perf_counter()
        windows = solve_windows(nmpc, solver, warm_start, args.windows)
        total_wall = time.perf_counter() - wall_start
        for window_index, solution in enumerate(windows):
            constraints = np.asarray(solution.constraints, dtype=float)
            results.append(
                WindowResult(
                    solver=args.solver,
                    repetition=repetition,
                    window=window_index,
                    status=int(solution.status),
                    cost=float(solution.cost),
                    mean_wall_s=total_wall / len(windows),
                    solver_s=float(solution.solver_time_to_optimize),
                    max_constraint_violation=float(np.max(np.abs(constraints))),
                    iterations=iteration_count(solution),
                )
            )
    return results


def write_results(args, results: list[WindowResult]) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "casadi": cas.__version__,
        "bioptim": bioptim.__version__,
        "arguments": vars(args),
        "results": [asdict(result) for result in results],
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=WindowResult.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    successful = [result for result in results if result.status == 0]
    if successful:
        print(
            f"{args.solver}: {len(successful)}/{len(results)} successful windows, "
            f"median solver {statistics.median(r.solver_s for r in successful):.6f} s, "
            f"median cost {statistics.median(r.cost for r in successful):.9f}, "
            f"max violation {max(r.max_constraint_violation for r in successful):.3e}"
        )
    print(f"Results written to {output.with_suffix('.json')} and {output.with_suffix('.csv')}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("ipopt", "alpaqa"), required=True)
    parser.add_argument("--seed-input", help="IPOPT seed produced by --seed-output")
    parser.add_argument("--seed-output", help="Generate an IPOPT seed and exit")
    parser.add_argument("--cycle-len", type=int, default=20)
    parser.add_argument("--cycle-duration", type=float, default=1.0)
    parser.add_argument("--max-torque", type=float, default=50.0)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--max-wall-time", type=float, default=0.5)
    parser.add_argument("--output", default="benchmarks/results/nmpc_solver_benchmark")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed_output:
        save_ipopt_seed(args, Path(args.seed_output))
        return
    results = run(args)
    write_results(args, results)


if __name__ == "__main__":
    main()
