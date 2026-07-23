"""Compare Bioptim nonlinear-programming solvers on representative OCPs.

Run from the repository root, for example::

    python -m benchmarks.solver_benchmark --solvers ipopt fatrop madnlp --sizes 20 50
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import casadi as cas
import numpy as np

import bioptim
from bioptim import ObjectiveFcn, ObjectiveList, OrderingStrategy, Solver
from bioptim.examples.getting_started.example_inequality_constraint import prepare_ocp as prepare_contact_inequality
from bioptim.examples.getting_started.basic_ocp import prepare_ocp as prepare_pendulum
from bioptim.examples.toy_examples.acados.cube import prepare_ocp as prepare_cube
from bioptim.examples.toy_examples.acados.static_arm import prepare_ocp as prepare_static_arm
from bioptim.examples.toy_examples.holonomic_constraints.arm26_pendulum_swingup_muscle import (
    prepare_ocp as prepare_holonomic_muscle,
)
from bioptim.examples.toy_examples.optimal_time_ocp.multiphase_time_constraint import (
    prepare_ocp as prepare_multiphase,
)
from bioptim.examples.toy_examples.optimal_time_ocp.time_constraint import prepare_ocp as prepare_free_time
from bioptim.examples.utils import ExampleUtils


SOLVER_NAMES = ("ipopt", "fatrop", "acados", "madnlp")
CASE_NAMES = (
    "pendulum",
    "cube",
    "static_arm",
    "free_time",
    "multiphase",
    "contact_inequality",
    "holonomic_muscle",
)
DEFAULT_CASE_NAMES = CASE_NAMES[:-1]


@dataclass
class RunResult:
    case: str
    solver: str
    n_shooting: int
    repetition: int
    temperature: str
    outcome: str
    build_s: float | None = None
    solve_wall_s: float | None = None
    solver_s: float | None = None
    iterations: int | None = None
    status: int | None = None
    cost: float | None = None
    max_constraint_violation: float | None = None
    inf_pr: float | None = None
    inf_du: float | None = None
    error: str | None = None


def nlpsol_available(name: str) -> tuple[bool, str | None]:
    try:
        available = bool(cas.has_nlpsol(name))
    except (RuntimeError, AttributeError) as error:
        return False, str(error)
    return available, None if available else f"CasADi nlpsol plugin '{name}' is unavailable"


def solver_available(name: str) -> tuple[bool, str | None]:
    if name in ("ipopt", "fatrop", "madnlp"):
        return nlpsol_available(name)
    if name == "acados":
        available = importlib.util.find_spec("acados_template") is not None
        return available, None if available else "Python package 'acados_template' is unavailable"
    raise ValueError(f"Unknown solver: {name}")


def make_solver(name: str, tolerance: float, max_iterations: int, acados_dir: str | None):
    factories = {
        "ipopt": Solver.IPOPT,
        "fatrop": Solver.FATROP,
        "madnlp": Solver.MADNLP,
        "acados": Solver.ACADOS,
    }
    solver = factories[name]()
    solver.set_convergence_tolerance(tolerance)
    solver.set_constraint_tolerance(tolerance)
    solver.set_maximum_iterations(max_iterations)
    solver.set_print_level("ERROR" if name == "madnlp" else 0)
    if name == "acados" and acados_dir:
        solver.set_acados_dir(acados_dir)
    return solver


def prepare_case(case: str, n_shooting: int):
    common = {"n_shooting": n_shooting, "ordering_strategy": OrderingStrategy.TIME_MAJOR}
    if case == "pendulum":
        return prepare_pendulum(
            ExampleUtils.folder + "/models/pendulum.bioMod",
            final_time=1.0,
            n_threads=1,
            **common,
        )
    if case == "cube":
        ocp = prepare_cube(
            ExampleUtils.folder + "/models/cube.bioMod",
            tf=2.0,
            use_sx=True,
            **common,
        )
        objectives = ObjectiveList()
        objectives.add(
            ObjectiveFcn.Mayer.MINIMIZE_STATE,
            key="q",
            weight=1000,
            index=[0, 1],
            target=np.array([[1.0, 2.0]]).T,
            multi_thread=False,
        )
        objectives.add(
            ObjectiveFcn.Mayer.MINIMIZE_STATE,
            key="q",
            weight=10000,
            index=[2],
            target=np.array([[3.0]]),
            multi_thread=False,
        )
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, multi_thread=False)
        ocp.update_objectives(objectives)
        return ocp
    if case == "static_arm":
        return prepare_static_arm(
            ExampleUtils.folder + "/models/arm26.bioMod",
            final_time=1.0,
            use_sx=True,
            n_threads=1,
            **common,
        )
    if case == "free_time":
        return prepare_free_time(
            ExampleUtils.folder + "/models/pendulum.bioMod",
            final_time=0.8,
            time_min=0.6,
            time_max=1.0,
            **common,
        )
    if case == "multiphase":
        return prepare_multiphase(
            biorbd_model_path=ExampleUtils.folder + "/models/cube.bioMod",
            final_time=(1.0, 1.0, 1.0),
            time_min=(0.5, 0.5, 0.5),
            time_max=(2.0, 2.0, 2.0),
            n_shooting=(n_shooting, n_shooting, n_shooting),
            ordering_strategy=OrderingStrategy.TIME_MAJOR,
        )
    if case == "contact_inequality":
        return prepare_contact_inequality(
            ExampleUtils.folder + "/models/2segments_4dof_2contacts.bioMod",
            phase_time=0.3,
            n_shooting=n_shooting,
            min_bound=50,
            max_bound=np.inf,
            mu=0.2,
            ordering_strategy=OrderingStrategy.TIME_MAJOR,
        )
    if case == "holonomic_muscle":
        ocp, _ = prepare_holonomic_muscle(
            ExampleUtils.folder + "/models/arm26_w_pendulum.bioMod",
            n_shooting=n_shooting,
            final_time=0.5,
            n_threads=1,
            ordering_strategy=OrderingStrategy.TIME_MAJOR,
        )
        return ocp
    raise ValueError(f"Unknown benchmark case: {case}")


def optional_float(value) -> float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return float(array.squeeze()) if array.size == 1 else None


def run_once(
    case: str,
    solver_name: str,
    n_shooting: int,
    repetition: int,
    temperature: str,
    tolerance: float,
    max_iterations: int,
    acados_dir: str | None,
) -> RunResult:
    try:
        start = time.perf_counter()
        ocp = prepare_case(case, n_shooting)
        build_s = time.perf_counter() - start
        solver = make_solver(solver_name, tolerance, max_iterations, acados_dir)

        start = time.perf_counter()
        solution = ocp.solve(solver)
        solve_wall_s = time.perf_counter() - start
        constraints = solution.constraints
        max_violation = None
        if constraints is not None:
            constraint_array = np.asarray(constraints, dtype=float)
            lower_bounds = np.asarray(ocp.ocp_solver.limits["lbg"], dtype=float)
            upper_bounds = np.asarray(ocp.ocp_solver.limits["ubg"], dtype=float)
            lower_violation = np.maximum(lower_bounds - constraint_array, 0.0)
            upper_violation = np.maximum(constraint_array - upper_bounds, 0.0)
            max_violation = (
                float(np.max(np.maximum(lower_violation, upper_violation))) if constraint_array.size else 0.0
            )

        if solution.status != 0:
            outcome = "solver_failure"
        elif max_violation is not None and max_violation > 10 * tolerance:
            outcome = "infeasible"
        else:
            outcome = "success"

        return RunResult(
            case=case,
            solver=solver_name,
            n_shooting=n_shooting,
            repetition=repetition,
            temperature=temperature,
            outcome=outcome,
            build_s=build_s,
            solve_wall_s=solve_wall_s,
            solver_s=optional_float(solution.solver_time_to_optimize),
            iterations=solution.iterations,
            status=solution.status,
            cost=optional_float(solution.cost),
            max_constraint_violation=max_violation,
            inf_pr=optional_float(solution.inf_pr),
            inf_du=optional_float(solution.inf_du),
        )
    except Exception as error:  # A benchmark must report one solver failure without aborting the matrix.
        return RunResult(
            case=case,
            solver=solver_name,
            n_shooting=n_shooting,
            repetition=repetition,
            temperature=temperature,
            outcome="exception",
            error=f"{type(error).__name__}: {error}",
        )


def median_or_none(rows: list[RunResult], field: str) -> float | None:
    values = [getattr(row, field) for row in rows if getattr(row, field) is not None]
    return statistics.median(values) if values else None


def summarize(rows: list[RunResult]) -> list[dict]:
    groups: dict[tuple[str, str, int], list[RunResult]] = {}
    for row in rows:
        groups.setdefault((row.case, row.solver, row.n_shooting), []).append(row)

    summary = []
    for (case, solver, n_shooting), group in sorted(groups.items()):
        successful = [row for row in group if row.outcome == "success"]
        cold = [row for row in successful if row.temperature == "cold"]
        hot = [row for row in successful if row.temperature == "hot"]
        summary.append(
            {
                "case": case,
                "solver": solver,
                "n_shooting": n_shooting,
                "successful_runs": len(successful),
                "total_runs": len(group),
                "cold_solve_wall_s": median_or_none(cold, "solve_wall_s"),
                "hot_solve_wall_s": median_or_none(hot, "solve_wall_s"),
                "cold_solver_s": median_or_none(cold, "solver_s"),
                "hot_solver_s": median_or_none(hot, "solver_s"),
                "median_build_s": median_or_none(successful, "build_s"),
                "median_solve_wall_s": median_or_none(successful, "solve_wall_s"),
                "median_solver_s": median_or_none(successful, "solver_s"),
                "median_iterations": median_or_none(successful, "iterations"),
                "cost": successful[-1].cost if successful else None,
                "max_constraint_violation": (
                    max(
                        row.max_constraint_violation
                        for row in successful
                        if row.max_constraint_violation is not None
                    )
                    if any(row.max_constraint_violation is not None for row in successful)
                    else None
                ),
            }
        )
    return summary


def write_results(output: Path, metadata: dict, rows: list[RunResult]) -> tuple[Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    csv_path = output.with_suffix(".csv")
    payload = {"metadata": metadata, "summary": summarize(rows), "runs": [asdict(row) for row in rows]}
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return json_path, csv_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASE_NAMES, default=list(DEFAULT_CASE_NAMES))
    parser.add_argument("--solvers", nargs="+", choices=SOLVER_NAMES, default=list(SOLVER_NAMES))
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--acados-dir")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repetitions < 1 or args.warmups < 0 or any(size < 1 for size in args.sizes):
        raise ValueError("repetitions and sizes must be positive; warmups must be non-negative")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("benchmarks/results") / f"solver_benchmark_{timestamp}"
    metadata = {
        "created_at_utc": timestamp,
        "python": sys.version,
        "platform": platform.platform(),
        "bioptim": bioptim.__version__,
        "casadi": cas.__version__,
        "cases": args.cases,
        "sizes": args.sizes,
        "repetitions": args.repetitions,
        "warmups": args.warmups,
        "tolerance": args.tolerance,
        "max_iterations": args.max_iterations,
    }
    rows: list[RunResult] = []
    for case in args.cases:
        for solver_name in args.solvers:
            available, reason = solver_available(solver_name)
            for n_shooting in args.sizes:
                if not available:
                    rows.append(
                        RunResult(
                            case=case,
                            solver=solver_name,
                            n_shooting=n_shooting,
                            repetition=0,
                            temperature="n/a",
                            outcome="unavailable",
                            error=reason,
                        )
                    )
                    continue
                if args.warmups:
                    for warmup in range(args.warmups):
                        warmup_result = run_once(
                            case,
                            solver_name,
                            n_shooting,
                            -(warmup + 1),
                            "cold" if warmup == 0 else "warmup",
                            args.tolerance,
                            args.max_iterations,
                            args.acados_dir,
                        )
                        rows.append(warmup_result)
                        if warmup_result.outcome != "success":
                            break
                    else:
                        rows.extend(
                            run_once(
                                case,
                                solver_name,
                                n_shooting,
                                repetition,
                                "hot",
                                args.tolerance,
                                args.max_iterations,
                                args.acados_dir,
                            )
                            for repetition in range(1, args.repetitions + 1)
                        )
                else:
                    rows.extend(
                        run_once(
                            case,
                            solver_name,
                            n_shooting,
                            repetition,
                            "cold" if repetition == 1 else "hot",
                            args.tolerance,
                            args.max_iterations,
                            args.acados_dir,
                        )
                        for repetition in range(1, args.repetitions + 1)
                    )

    json_path, csv_path = write_results(output, metadata, rows)
    print(json.dumps(summarize(rows), indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    return int(not any(row.outcome == "success" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
