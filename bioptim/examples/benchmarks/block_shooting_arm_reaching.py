"""Benchmark DMS and block-shooting transcriptions on a muscle-driven reaching task.

Each measurement runs in a fresh subprocess so peak resident memory is comparable
between transcriptions. The benchmark reports NLP size, constraint-Jacobian
sparsity, construction time, solve time, iterations, objective value, and peak RSS.

Example
-------
python -m bioptim.examples.benchmarks.block_shooting_arm_reaching \
    --n-shooting 50 --blocks 1 5 10 25 50 --repeat 3 \
    --output block_shooting_arm_reaching.json
"""

import argparse
import ctypes
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import biorbd
import casadi

import bioptim
from bioptim import (
    BlockShooting,
    BoundsList,
    DynamicsOptions,
    InitialGuessList,
    MusclesBiorbdModel,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    PhaseDynamics,
    Solver,
)
from bioptim.examples.utils import ExampleUtils

RESULT_PREFIX = "BIOPTIM_BLOCK_BENCHMARK_RESULT="


def prepare_reaching_ocp(n_shooting: int, n_blocks: int | None) -> OptimalControlProgram:
    """Build a deterministic two-joint, six-muscle reaching problem."""
    model = MusclesBiorbdModel(
        ExampleUtils.folder + "/models/arm26_muscle_driven_ocp.bioMod",
        with_residual_torque=True,
    )

    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles")
    objectives.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
        first_marker="target",
        second_marker="COM_hand",
        weight=1000,
    )

    x_bounds = BoundsList()
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = (0.07, 1.4)
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, 0] = 0

    x_init = InitialGuessList()
    x_init["q"] = [1.57] * model.nb_q

    u_bounds = BoundsList()
    u_bounds["tau"] = [-1.0] * model.nb_tau, [1.0] * model.nb_tau
    u_bounds["muscles"] = [0.0] * model.nb_muscles, [1.0] * model.nb_muscles

    u_init = InitialGuessList()
    u_init["muscles"] = [0.5] * model.nb_muscles

    return OptimalControlProgram(
        model,
        n_shooting=n_shooting,
        phase_time=0.5,
        dynamics=DynamicsOptions(
            ode_solver=OdeSolver.RK4(),
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
            expand_dynamics=True,
        ),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objectives,
        block_shooting=None if n_blocks is None else BlockShooting(n_blocks=n_blocks),
        n_threads=1,
        use_sx=False,
    )


def _peak_rss_megabytes() -> float:
    if sys.platform == "win32":

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / 1024**2

    import resource

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak_rss / (1024**2 if sys.platform == "darwin" else 1024)


def run_case(
    n_shooting: int,
    n_blocks: int | None,
    solve: bool = True,
    max_iterations: int = 1000,
    hessian_approximation: str = "limited-memory",
) -> dict:
    """Run one isolated benchmark case and return serializable metrics."""
    build_start = time.perf_counter()
    ocp = prepare_reaching_ocp(n_shooting=n_shooting, n_blocks=n_blocks)
    build_seconds = time.perf_counter() - build_start

    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(max_iterations)
    solver.set_hessian_approximation(hessian_approximation)
    ocp.set_ocp_solver(solver)

    variables = ocp.variables_vector
    constraints, _ = ocp.ocp_solver.dispatch_bounds()
    jacobian_sparsity = casadi.jacobian(constraints, variables).sparsity()
    jacobian_entries = constraints.shape[0] * variables.shape[0]

    result = {
        "transcription": "DMS" if n_blocks is None else ("DSS" if n_blocks == 1 else f"B={n_blocks}"),
        "n_shooting": n_shooting,
        "n_blocks": n_blocks,
        "variables": int(variables.shape[0]),
        "constraints": int(constraints.shape[0]),
        "constraint_jacobian_nnz": int(jacobian_sparsity.nnz()),
        "constraint_jacobian_density": (float(jacobian_sparsity.nnz() / jacobian_entries) if jacobian_entries else 0.0),
        "build_seconds": build_seconds,
        "solve_seconds": None,
        "solver_seconds": None,
        "iterations": None,
        "status": None,
        "cost": None,
    }

    if solve:
        solve_start = time.perf_counter()
        solution = ocp.solve(solver)
        result.update(
            solve_seconds=time.perf_counter() - solve_start,
            solver_seconds=float(solution.solver_time_to_optimize),
            iterations=int(solution.iterations),
            status=int(solution.status),
            cost=float(solution.cost),
        )

    result["peak_rss_mb"] = _peak_rss_megabytes()
    return result


def _worker_command(args, case: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "bioptim.examples.benchmarks.block_shooting_arm_reaching",
        "--worker",
        "--case",
        case,
        "--n-shooting",
        str(args.n_shooting),
        "--max-iterations",
        str(args.max_iterations),
        "--hessian-approximation",
        args.hessian_approximation,
    ]
    if args.build_only:
        command.append("--build-only")
    return command


def _run_isolated(args, case: str) -> dict:
    completed = subprocess.run(
        _worker_command(args, case),
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"The benchmark worker for {case} failed with status {completed.returncode}."
            f"\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line.removeprefix(RESULT_PREFIX))
    raise RuntimeError(
        f"The benchmark worker returned no result.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def _summaries(results: list[dict]) -> list[dict]:
    summaries = []
    for transcription in dict.fromkeys(result["transcription"] for result in results):
        cases = [result for result in results if result["transcription"] == transcription]
        summary = dict(cases[0])
        for key in ("build_seconds", "solve_seconds", "solver_seconds", "iterations", "cost", "peak_rss_mb"):
            values = [case[key] for case in cases if case[key] is not None]
            summary[key] = statistics.median(values) if values else None
        summary["repetitions"] = len(cases)
        summaries.append(summary)
    return summaries


def _format_number(value, precision: int = 3) -> str:
    return "-" if value is None else f"{value:.{precision}f}"


def _print_table(summaries: list[dict]) -> None:
    headers = ("case", "vars", "cons", "jac nnz", "jac %", "build s", "solve s", "iter", "status", "cost", "RSS MB")
    rows = [headers]
    for result in summaries:
        rows.append(
            (
                result["transcription"],
                str(result["variables"]),
                str(result["constraints"]),
                str(result["constraint_jacobian_nnz"]),
                _format_number(100 * result["constraint_jacobian_density"], 2),
                _format_number(result["build_seconds"]),
                _format_number(result["solve_seconds"]),
                _format_number(result["iterations"], 0),
                _format_number(result["status"], 0),
                _format_number(result["cost"], 6),
                _format_number(result["peak_rss_mb"], 1),
            )
        )

    widths = [max(len(row[column]) for row in rows) for column in range(len(headers))]
    for row_index, row in enumerate(rows):
        print("  ".join(value.rjust(widths[column]) for column, value in enumerate(row)))
        if row_index == 0:
            print("  ".join("-" * width for width in widths))


def _metadata(args) -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "casadi": casadi.__version__,
        "biorbd": biorbd.__version__,
        "bioptim": bioptim.__version__,
        "n_shooting": args.n_shooting,
        "blocks": args.blocks,
        "repeat": args.repeat,
        "solved": not args.build_only,
        "max_iterations": args.max_iterations,
        "hessian_approximation": args.hessian_approximation,
    }


def _write_results(path: Path, metadata: dict, results: list[dict], summaries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"metadata": metadata, "results": results, "summaries": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-shooting", type=int, default=50)
    parser.add_argument("--blocks", type=int, nargs="+", default=[1, 5, 10, 25, 50])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument(
        "--hessian-approximation",
        choices=("limited-memory", "exact"),
        default="limited-memory",
        help="IPOPT Hessian mode; limited-memory keeps the muscle-driven benchmark tractable.",
    )
    parser.add_argument("--build-only", action="store_true", help="Measure construction and sparsity without solving.")
    parser.add_argument("--output", type=Path, help="Optional JSON file receiving raw results and medians.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.n_shooting < 2:
        raise ValueError("--n-shooting must be at least 2")
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")

    if args.worker:
        n_blocks = None if args.case == "dms" else int(args.case)
        result = run_case(
            n_shooting=args.n_shooting,
            n_blocks=n_blocks,
            solve=not args.build_only,
            max_iterations=args.max_iterations,
            hessian_approximation=args.hessian_approximation,
        )
        print(RESULT_PREFIX + json.dumps(result))
        return

    if any(n_blocks < 1 or n_blocks > args.n_shooting for n_blocks in args.blocks):
        raise ValueError("Every --blocks value must be between 1 and --n-shooting")

    cases = ["dms", *(str(n_blocks) for n_blocks in dict.fromkeys(args.blocks))]
    results = [_run_isolated(args, case) for case in cases for _ in range(args.repeat)]
    summaries = _summaries(results)
    metadata = _metadata(args)

    _print_table(summaries)
    if args.output:
        _write_results(args.output, metadata, results, summaries)
        print(f"\nRaw results and medians written to {args.output}")


if __name__ == "__main__":
    main()
