"""Archive the parameter COLLOCATION/SHARED regression without changing its oracle.

Run both checkouts in fresh Python processes in the same active environment:
    python tools/diagnose_parameter_collocation.py --baseline /path/to/upstream --output /tmp/parameter-diagnostic

Inspect previously saved JSON/NPZ pairs without solving:
    python tools/diagnose_parameter_collocation.py --compare-saved head.json baseline.json --output /tmp/comparison
"""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch

CI_VERSIONS = {
    "python": "3.14.8",
    "biorbd": "1.12.3",
    "casadi": "3.7.2",
    "numpy": "2.5.4",
    "libopenblas": "0.3.34",
    "ipopt": "3.14.19",
    "mumps-seq": "5.8.2",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def versions():
    import biorbd_casadi
    import casadi
    import numpy as np

    packages = subprocess.run(
        [os.environ.get("CONDA_EXE", "conda"), "list", "--prefix", sys.prefix, "--json"], capture_output=True, text=True
    )
    if packages.returncode:
        raise RuntimeError(packages.stderr)
    conda_packages = json.loads(packages.stdout)
    observed = {item["name"]: item["version"] for item in conda_packages}
    observed.update(
        python=platform.python_version(),
        biorbd=biorbd_casadi.__version__,
        casadi=casadi.__version__,
        numpy=np.__version__,
    )
    return {
        "observed": observed,
        "conda_packages": conda_packages,
        "numpy_blas_build": getattr(np.__config__, "CONFIG", {}).get("Build Dependencies", {}).get("blas"),
        "platform": sys.platform,
        "python_executable": sys.executable,
        "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }


def capture(ocp, solution, destination, metadata):
    # Adapted from round6's validated issue863/#1096 reproducer. Dispatch only
    # after the original solve, so instrumentation does not change its graph.
    import casadi
    import numpy as np
    from bioptim import SolutionMerge
    from bioptim.interfaces.ipopt_interface import IpoptInterface

    interface = IpoptInterface(ocp)
    g, _ = interface.dispatch_bounds()
    f = interface.dispatch_obj_func()
    x = ocp.variables_vector
    evaluate = casadi.Function("diagnostic", [x], [f, g, casadi.jacobian(g, x)])
    f0, g0, j0 = evaluate(ocp.init_vector)
    fs, gs, js = evaluate(solution.vector)
    states = solution.decision_states(to_merge=SolutionMerge.NODES)
    controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
    parameters = solution.parameters
    arrays = {
        "init": np.asarray(ocp.init_vector),
        "bounds_min": np.asarray(ocp.bounds_vectors[0]),
        "bounds_max": np.asarray(ocp.bounds_vectors[1]),
        "f0": np.asarray(f0),
        "g0": np.asarray(g0),
        "j0": np.asarray(j0),
        "vector": np.asarray(solution.vector),
        "f_solution": np.asarray(fs),
        "g_solution": np.asarray(gs),
        "jacobian_solution": np.asarray(js),
    }
    for prefix, values in (("state", states), ("control", controls), ("parameter", parameters)):
        arrays.update({f"{prefix}__{key}": np.asarray(value) for key, value in values.items()})
    np.savez(destination.with_suffix(".npz"), **arrays)
    metadata.update(
        status=solution.status,
        iterations=solution.iterations,
        cost=float(solution.cost),
        max_constraint=float(np.abs(np.asarray(solution.constraints)).max()),
        solver_stats=ocp.ocp_solver.shaked_ocp_solver.stats(),
        states={key: np.asarray(value).tolist() for key, value in states.items()},
        controls={key: np.asarray(value).tolist() for key, value in controls.items()},
        parameters={key: np.asarray(value).tolist() for key, value in parameters.items()},
        arrays={key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
        jacobian_nonzeros=j0.nnz(),
    )
    write_json(destination, metadata)


def run_case(args):
    checkout = args.case.resolve()
    sys.path.insert(0, str(checkout))
    os.chdir(checkout)
    destination = args.output / f"{args.label}.json"
    metadata = {"checkout": str(checkout), "test_result": "not_run"}
    try:
        metadata["sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        metadata["versions"] = versions()
        if args.require_ci_versions:
            observed = metadata["versions"]["observed"]
            mismatches = {
                key: {"expected": value, "actual": observed.get(key)}
                for key, value in CI_VERSIONS.items()
                if observed.get(key) != value
            }
            if mismatches:
                raise RuntimeError(f"CI runtime version mismatch: {mismatches}")

        from bioptim import OdeSolver, OptimalControlProgram, PhaseDynamics
        from tests.shard2 import test_global_getting_started as original

        if sys.platform == "darwin":
            # Exercise the Linux regression locally despite its existing Darwin
            # skip. Only the test module changes; dependency platform reporting stays native.
            original.platform = SimpleNamespace(system=lambda: "Linux")
            metadata["test_module_platform_proxy"] = "Linux"

        original_solve = OptimalControlProgram.solve
        captured = False

        def solve_and_capture(ocp, *positional, **keyword):
            nonlocal captured
            solution = original_solve(ocp, *positional, **keyword)
            if not captured:
                captured = True
                capture(ocp, solution, destination, metadata)
            return solution

        with patch.object(OptimalControlProgram, "solve", solve_and_capture):
            # Run the existing oracle, including feasibility, golden values and
            # warm start; its references, tolerances and solver options stay intact.
            original.test_parameter_optimization(OdeSolver.COLLOCATION, PhaseDynamics.SHARED_DURING_THE_PHASE)
        if not captured:
            raise RuntimeError("The original test did not solve the case")
        metadata["test_result"] = "passed"
        return 0
    except BaseException as error:
        metadata.update(test_result="failed", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        return 1
    finally:
        write_json(destination, metadata)


def compare_saved(paths, destination):
    import numpy as np

    left, right = [json.loads(path.read_text()) for path in paths]
    report = {"inputs": [str(path) for path in paths], "costs": [left.get("cost"), right.get("cost")], "arrays": {}}
    with np.load(paths[0].with_suffix(".npz")) as a, np.load(paths[1].with_suffix(".npz")) as b:
        report["keys_only_in_first"] = sorted(set(a.files) - set(b.files))
        report["keys_only_in_second"] = sorted(set(b.files) - set(a.files))
        for key in sorted(set(a.files) & set(b.files)):
            same_shape = a[key].shape == b[key].shape
            report["arrays"][key] = {
                "shapes": [list(a[key].shape), list(b[key].shape)],
                "exactly_equal": same_shape and bool(np.array_equal(a[key], b[key], equal_nan=True)),
                "max_absolute_difference": (
                    float(np.max(np.abs(a[key] - b[key]))) if same_shape and a[key].size else None
                ),
            }
    write_json(destination / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkout", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-ci-versions", action="store_true")
    parser.add_argument("--compare-saved", type=Path, nargs=2)
    parser.add_argument("--case", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.case:
        return run_case(args)
    if args.compare_saved:
        compare_saved([path.resolve() for path in args.compare_saved], args.output)
        return 0
    if args.baseline is None:
        parser.error("--baseline is required when running the diagnostic")

    outcomes = {}
    for label, checkout in (("checkout", args.checkout), ("baseline", args.baseline)):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--case",
            str(checkout.resolve()),
            "--label",
            label,
            "--output",
            str(args.output),
        ]
        if args.require_ci_versions:
            command.append("--require-ci-versions")
        with (args.output / f"{label}.log").open("w") as log:
            outcomes[label] = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{label}: exit={outcomes[label]}, evidence={args.output / label}", flush=True)
    write_json(args.output / "outcomes.json", outcomes)
    if all((args.output / f"{label}.npz").exists() for label in outcomes):
        compare_saved([args.output / f"{label}.json" for label in outcomes], args.output)
    return int(any(outcomes.values()))


if __name__ == "__main__":
    raise SystemExit(main())
