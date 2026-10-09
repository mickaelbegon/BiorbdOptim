"""Fit and validate BRA (1 DOF) and Wu DELT1 (6 DOFs) geometry surrogates.

Run from the repository root:
    python -m bioptim.examples.toy_examples.muscle_driven_ocp.muscle_geometry_surrogates \
        --shoulder-model /path/to/Wu_Shoulder_Model_via_points.bioMod \
        --output /tmp/muscle_geometry --solve-ocp

The shoulder model is not shipped with bioptim. The default looks in the sibling
biobuddy repository, without copying or modifying that model. Training and test
sets are independent. Output includes frozen CasADi length functions, MLP weights,
validation data, a JSON report, and figures. This example approximates geometric
muscle-tendon length, NOT fibre length or a muscle force law.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import casadi as ca
import numpy as np

from bioptim import BiorbdModel, MuscleGeometry, MuscleGeometrySpline, CasadiMLP


def prepare_geometry_ocp(model_path, geometry, target_length, n_shooting=12):
    """Physical arm OCP with surrogate geometry in constraints and an objective.

    Fix shoulder endpoints; drive the arm with joint torque. The endpoint is
    defined by muscle-tendon length, not an imposed elbow angle. This demonstrates
    differentiation through length and velocity in a real bioptim NLP; it does
    not replace biorbd's muscle force model.
    """
    from bioptim import (
        BoundsList,
        ConstraintList,
        DynamicsOptions,
        TorqueBiorbdModel,
        ObjectiveList,
        ObjectiveFcn,
        OptimalControlProgram,
        Node,
    )

    def final_length(controller):
        return geometry.length(controller.q) - target_length

    def muscle_velocity(controller):
        return geometry.velocity(controller.q, controller.qdot)

    model = TorqueBiorbdModel(str(model_path))
    x_bounds = BoundsList()
    x_bounds["q"] = [[-0.3, 0.3], [0.3, 2.2]]
    x_bounds["q"][0, [0, -1]] = 0
    x_bounds["q"][1, 0] = 0.7
    x_bounds["qdot"] = [[-8, -8], [8, 8]]
    x_bounds["qdot"][:, [0, -1]] = 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [[-100, -100], [100, 100]]
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.01)
    objectives.add(muscle_velocity, custom_type=ObjectiveFcn.Lagrange, quadratic=True, weight=10)
    constraints = ConstraintList()
    constraints.add(final_length, node=Node.END)
    return OptimalControlProgram(
        model,
        n_shooting,
        0.5,
        dynamics=DynamicsOptions(),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objectives,
        constraints=constraints,
        use_sx=False,
    )


def _evaluate(geometry, q, qdot):
    return {
        "length": np.asarray(geometry.length(q)).ravel(),
        "velocity": np.asarray(geometry.velocity(q, qdot)).ravel(),
        "moment_arms": np.asarray(geometry.moment_arms(q)).reshape(q.shape[1], q.shape[0]).T,
    }


def _errors(prediction, reference):
    return {
        key: {
            "rmse": float(np.sqrt(np.mean((prediction[key] - value) ** 2))),
            "max_abs": float(np.max(np.abs(prediction[key] - value))),
        }
        for key, value in reference.items()
    }


def _timing(geometry, q, qdot):
    # Same serial CasADi call pattern for all methods; no GPU/thread comparison.
    samples = q.shape[1]
    for _ in range(2):
        _evaluate(geometry, q, qdot)
    durations = []
    for _ in range(5):
        start = perf_counter()
        _evaluate(geometry, q, qdot)
        durations.append(perf_counter() - start)
    return float(np.median(durations) * 1e6 / samples)


def _plot(output, label, time, trajectory, active_indices, dof_names):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    colors = {"reference": "black", "spline": "tab:blue", "mlp": "tab:orange"}
    styles = {"reference": "-", "spline": "--", "mlp": ":"}
    for name, values in trajectory.items():
        axes[0].plot(time, values["length"] * 1000, styles[name], color=colors[name], label=name)
        axes[1].plot(time, values["velocity"] * 1000, styles[name], color=colors[name], label=name)
    arm_colors = plt.cm.tab10(np.linspace(0, 1, len(active_indices)))
    for color, index in zip(arm_colors, active_indices):
        for name, values in trajectory.items():
            axes[2].plot(
                time,
                values["moment_arms"][index] * 1000,
                styles[name],
                color=color,
                label=dof_names[index] if name == "reference" else None,
            )
    axes[0].set(ylabel="Muscle-tendon length (mm)", title=label)
    axes[1].set(ylabel="Lengthening velocity (mm/s)")
    axes[2].set(ylabel="Moment arm -dl/dq (mm/rad)", xlabel="Time (s)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output / f"{label}.png", dpi=160)
    plt.close(fig)


def build_case(
    label, model_path, muscle_name, indices, center, half_width, grid_size, output, seed, solve_ocp, mlp_iterations=6000
):
    model = BiorbdModel(str(model_path))
    reference = MuscleGeometry.from_biorbd(model, muscle_name)
    lower = center[list(indices)] - half_width
    upper = center[list(indices)] + half_width
    rng = np.random.default_rng(seed + 10000)
    q_test = np.repeat(center[:, None], 1000, axis=1)
    q_test[list(indices)] = rng.uniform(lower[:, None], upper[:, None], (len(indices), 1000))
    qdot_test = np.zeros_like(q_test)
    qdot_test[list(indices)] = rng.uniform(-2, 2, (len(indices), 1000))
    true = _evaluate(reference, q_test, qdot_test)
    biorbd_jacobian = np.asarray(model.muscle_length_jacobian()(q_test, []))
    biorbd_jacobian = biorbd_jacobian[list(model.muscle_names).index(muscle_name)].reshape(1000, model.nb_q).T
    jacobian_agreement = float(np.max(np.abs(-true["moment_arms"] - biorbd_jacobian)))
    # The Wu file contains rounded rotation matrices. Its analytical geometric
    # Jacobian differs slightly from AD of the actual length expression. Record
    # that discrepancy and independently check AD against finite differences.
    if jacobian_agreement > 1e-6:
        raise RuntimeError(f"Length AD and biorbd's geometric Jacobian disagree: {jacobian_agreement}")
    finite_difference_error = 0.0
    for axis in range(model.nb_q):
        step = np.eye(model.nb_q)[axis, :, None] * 1e-5
        finite = (
            np.asarray(reference.length(q_test[:, :10] + step)) - np.asarray(reference.length(q_test[:, :10] - step))
        ).ravel() / 2e-5
        finite_difference_error = max(
            finite_difference_error, float(np.max(np.abs(finite + true["moment_arms"][axis, :10])))
        )
    if finite_difference_error > 1e-9:
        raise RuntimeError(f"AD and finite differences disagree: {finite_difference_error}")
    active_amplitudes = np.max(np.abs(true["moment_arms"]), axis=1)
    omitted = [i for i in range(model.nb_q) if i not in indices]
    omitted_max = float(np.max(active_amplitudes[omitted])) if omitted else 0.0
    if omitted_max > 1e-6:
        raise RuntimeError(f"Omitted DOFs have significant moment arms ({omitted_max} m/unit)")
    start = perf_counter()
    spline = MuscleGeometrySpline.fit(
        reference, indices, [np.linspace(a, b, grid_size) for a, b in zip(lower, upper)], center
    )
    spline_fit_seconds = perf_counter() - start
    start = perf_counter()
    mlp = CasadiMLP.fit(
        reference,
        indices,
        lower,
        upper,
        center,
        n_samples=256 if len(indices) == 1 else 2048,
        hidden_size=24 if len(indices) == 1 else 96,
        max_iterations=mlp_iterations,
        seed=seed,
    )
    mlp_fit_seconds = perf_counter() - start
    time = np.linspace(0, 2, 301)
    frequency = np.linspace(0.7, 1.2, len(indices))[:, None]
    phase = np.linspace(0, 0.8, len(indices))[:, None]
    q = np.repeat(center[:, None], len(time), axis=1)
    qdot = np.zeros_like(q)
    q[list(indices)] += 0.85 * half_width[:, None] * np.sin(frequency * time + phase)
    qdot[list(indices)] = 0.85 * half_width[:, None] * frequency * np.cos(frequency * time + phase)
    methods = {"reference": reference, "spline": spline, "mlp": mlp}
    report = {
        "source_model": str(Path(model_path).resolve()),
        "muscle": muscle_name,
        "nb_q": model.nb_q,
        "active_dof_indices": indices,
        "active_dof_names": [model.name_dof[i] for i in indices],
        "q_reference": center.tolist(),
        "lower_rad": lower.tolist(),
        "upper_rad": upper.tolist(),
        "max_abs_biorbd_jacobian_disagreement": jacobian_agreement,
        "max_abs_reference_finite_difference_error": finite_difference_error,
        "max_abs_omitted_moment_arm": omitted_max,
        "units": {"length": "m", "velocity": "m/s", "moment_arms": "m/rad"},
        "test_samples": 1000,
        "test_velocity_bounds_rad_per_s": [-2, 2],
        "spline_grid_points": spline.grid_point_count,
        "spline_degree": spline.degree,
        "fit_seconds": {"spline": spline_fit_seconds, "mlp": mlp_fit_seconds},
        "mlp_fit": asdict(mlp.fit_result),
        "methods": {},
    }
    trajectory = {}
    for name, method in methods.items():
        prediction = _evaluate(method, q_test, qdot_test)
        report["methods"][name] = {
            "errors": _errors(prediction, true),
            "geometry_us_per_sample": _timing(method, q_test, qdot_test),
        }
        report["methods"][name]["active_moment_arms_errors"] = _errors(
            {"moment_arms": prediction["moment_arms"][indices]},
            {"moment_arms": true["moment_arms"][indices]},
        )["moment_arms"]
        trajectory[name] = _evaluate(method, q, qdot)
        method.length.save(str(output / f"{label}_{name}.casadi"))
        np.savez(output / f"{label}_{name}_validation.npz", q=q_test, qdot=qdot_test, **prediction)
        # Geometry in a force-driven dynamics requires third derivatives of l.
        q_symbol = ca.MX.sym("q", model.nb_q)
        hessian = ca.hessian(ca.sum1(method.joint_torque(q_symbol, 100)), q_symbol)[0]
        values = np.asarray(ca.Function("torque_hessian", [q_symbol], [hessian])(center))
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite torque Hessian for {name}")
        report["methods"][name]["torque_hessian_finite"] = True
        if solve_ocp and label == "BRA":
            from bioptim import Solver, SolutionMerge

            ocp = prepare_geometry_ocp(model_path, method, float(reference.length([0, 1.2])))
            solver = Solver.IPOPT()
            solver.set_print_level(0)
            solution = ocp.solve(solver)
            states = solution.decision_states(to_merge=SolutionMerge.NODES)
            report["methods"][name]["ocp"] = {
                "status": solution.status,
                "iterations": solution.iterations,
                "max_abs_constraint": float(np.max(np.abs(solution.constraints))),
                "final_q": states["q"][:, -1].tolist(),
                "true_final_length_error_m": float(reference.length(states["q"][:, -1]))
                - float(reference.length([0, 1.2])),
            }
            if solution.status != 0 or report["methods"][name]["ocp"]["max_abs_constraint"] > 1e-6:
                raise RuntimeError(f"OCP did not converge for {name}")
    np.savez(
        output / f"{label}_mlp_weights.npz",
        w1=mlp.weights[0],
        w2=mlp.weights[1],
        b1=mlp.biases[0],
        b2=mlp.biases[1],
        lower=lower,
        upper=upper,
        dof_indices=indices,
        nb_q=model.nb_q,
        muscle_name=muscle_name,
        output_mean=mlp.output_mean,
        output_scale=mlp.output_scale,
    )
    _plot(output, label, time, trajectory, indices, model.name_dof)
    print(label, json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shoulder-model",
        type=Path,
        default=Path(__file__).resolve().parents[5] / "biobuddy/examples/models/Wu_Shoulder_Model_via_points.bioMod",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/muscle_geometry"))
    parser.add_argument("--solve-ocp", action="store_true")
    parser.add_argument("--mlp-iterations", type=int, default=6000)
    args = parser.parse_args()
    if not args.shoulder_model.is_file():
        parser.error("Supply --shoulder-model pointing to the Wu via-points model (DELT1, 16 coordinates)")
    args.output.mkdir(parents=True, exist_ok=True)
    arm_path = Path(__file__).resolve().parents[2] / "models/arm26.bioMod"
    shoulder = BiorbdModel(str(args.shoulder_model))
    if shoulder.nb_q != 16 or "DELT1" not in shoulder.muscle_names:
        parser.error("Expected the 16-coordinate Wu via-points model containing DELT1")
    center = np.zeros(16)
    center[8:14] = [-0.05, 0.15, 0, -0.3, 1.0, 0.7]
    center[14] = 1.0
    report = {
        "casadi_version": ca.__version__,
        "numpy_version": np.__version__,
        "seed": 2026,
        "BRA": build_case(
            "BRA",
            arm_path,
            "BRA",
            [1],
            np.array([0, 1.25]),
            np.array([0.95]),
            25,
            args.output,
            2026,
            args.solve_ocp,
            args.mlp_iterations,
        ),
        "DELT1": build_case(
            "DELT1",
            args.shoulder_model,
            "DELT1",
            list(range(8, 14)),
            center,
            np.array([0.25, 0.25, 0.25, 0.35, 0.45, 0.35]),
            6,
            args.output,
            2027,
            False,
            args.mlp_iterations,
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
