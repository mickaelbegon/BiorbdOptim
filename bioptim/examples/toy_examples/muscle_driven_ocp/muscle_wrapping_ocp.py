"""Muscle-driven OCPs: native biorbd wrapping versus frozen CasADi MLP geometry.

Requires https://github.com/mickaelbegon/biorbd/tree/codex/muscle-wrapping
(tested at 36851cca). For its experimental 1.9.11 backend, explicitly enable:

    BIOPTIM_ENABLE_EXPERIMENTAL_WRAPPING=1 python -m \
        bioptim.examples.toy_examples.muscle_driven_ocp.muscle_wrapping_ocp \
        --case all --output /tmp/muscle_wrapping_ocp

Select the freshly compiled biorbd_casadi package on PYTHONPATH, with the same
Python ABI as the build. No package installation or version spoofing is done
by this example. The compatibility flag is unnecessary on supported versions.
The tested legacy build cannot read arm26's VTP meshes. The wrapping fixtures
are mesh-free; this opt-in is not blanket support for existing bioptim examples.

Two *toy*, uncalibrated models are shipped: a 1-DOF elbow with two antagonist
Hill muscles/cylinder paths and a 2-DOF shoulder with four spatial ellipsoid
paths. Gravity is zero; there are no residual torques. Activations alone drive
the rigid-body dynamics. Both backends use identical bounds, initial guesses,
costs, terminal conditions, time grid and IPOPT limited-memory Hessians.
Timings are single local runs, not CPU-isolated performance guarantees; fit
time is reported separately from OCP solution time.

The MLP approximates LMT(q), supervising values AND gradients. Velocity J*qdot
and moment arms -J are obtained by differentiation, never fitted separately.
The surrogate uses biorbd's Type=hill force law with the same characteristics,
not an activation-to-torque fit. Native torque is checked against that law
using native geometry before fitting. These are rigid-tendon muscles, not
compliant-tendon equilibrium models. Native higher derivatives are numerical,
and both wrapping branch changes and the Hill velocity switch are nonsmooth.

Independent held-out geometry samples, native torque checks and native
forward replays of each optimized activation trajectory are reported. Native
callbacks are not serialized; only frozen MLP functions/weights are exported.
The prescribed contact branches stay active over these small angular domains.
No claim is made about contact switching, global branch optimality or anatomy.
"""

import argparse
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import biorbd_casadi as biorbd
import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp

from bioptim import (
    BiorbdModel,
    BoundsList,
    CasadiMLP,
    DynamicsOptions,
    InitialGuessList,
    InterpolationType,
    MuscleGeometry,
    MusclesBiorbdModel,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    SolutionMerge,
    Solver,
)
from bioptim.models.utils import cache_function


@dataclass(frozen=True)
class WrappingCase:
    name: str
    lower: tuple
    upper: tuple
    start: tuple
    end: tuple
    duration: float = 0.6

    @property
    def model_path(self):
        return Path(__file__).resolve().parents[2] / "models" / f"wrapping_{self.name}.bioMod"


CASES = {
    "elbow": WrappingCase("elbow", (-0.20,), (0.20,), (-0.08,), (0.08,)),
    "shoulder": WrappingCase("shoulder", (-0.04, -0.04), (0.04, 0.04), (-0.02, -0.015), (0.02, 0.015)),
}


def hill_force(activation, length, velocity, characteristics):
    """Exactly the biorbd HillType.cpp law for the Type=hill toy muscles.

    LMT and tendon slack length: m; velocity is positive lengthening in m/s.
    Preserve biorbd's existing velocity convention/constants (including its
    velocity scaling), rather than silently substituting another Hill model.
    """
    optimal = characteristics.optimalLength().to_mx()
    pennation = characteristics.pennationAngle().to_mx()
    slack = characteristics.tendonSlackLength().to_mx()
    fiber = (length - slack) / ca.cos(pennation)
    active_length = ca.exp(-((fiber / optimal / (1 + 0.15 * (1 - activation)) - 1) ** 2) / 0.45)
    passive = ca.if_else(fiber > 0, ca.exp(10 * (fiber / optimal - 1) - 5), 0)
    force_velocity = ca.if_else(
        velocity <= 0,
        (1 - ca.fabs(velocity) / 10) / (1 + ca.fabs(velocity) / 10),
        (1 - 1.33 * velocity / 10 / (-0.0825)) / (1 - velocity / 10 / (-0.0825)),
    )
    damping = ca.if_else(velocity > 0, 0.1 * velocity / (optimal * 10), 0)
    if not characteristics.useDamping():
        damping = 0
    return (
        characteristics.forceIsoMax().to_mx()
        * (activation * active_length * force_velocity + passive + damping)
        * ca.cos(pennation)
    )


class GeometryMusclesBiorbdModel(MusclesBiorbdModel):
    """Keep rigid-body dynamics/Hill characteristics; replace only LMT(q).

    Intended for these Type=hill examples. Not a generic replacement for the
    other biorbd muscle types, tendon compliance, fatigue or variable parameters.
    """

    def __init__(self, model_path, geometries):
        super().__init__(str(model_path), with_residual_torque=False)
        self.geometries = tuple(geometries)
        if any(self.model.muscle(i).type() != biorbd.HILL for i in range(self.nb_muscles)):
            raise NotImplementedError("This example implements only biorbd Type=hill muscles")
        if len(self.geometries) != self.nb_muscles or any(
            geometry.muscle_name != name or geometry.nb_q != self.nb_q
            for geometry, name in zip(self.geometries, self.muscle_names)
        ):
            raise ValueError("Provide one geometry per model muscle, in the model's order")
        if self.parameters.numel() or self.nb_quaternions or self.nb_q != self.nb_qdot:
            raise NotImplementedError("The toy surrogate model requires fixed parameters and qdot = dq/dt")

    def serialize(self):
        return GeometryMusclesBiorbdModel, dict(model_path=self.path, geometries=self.geometries)

    @cache_function
    def muscle_joint_torque(self):
        torque = ca.MX.zeros(self.nb_tau, 1)
        forces = []
        for index, geometry in enumerate(self.geometries):
            force = hill_force(
                self.muscle[index],
                geometry.length(self.q),
                geometry.velocity(self.q, self.qdot),
                self.model.muscle(index).characteristics(),
            )
            forces.append(force)
            torque += geometry.joint_torque(self.q, force)
        self.muscle_forces = ca.Function(
            "geometry_muscle_forces", [self.muscle, self.q, self.qdot], [ca.vertcat(*forces)]
        )
        return ca.Function(
            "geometry_muscle_joint_torque",
            [self.muscle, self.q, self.qdot, self.parameters],
            [torque],
            ["activation", "q", "qdot", "parameters"],
            ["tau"],
        )


def reference_geometries(case):
    if not all(hasattr(biorbd, shape) for shape in ("WrappingCylinder", "WrappingEllipsoid")):
        raise RuntimeError("This example requires the biorbd codex/muscle-wrapping branch")
    model = BiorbdModel(str(case.model_path))
    return tuple(MuscleGeometry.from_biorbd(model, name, expand=False) for name in model.muscle_names)


def prepare_ocp(case, geometries=None, n_shooting=20):
    """Same pure muscle-activation OCP for native or approximated geometry."""
    if n_shooting < 4:
        raise ValueError("Use at least four shooting intervals")
    model = (
        MusclesBiorbdModel(str(case.model_path), with_residual_torque=False)
        if geometries is None
        else GeometryMusclesBiorbdModel(case.model_path, geometries)
    )
    bounds = BoundsList()
    bounds["q"] = [list(case.lower), list(case.upper)]
    bounds["q"][:, 0] = case.start
    bounds["q"][:, -1] = case.end
    bounds["qdot"] = [[-2.0] * model.nb_q, [2.0] * model.nb_q]
    bounds["qdot"][:, [0, -1]] = 0
    control_bounds = BoundsList()
    control_bounds["muscles"] = [[0.0] * model.nb_muscles, [1.0] * model.nb_muscles]
    s = np.linspace(0, 1, n_shooting + 1)[None, :]
    start, delta = np.array(case.start)[:, None], (np.array(case.end) - case.start)[:, None]
    initial = InitialGuessList()
    initial.add("q", start + delta * (3 * s**2 - 2 * s**3), interpolation=InterpolationType.EACH_FRAME)
    initial.add("qdot", delta * 6 * s * (1 - s) / case.duration, interpolation=InterpolationType.EACH_FRAME)
    controls = InitialGuessList()
    controls["muscles"] = [0.05] * model.nb_muscles
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1)
    return OptimalControlProgram(
        model,
        n_shooting,
        case.duration,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4(n_integration_steps=1), expand_dynamics=False),
        x_bounds=bounds,
        u_bounds=control_bounds,
        x_init=initial,
        u_init=controls,
        objective_functions=objectives,
        use_sx=False,
        n_threads=1,
    )


def geometry_values(geometries, q, qdot):
    samples = q.shape[1]
    return {
        "length_m": np.vstack([np.asarray(g.length(q)).reshape(1, samples) for g in geometries]),
        "velocity_m_per_s": np.vstack([np.asarray(g.velocity(q, qdot)).reshape(1, samples) for g in geometries]),
        "moment_arms_m_per_rad": np.stack(
            [np.asarray(g.moment_arms(q)).reshape(samples, q.shape[0]) for g in geometries], axis=0
        ),
    }


def errors(prediction, reference):
    return {
        key: {
            "max_abs": float(np.max(np.abs(prediction[key] - value))),
            "rmse": float(np.sqrt(np.mean((prediction[key] - value) ** 2))),
        }
        for key, value in reference.items()
    }


def validate_reference(case, geometries):
    """Check AD, envelope signs, torque/force-law agreement and active contact."""
    rng = np.random.default_rng(781)
    q = rng.uniform(case.lower, case.upper, (12, len(case.start))).T
    corners = np.array(list(product(*zip(case.lower, case.upper)))).T
    q = np.hstack((q, corners))
    qdot = rng.uniform(-0.3, 0.3, q.shape)
    activation = rng.uniform(0, 1, (len(geometries), q.shape[1]))
    native = MusclesBiorbdModel(str(case.model_path), with_residual_torque=False)
    reconstructed = GeometryMusclesBiorbdModel(case.model_path, geometries)
    difference = np.asarray(native.muscle_joint_torque()(activation, q, qdot, [])) - np.asarray(
        reconstructed.muscle_joint_torque()(activation, q, qdot, [])
    )
    torque_error = float(np.max(np.abs(difference)))
    if torque_error > 1e-8:
        raise RuntimeError(f"Hill law/native torque disagreement: {torque_error} N.m")
    max_fd = 0.0
    minimum_wrap_excess = float("inf")
    raw = biorbd.Model(str(case.model_path))
    symbol = ca.MX.sym("q", len(case.start))
    raw.updateMuscles(symbol, True)
    envelope = ca.Function("native_envelope", [symbol], [raw.musclesLengthJacobian(symbol).to_mx()])
    ad = ca.vertcat(*[g.length_jacobian(symbol) for g in geometries])
    envelope_error = float(np.max(np.abs(np.asarray(ca.Function("ad", [symbol], [ad])(q)) - np.asarray(envelope(q)))))
    for index, geometry in enumerate(geometries):
        points = raw.muscle(index).position().pointsInGlobal()
        chord = ca.norm_2(points[-1].to_mx() - points[0].to_mx())
        excess = ca.Function("wrap_excess", [symbol], [geometry.length(symbol) - chord])
        minimum_wrap_excess = min(minimum_wrap_excess, float(np.min(np.asarray(excess(q)))))
        for axis in range(len(case.start)):
            step = np.eye(len(case.start))[axis, :, None] * 2e-6
            finite = (np.asarray(geometry.length(q + step)) - np.asarray(geometry.length(q - step))) / 4e-6
            exact = np.asarray(geometry.length_jacobian(q)).reshape(q.shape[1], len(case.start))[:, axis]
            max_fd = max(max_fd, float(np.max(np.abs(finite.ravel() - exact))))
        torque = np.asarray(geometry.joint_torque(q[:, 0], 100)).ravel()
        if abs(torque @ qdot[:, 0] + 100 * float(geometry.velocity(q[:, 0], qdot[:, 0]))) > 1e-10:
            raise RuntimeError("Virtual power convention failed")
    if max_fd > 2e-6 or envelope_error > 1e-9 or minimum_wrap_excess <= 1e-4:
        raise RuntimeError("Reference derivative or active wrapping validation failed")
    return {
        "hill_native_torque_max_abs_error_Nm": torque_error,
        "ad_envelope_max_abs_error_m_per_rad": envelope_error,
        "length_gradient_finite_difference_max_abs_error_m_per_rad": max_fd,
        "minimum_wrapped_minus_straight_length_m": minimum_wrap_excess,
    }


def fit_geometries(case, references, n_samples=256, hidden_size=32, max_iterations=1500):
    start = perf_counter()
    fitted = tuple(
        CasadiMLP.fit(
            reference,
            range(reference.nb_q),
            case.lower,
            case.upper,
            np.zeros(reference.nb_q),
            n_samples=n_samples,
            hidden_size=hidden_size,
            max_iterations=max_iterations,
            seed=120 + index,
        )
        for index, reference in enumerate(references)
    )
    return fitted, perf_counter() - start


def dynamics_function(model):
    torque = model.muscle_joint_torque()(model.muscle, model.q, model.qdot, [])
    acceleration = model.forward_dynamics()(model.q, model.qdot, torque, [], [])
    return ca.Function("rhs", [ca.vertcat(model.q, model.qdot), model.muscle], [ca.vertcat(model.qdot, acceleration)])


def replay_native(case, q, qdot, activation):
    """Independent DOP853 replay, plus local defects starting from each node.

    Controls are piecewise constant, just as in the OCP. Restart integration at
    every control discontinuity. No optimized states are injected into the
    cumulative replay after its initial state.
    """
    model = MusclesBiorbdModel(str(case.model_path), with_residual_torque=False)
    rhs = dynamics_function(model)
    intervals = activation.shape[1]
    dt = case.duration / intervals
    planned = np.vstack((q, qdot))
    cumulative = np.empty_like(planned)
    cumulative[:, 0] = planned[:, 0]
    local = np.empty((planned.shape[0], intervals))

    def integrate(state, controls):
        solution = solve_ivp(
            lambda time, x: np.asarray(rhs(x, controls)).ravel(),
            [0, dt],
            state,
            method="DOP853",
            rtol=1e-9,
            atol=1e-11,
            max_step=dt / 2,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        angles = solution.y[: len(case.start)]
        if np.any(angles < np.array(case.lower)[:, None] - 1e-9) or np.any(
            angles > np.array(case.upper)[:, None] + 1e-9
        ):
            raise RuntimeError("Native replay escaped the validated fitting domain")
        return solution.y[:, -1]

    for node in range(intervals):
        cumulative[:, node + 1] = integrate(cumulative[:, node], activation[:, node])
        local[:, node] = integrate(planned[:, node], activation[:, node]) - planned[:, node + 1]
    nq = q.shape[0]
    return {
        "max_abs_local_q_defect_rad": float(np.max(np.abs(local[:nq]))),
        "max_abs_local_qdot_defect_rad_per_s": float(np.max(np.abs(local[nq:]))),
        "max_abs_q_drift_rad": float(np.max(np.abs(cumulative[:nq] - q))),
        "final_q_error_rad": (cumulative[:nq, -1] - np.array(case.end)).tolist(),
        "final_qdot_rad_per_s": cumulative[nq:, -1].tolist(),
    }, cumulative


def solve_case(case, geometries, n_shooting=20, hessian="limited-memory"):
    start = perf_counter()
    ocp = prepare_ocp(case, geometries, n_shooting)
    build_seconds = perf_counter() - start
    solver = Solver.IPOPT()
    solver.set_print_level(0)
    solver.set_maximum_iterations(400)
    solver.set_convergence_tolerance(1e-8)
    solver.set_hessian_approximation(hessian)
    solver.set_option_unsafe(0.0, "bound_relax_factor")
    start = perf_counter()
    solution = ocp.solve(solver)
    solve_seconds = perf_counter() - start
    states = solution.decision_states(to_merge=SolutionMerge.NODES)
    activation = solution.decision_controls(to_merge=SolutionMerge.NODES)["muscles"][:, :n_shooting]
    report = {
        "status": int(solution.status),
        "iterations": int(solution.iterations),
        "cost": float(solution.cost),
        "max_abs_nlp_constraints": float(np.max(np.abs(solution.constraints))),
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "hessian": hessian,
        "final_q_rad": states["q"][:, -1].tolist(),
        "activation_min": float(np.min(activation)),
        "activation_max": float(np.max(activation)),
    }
    if report["status"] != 0 or report["max_abs_nlp_constraints"] > 1e-6:
        raise RuntimeError(f"{case.name} OCP did not converge: {report}")
    replay, trajectory = replay_native(case, states["q"], states["qdot"], activation)
    report["native_replay"] = replay
    if replay["max_abs_q_drift_rad"] > 5e-4 or max(abs(v) for v in replay["final_qdot_rad_per_s"]) > 2e-3:
        raise RuntimeError(f"{case.name}: surrogate/discretization failed native replay validation: {replay}")
    return report, {**states, "activation": activation, "native_replay_state": trajectory}


def _timing(function, q, qdot, activation):
    for _ in range(2):
        function(activation, q, qdot, [])
    durations = []
    for _ in range(5):
        start = perf_counter()
        function(activation, q, qdot, [])
        durations.append(perf_counter() - start)
    return float(np.median(durations) * 1e6 / q.shape[1])


def _plot(case, references, fitted, trajectories, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = {"native": "black", "mlp": "tab:orange"}
    for name, trajectory in trajectories.items():
        time = np.linspace(0, case.duration, trajectory["q"].shape[1])
        for axis in range(len(case.start)):
            axes[0, 0].plot(
                time, trajectory["q"][axis], color=colors[name], ls=["-", "--"][axis], label=f"{name} q{axis}"
            )
            axes[0, 0].plot(time, trajectory["native_replay_state"][axis], color=colors[name], ls=":", alpha=0.6)
        for index in range(len(references)):
            axes[0, 1].step(time[:-1], trajectory["activation"][index], where="post", color=colors[name], alpha=0.7)
    q = trajectories["native"]["q"]
    qdot = trajectories["native"]["qdot"]
    true, predicted = geometry_values(references, q, qdot), geometry_values(fitted, q, qdot)
    for name, values in (("native", true), ("mlp", predicted)):
        time = np.linspace(0, case.duration, q.shape[1])
        for index in range(len(references)):
            axes[1, 0].plot(time, values["length_m"][index] * 1000, color=colors[name], alpha=0.7)
            axes[1, 1].plot(time, values["velocity_m_per_s"][index] * 1000, color=colors[name], alpha=0.7)
    axes[0, 0].set_title("Joint angles (dotted: native replay)")
    axes[0, 0].set_ylabel("rad")
    axes[0, 0].legend()
    axes[0, 1].set_title("Muscle activations (no residual torques)")
    peak_activation = max(np.max(trajectory["activation"]) for trajectory in trajectories.values())
    axes[0, 1].set_ylim(0, max(0.001, 1.15 * peak_activation))
    for name, color in colors.items():
        axes[0, 1].plot([], [], color=color, label=name)
    axes[0, 1].legend()
    axes[1, 0].set_title("LMT, same native trajectory")
    axes[1, 0].set_ylabel("mm")
    axes[1, 1].set_title("Lengthening velocity, same native trajectory")
    axes[1, 1].set_ylabel("mm/s")
    for axis in axes.flat:
        axis.set_xlabel("time (s)")
        axis.grid(alpha=0.2)
    fig.suptitle(f"Toy {case.name}: native wrapping versus CasADi MLP")
    fig.savefig(output / f"{case.name}.png", dpi=160)
    plt.close(fig)


def run_comparison(case, output, n_shooting=20, n_samples=256, hidden_size=32, max_iterations=1500):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    references = reference_geometries(case)
    reference_validation = validate_reference(case, references)
    fitted, fit_seconds = fit_geometries(case, references, n_samples, hidden_size, max_iterations)
    rng = np.random.default_rng(8301)
    q = rng.uniform(case.lower, case.upper, (256, len(case.start))).T
    qdot = rng.uniform(-0.5, 0.5, q.shape)
    actual = geometry_values(references, q, qdot)
    predicted = geometry_values(fitted, q, qdot)
    geometry_errors = errors(predicted, actual)
    if (
        geometry_errors["length_m"]["max_abs"] > 5e-5
        or geometry_errors["moment_arms_m_per_rad"]["max_abs"] > 5e-4
        or geometry_errors["velocity_m_per_s"]["max_abs"] > 5e-4
    ):
        raise RuntimeError(f"MLP held-out validation failed: {geometry_errors}")
    native = MusclesBiorbdModel(str(case.model_path), with_residual_torque=False)
    mlp_model = GeometryMusclesBiorbdModel(case.model_path, fitted)
    activation = rng.uniform(0.01, 0.8, (len(fitted), q.shape[1]))
    torque_functions = {"native": native.muscle_joint_torque(), "mlp": mlp_model.muscle_joint_torque()}
    torque_error = np.asarray(torque_functions["mlp"](activation, q, qdot, [])) - np.asarray(
        torque_functions["native"](activation, q, qdot, [])
    )
    report = {
        "model": str(case.model_path),
        "biorbd_version": biorbd.__version__,
        "casadi_version": ca.__version__,
        "native_branch_tested_commit": "36851cca",
        "domain_rad": {"lower": case.lower, "upper": case.upper},
        "start_rad": case.start,
        "end_rad": case.end,
        "duration_s": case.duration,
        "n_shooting": n_shooting,
        "discretization": "direct multiple shooting, one RK4 step per interval",
        "native_replay_integrator": "DOP853, rtol=1e-9, atol=1e-11, max_step=dt/2",
        "reference_checks": reference_validation,
        "fit_seconds": fit_seconds,
        "training_samples_per_muscle": n_samples,
        "validation_samples": q.shape[1],
        "mlp_fits": {g.muscle_name: asdict(g.fit_result) for g in fitted},
        "held_out_geometry_errors": geometry_errors,
        "held_out_torque_max_abs_error_Nm": float(np.max(np.abs(torque_error))),
        "torque_us_per_sample": {
            name: _timing(function, q[:, :32], qdot[:, :32], activation[:, :32])
            for name, function in torque_functions.items()
        },
        "ocp": {},
    }
    trajectories = {}
    for name, geometries in (("native", None), ("mlp", fitted)):
        print(f"{case.name}: solving {name} OCP", flush=True)
        report["ocp"][name], trajectories[name] = solve_case(case, geometries, n_shooting)
        np.savez(output / f"{case.name}_{name}_solution.npz", **trajectories[name])
    report["comparison"] = {
        "max_abs_q_difference_rad": float(np.max(np.abs(trajectories["native"]["q"] - trajectories["mlp"]["q"]))),
        "max_abs_activation_difference": float(
            np.max(np.abs(trajectories["native"]["activation"] - trajectories["mlp"]["activation"]))
        ),
        "relative_cost_difference": abs(report["ocp"]["mlp"]["cost"] - report["ocp"]["native"]["cost"])
        / max(abs(report["ocp"]["native"]["cost"]), 1e-12),
    }
    for geometry in fitted:
        geometry.length.save(str(output / f"{case.name}_{geometry.muscle_name}_mlp.casadi"))
        np.savez(
            output / f"{case.name}_{geometry.muscle_name}_weights.npz",
            w1=geometry.weights[0],
            w2=geometry.weights[1],
            b1=geometry.biases[0],
            b2=geometry.biases[1],
            lower=geometry.lower,
            upper=geometry.upper,
            output_mean=geometry.output_mean,
            output_scale=geometry.output_scale,
        )
    np.savez(
        output / f"{case.name}_validation.npz",
        q=q,
        qdot=qdot,
        **{f"native_{k}": v for k, v in actual.items()},
        **{f"mlp_{k}": v for k, v in predicted.items()},
    )
    _plot(case, references, fitted, trajectories, output)
    (output / f"{case.name}_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--output", type=Path, default=Path("/tmp/muscle_wrapping_ocp"))
    parser.add_argument("--n-shooting", type=int, default=20)
    parser.add_argument("--training-samples", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--mlp-iterations", type=int, default=1500)
    args = parser.parse_args()
    for name in CASES if args.case == "all" else [args.case]:
        report = run_comparison(
            CASES[name], args.output, args.n_shooting, args.training_samples, args.hidden_size, args.mlp_iterations
        )
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
