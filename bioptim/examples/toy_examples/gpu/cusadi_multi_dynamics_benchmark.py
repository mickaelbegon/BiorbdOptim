"""Benchmark representative Bioptim OCPs and their numerical oracles on CPU and GPU.

The nonlinear programs are solved with IPOPT on the CPU while varying the
number of Bioptim/CasADi and native-library threads.  The dynamics extracted
from those same OCPs are then evaluated with mapped CasADi functions on the CPU
and with CusADi on an NVIDIA GPU.

CusADi is a batched numerical evaluator, not an NLP solver backend.  In
addition to local dynamics, this benchmark can nevertheless compile the full
discretized NLP and its matrix-free derivatives (gradient, JVP, VJP, and HVP).
Those oracles are the part of a hybrid SQP method that can run on the GPU; the
globalization and sparse KKT solve would remain on the CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import casadi
import numpy as np

import bioptim
from bioptim import ObjectiveFcn, ObjectiveList, OdeSolver, OrderingStrategy, Solver
from bioptim.examples.biomechanics.gait_optimal_estimation.gait_example import (
    prepare_ocp as prepare_wholebody_gait,
)
from bioptim.examples.getting_started.basic_ocp import prepare_ocp as prepare_pendulum
from bioptim.examples.getting_started.example_inequality_constraint import (
    prepare_ocp as prepare_contact_inequality,
)
from bioptim.examples.toy_examples.acados.cube import prepare_ocp as prepare_cube
from bioptim.examples.toy_examples.acados.static_arm import (
    prepare_ocp as prepare_static_arm,
)
from bioptim.examples.toy_examples.feature_examples.example_joints_acceleration_driven import (
    prepare_ocp as prepare_joint_acceleration,
)
from bioptim.examples.toy_examples.muscle_driven_ocp.static_arm_with_contact import (
    prepare_ocp as prepare_muscle_contact,
)
from bioptim.examples.toy_examples.gpu.cusadi_pendulum_benchmark import (
    _compile_cuda_source,
    _error_metrics,
    _generate_cuda_source,
    _git_revision,
    _import_torch,
    _load_cusadi_function,
    _time_cusadi_gpu,
)
from bioptim.examples.utils import ExampleUtils


CASE_NAMES = (
    "pendulum",
    "cube",
    "joint_acceleration",
    "contact_inequality",
    "static_arm",
    "muscle_contact",
    "wholebody_gait",
)
DEFAULT_N_SHOOTING = {
    "pendulum": 50,
    "cube": 30,
    "joint_acceleration": 30,
    "contact_inequality": 10,
    "static_arm": 10,
    "muscle_contact": 10,
    "wholebody_gait": 3,
}
CASE_DESCRIPTIONS = {
    "pendulum": "two-degree-of-freedom torque-driven pendulum",
    "cube": "three-degree-of-freedom rigid body",
    "joint_acceleration": "double pendulum driven by joint acceleration",
    "contact_inequality": "four-degree-of-freedom rigid-contact model with friction inequalities",
    "static_arm": "two-degree-of-freedom arm driven by residual torques and six muscles",
    "muscle_contact": "three-degree-of-freedom arm with rigid contact, residual torques, and six muscles",
    "wholebody_gait": "34-degree-of-freedom gait model with 20 muscles and measured external forces",
}
DYNAMICS_ONLY_CASES = {
    "wholebody_gait": (
        "The reference 105-interval OCP requires about 20 GB of RAM, while even a two-interval IPOPT solve takes "
        "more than 90 seconds. The benchmark therefore measures its representative 47k-instruction dynamics only."
    )
}
FULL_DERIVATIVE_EXCLUSIONS = {
    "wholebody_gait": (
        "The full Jacobian and weighted Hessian contain 2,835,505 and 3,474,646 CasADi instructions. Their generated "
        "CUDA sources are 224 MB and 275 MB; NVCC did not finish the Jacobian compilation after 10 minutes and used "
        "about 15.5 GB of RAM. Directional or block derivatives are required for this case."
    )
}
WORKER_RESULT_PREFIX = "BIOPTIM_CUSADI_CPU_RESULT="
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass
class OcpSolveResult:
    case: str
    n_shooting: int
    cpu_cores: int
    repetition: int
    outcome: str
    build_s: float | None = None
    solve_wall_s: float | None = None
    solver_s: float | None = None
    iterations: int | None = None
    status: int | None = None
    cost: float | None = None
    max_constraint_violation: float | None = None
    error: str | None = None


@dataclass
class DynamicsBenchmarkResult:
    case: str
    n_shooting: int
    batch_size: int
    cpu_cores: int
    gpu_block_size: int
    gpu_grid_size: int
    cuda_threads: int
    max_absolute_error: float
    max_relative_error: float
    casadi_cpu_ms: float
    cusadi_kernel_ms: float
    cusadi_gpu_resident_ms: float
    cusadi_with_transfers_ms: float
    resident_speedup: float
    end_to_end_speedup: float


@dataclass
class DerivativeBenchmarkResult:
    case: str
    derivative_kind: str
    variable_size: int
    output_rows: int
    output_columns: int
    output_nonzeros: int
    instructions: int
    work_size: int
    batch_size: int
    cpu_cores: int
    gpu_block_size: int
    gpu_grid_size: int
    cuda_threads: int
    max_absolute_error: float
    max_relative_error: float
    casadi_cpu_ms: float
    cusadi_kernel_ms: float
    cusadi_gpu_resident_ms: float
    cusadi_with_transfers_ms: float
    resident_speedup: float
    end_to_end_speedup: float


@dataclass
class NlpOracleBenchmarkResult:
    case: str
    oracle_kind: str
    n_shooting: int
    decision_variables: int
    constraints: int
    output_nonzeros: int
    instructions: int
    work_size: int
    batch_size: int
    cpu_cores: int
    gpu_block_size: int
    gpu_grid_size: int
    cuda_threads: int
    max_absolute_error: float
    max_relative_error: float
    casadi_cpu_ms: float
    cusadi_kernel_ms: float
    cusadi_gpu_resident_ms: float
    cusadi_with_transfers_ms: float
    resident_speedup: float
    end_to_end_speedup: float


NLP_ORACLE_KINDS = ("primal", "gradient", "jvp", "vjp", "hvp")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _parse_int_list(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return list(dict.fromkeys(parsed))


def _parse_gpu_block_sizes(value: str) -> list[int]:
    parsed = _parse_int_list(value)
    if any(item < 32 or item > 1024 or item % 32 for item in parsed):
        raise argparse.ArgumentTypeError(
            "GPU block sizes must be multiples of 32 between 32 and 1024"
        )
    return parsed


def _case_n_shooting(case: str, override: int | None) -> int:
    return override if override is not None else DEFAULT_N_SHOOTING[case]


def _models_directory() -> Path:
    return Path(ExampleUtils.folder) / "models"


def _prepare_case(
    case: str,
    n_shooting: int,
    cpu_cores: int,
    ordering_strategy: OrderingStrategy = OrderingStrategy.VARIABLE_MAJOR,
):
    models = _models_directory()
    if case == "pendulum":
        return prepare_pendulum(
            biorbd_model_path=str(models / "pendulum.bioMod"),
            final_time=1.0,
            n_shooting=n_shooting,
            use_sx=True,
            n_threads=cpu_cores,
            expand_dynamics=True,
            ordering_strategy=ordering_strategy,
        )
    if case == "cube":
        ocp = prepare_cube(
            biorbd_model_path=str(models / "cube.bioMod"),
            n_shooting=n_shooting,
            tf=2.0,
            use_sx=True,
            expand_dynamics=True,
            n_threads=cpu_cores,
            ordering_strategy=ordering_strategy,
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
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="tau",
            weight=1,
            multi_thread=False,
        )
        ocp.update_objectives(objectives)
        return ocp
    if case == "joint_acceleration":
        return prepare_joint_acceleration(
            biorbd_model_path=str(models / "double_pendulum.bioMod"),
            final_time=10.0,
            n_shooting=n_shooting,
            use_sx=True,
            n_threads=cpu_cores,
        )
    if case == "contact_inequality":
        return prepare_contact_inequality(
            biorbd_model_path=str(models / "2segments_4dof_2contacts.bioMod"),
            phase_time=0.3,
            n_shooting=n_shooting,
            min_bound=50,
            max_bound=np.inf,
            mu=0.2,
            ode_solver=OdeSolver.RK4(),
            expand_dynamics=True,
            n_threads=cpu_cores,
            use_sx=True,
            ordering_strategy=ordering_strategy,
        )
    if case == "static_arm":
        return prepare_static_arm(
            biorbd_model_path=str(models / "arm26.bioMod"),
            final_time=1.0,
            n_shooting=n_shooting,
            use_sx=True,
            n_threads=cpu_cores,
            expand_dynamics=True,
        )
    if case == "muscle_contact":
        return prepare_muscle_contact(
            biorbd_model_path=str(models / "arm26_with_contact.bioMod"),
            final_time=1.0,
            n_shooting=n_shooting,
            weight=1000,
            ode_solver=OdeSolver.RK4(),
            expand_dynamics=True,
            n_threads=cpu_cores,
            use_sx=True,
            ordering_strategy=ordering_strategy,
        )
    if case == "wholebody_gait":
        data_path = (
            Path(ExampleUtils.folder)
            / "biomechanics"
            / "gait_optimal_estimation"
            / "ocp_data.pkl"
        )
        with data_path.open("rb") as file:
            data = pickle.load(file)
        frame_indices = (
            np.linspace(0, data["n_shooting"], n_shooting + 1).round().astype(int)
        )
        external_forces = {
            name: values[:, frame_indices] for name, values in data["f_ext_exp"].items()
        }
        return prepare_wholebody_gait(
            biorbd_model_path=str(models / "wholebody_model.bioMod"),
            n_shooting=n_shooting,
            phase_time=float(data["phase_time"]),
            q_exp=data["q_exp"][:, frame_indices],
            qdot_exp=data["qdot_exp"][:, frame_indices],
            tau_exp=data["tau_exp"][:, frame_indices],
            f_ext_exp=external_forces,
            emg_normalized_exp=data["emg_normalized_exp"][:, frame_indices],
            markers_exp=data["markers_exp"][:, :, frame_indices],
            ode_solver=OdeSolver.RK2(n_integration_steps=1),
            n_threads=cpu_cores,
        )
    raise ValueError(f"Unknown benchmark case: {case}")


def _named_dynamics(
    case: str, dynamics: casadi.Function, gpu_block_size: int
) -> casadi.Function:
    inputs = [
        casadi.SX.sym(dynamics.name_in(index), dynamics.sparsity_in(index))
        for index in range(dynamics.n_in())
    ]
    outputs = dynamics.call(inputs)
    return casadi.Function(
        f"{case}_block_{gpu_block_size}_ForwardDyn",
        inputs,
        outputs,
        [dynamics.name_in(index) for index in range(dynamics.n_in())],
        [dynamics.name_out(index) for index in range(dynamics.n_out())],
    )


def _named_dynamics_derivative(
    case: str,
    dynamics: casadi.Function,
    derivative_kind: str,
    gpu_block_size: int,
) -> casadi.Function:
    if derivative_kind not in ("jacobian", "hessian"):
        raise ValueError("Derivative kind must be 'jacobian' or 'hessian'")

    inputs = [
        casadi.SX.sym(dynamics.name_in(index), dynamics.sparsity_in(index))
        for index in range(dynamics.n_in())
    ]
    dynamics_output = dynamics.call(inputs)[0]
    variables = casadi.vertcat(inputs[1], inputs[2])
    function_inputs = inputs
    input_names = [dynamics.name_in(index) for index in range(dynamics.n_in())]

    if derivative_kind == "jacobian":
        derivative = casadi.jacobian(dynamics_output, variables)
        function_name = f"{case}_block_{gpu_block_size}_JacobianDyn"
        output_name = "jacobian_x_u"
    else:
        adjoint = casadi.SX.sym("adjoint", dynamics_output.sparsity())
        derivative = casadi.hessian(casadi.dot(adjoint, dynamics_output), variables)[0]
        function_inputs = [*inputs, adjoint]
        input_names = [*input_names, "adjoint"]
        function_name = f"{case}_block_{gpu_block_size}_HessianDyn"
        output_name = "weighted_hessian_x_u"

    return casadi.Function(
        function_name,
        function_inputs,
        [derivative],
        input_names,
        [output_name],
    )


def _nlp_expressions(
    ocp: Any,
) -> tuple[casadi.SX | casadi.MX, casadi.SX | casadi.MX, casadi.SX | casadi.MX, Any]:
    """Return the decision vector, scalar objective, constraints, and constraint bounds."""

    solver = Solver.IPOPT(show_online_optim=False)
    ocp.set_ocp_solver(solver)
    interface = ocp.ocp_solver
    interface.opts = solver
    objectives = interface.dispatch_obj_func()
    constraints, constraint_bounds = interface.dispatch_bounds()
    return ocp.variables_vector, casadi.sum1(objectives), constraints, constraint_bounds


def _named_nlp_oracles(
    case: str,
    ocp: Any,
    gpu_block_size: int,
) -> tuple[dict[str, casadi.Function], Any]:
    """Build matrix-free numerical oracles for the complete discretized OCP."""

    decision_vector, objective, constraints, constraint_bounds = _nlp_expressions(ocp)
    n_variables = int(decision_vector.shape[0])
    n_constraints = int(constraints.shape[0])
    transcription = casadi.Function(
        f"{case}_NlpTranscription",
        [decision_vector],
        [objective, constraints],
        ["decision_vector"],
        ["objective", "constraints"],
    )

    z = casadi.SX.sym("decision_vector", n_variables)
    objective_at_z, constraints_at_z = transcription(z)
    direction = casadi.SX.sym("direction", n_variables)
    constraint_adjoint = casadi.SX.sym("constraint_adjoint", n_constraints)
    objective_factor = casadi.SX.sym("objective_factor")

    objective_gradient = casadi.gradient(objective_at_z, z)
    constraint_jvp = casadi.jtimes(constraints_at_z, z, direction)
    constraint_vjp = casadi.jtimes(constraints_at_z, z, constraint_adjoint, True)
    lagrangian_gradient = casadi.gradient(
        objective_factor * objective_at_z
        + casadi.dot(constraint_adjoint, constraints_at_z),
        z,
    )
    lagrangian_hvp = casadi.jtimes(lagrangian_gradient, z, direction)
    prefix = f"{case}_block_{gpu_block_size}_Nlp"
    return {
        "primal": casadi.Function(
            f"{prefix}Primal",
            [z],
            [objective_at_z, constraints_at_z],
            ["decision_vector"],
            ["objective", "constraints"],
        ),
        "gradient": casadi.Function(
            f"{prefix}Gradient",
            [z],
            [objective_gradient],
            ["decision_vector"],
            ["objective_gradient"],
        ),
        "jvp": casadi.Function(
            f"{prefix}Jvp",
            [z, direction],
            [constraint_jvp],
            ["decision_vector", "direction"],
            ["constraint_jvp"],
        ),
        "vjp": casadi.Function(
            f"{prefix}Vjp",
            [z, constraint_adjoint],
            [constraint_vjp],
            ["decision_vector", "constraint_adjoint"],
            ["constraint_vjp"],
        ),
        "hvp": casadi.Function(
            f"{prefix}Hvp",
            [z, constraint_adjoint, objective_factor, direction],
            [lagrangian_hvp],
            ["decision_vector", "constraint_adjoint", "objective_factor", "direction"],
            ["lagrangian_hvp"],
        ),
    }, constraint_bounds


def _optional_scalar(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        return None
    scalar = float(array.squeeze())
    return scalar if math.isfinite(scalar) else None


def _max_constraint_violation(ocp: Any, solution: Any) -> float | None:
    if solution.constraints is None:
        return None
    constraints = np.asarray(solution.constraints, dtype=float)
    lower_bounds = np.asarray(ocp.ocp_solver.limits["lbg"], dtype=float)
    upper_bounds = np.asarray(ocp.ocp_solver.limits["ubg"], dtype=float)
    lower_violation = np.maximum(lower_bounds - constraints, 0.0)
    upper_violation = np.maximum(constraints - upper_bounds, 0.0)
    violation = (
        float(np.max(np.maximum(lower_violation, upper_violation)))
        if constraints.size
        else 0.0
    )
    return violation if math.isfinite(violation) else None


def _cpu_worker(args: argparse.Namespace) -> OcpSolveResult:
    case = args._worker_case
    n_shooting = args._worker_n_shooting
    cpu_cores = args._worker_cpu_cores
    repetition = args._worker_repetition
    try:
        start = time.perf_counter()
        ocp = _prepare_case(case, n_shooting, cpu_cores)
        build_s = time.perf_counter() - start

        solver = Solver.IPOPT(show_online_optim=False)
        solver.set_print_level(0)
        solver.set_convergence_tolerance(args.tolerance)
        solver.set_constraint_tolerance(args.tolerance)
        solver.set_maximum_iterations(args.max_iterations)

        start = time.perf_counter()
        solution = ocp.solve(solver)
        solve_wall_s = time.perf_counter() - start
        max_violation = _max_constraint_violation(ocp, solution)
        status = int(solution.status)
        outcome = "success"
        if status != 0:
            outcome = "solver_failure"
        elif max_violation is not None and max_violation > 10 * args.tolerance:
            outcome = "infeasible"

        return OcpSolveResult(
            case=case,
            n_shooting=n_shooting,
            cpu_cores=cpu_cores,
            repetition=repetition,
            outcome=outcome,
            build_s=build_s,
            solve_wall_s=solve_wall_s,
            solver_s=_optional_scalar(solution.solver_time_to_optimize),
            iterations=int(solution.iterations),
            status=status,
            cost=_optional_scalar(solution.cost),
            max_constraint_violation=max_violation,
        )
    except Exception as error:
        return OcpSolveResult(
            case=case,
            n_shooting=n_shooting,
            cpu_cores=cpu_cores,
            repetition=repetition,
            outcome="exception",
            error=f"{type(error).__name__}: {error}",
        )


def _run_cpu_solve(
    args: argparse.Namespace,
    case: str,
    n_shooting: int,
    cpu_cores: int,
    repetition: int,
) -> OcpSolveResult:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_cpu-worker",
        "--_worker-case",
        case,
        "--_worker-n-shooting",
        str(n_shooting),
        "--_worker-cpu-cores",
        str(cpu_cores),
        "--_worker-repetition",
        str(repetition),
        "--tolerance",
        str(args.tolerance),
        "--max-iterations",
        str(args.max_iterations),
    ]
    environment = os.environ.copy()
    for name in THREAD_ENVIRONMENT_VARIABLES:
        environment[name] = str(cpu_cores)
    environment.setdefault("MPLCONFIGDIR", "/tmp/bioptim-cusadi-matplotlib")
    process = subprocess.run(
        command,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in reversed(process.stdout.splitlines()):
        if line.startswith(WORKER_RESULT_PREFIX):
            return OcpSolveResult(**json.loads(line.removeprefix(WORKER_RESULT_PREFIX)))
    error = process.stdout.strip()
    if len(error) > 2000:
        error = error[-2000:]
    return OcpSolveResult(
        case=case,
        n_shooting=n_shooting,
        cpu_cores=cpu_cores,
        repetition=repetition,
        outcome="worker_failure",
        error=f"worker exit code {process.returncode}: {error}",
    )


def _make_case_inputs(
    case: str, dynamics: casadi.Function, batch_size: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs = [
        np.zeros((batch_size, dynamics.nnz_in(index)), dtype=np.float64)
        for index in range(dynamics.n_in())
    ]
    if dynamics.n_in() > 0 and dynamics.nnz_in(0) == 2:
        inputs[0][:, 0] = rng.uniform(0.0, 1.0, size=batch_size)
        inputs[0][:, 1] = 0.01

    states = inputs[1]
    controls = inputs[2]
    if case == "pendulum":
        states[:, :2] = rng.uniform(-np.pi, np.pi, size=(batch_size, 2))
        states[:, 2:] = rng.uniform(-5.0, 5.0, size=(batch_size, 2))
        controls[:] = rng.uniform(-100.0, 100.0, size=controls.shape)
    elif case == "cube":
        states[:, :3] = rng.uniform(-1.0, 1.0, size=(batch_size, 3))
        states[:, 3:] = rng.uniform(-2.0, 2.0, size=(batch_size, 3))
        controls[:] = rng.uniform(-100.0, 100.0, size=controls.shape)
    elif case == "joint_acceleration":
        states[:, :2] = rng.uniform(-np.pi, np.pi, size=(batch_size, 2))
        states[:, 2:] = rng.uniform(-5.0, 5.0, size=(batch_size, 2))
        controls[:] = rng.uniform(-50.0, 50.0, size=controls.shape)
    elif case == "contact_inequality":
        reference_pose = np.array([0.0, 0.0, -0.75, 0.75])
        states[:, :4] = reference_pose + rng.uniform(-0.2, 0.2, size=(batch_size, 4))
        states[:, 4:] = rng.uniform(-1.0, 1.0, size=(batch_size, 4))
        controls[:] = rng.uniform(-100.0, 100.0, size=controls.shape)
    elif case == "static_arm":
        states[:, 0] = rng.uniform(0.05, 1.2, size=batch_size)
        states[:, 1] = rng.uniform(0.3, 2.2, size=batch_size)
        states[:, 2:] = rng.uniform(-1.0, 1.0, size=(batch_size, 2))
        controls[:, :2] = rng.uniform(-10.0, 10.0, size=(batch_size, 2))
        controls[:, 2:] = rng.uniform(0.05, 1.0, size=(batch_size, 6))
    elif case == "muscle_contact":
        states[:, :3] = rng.uniform(-1.0, 1.0, size=(batch_size, 3))
        states[:, 3:] = rng.uniform(-2.0, 2.0, size=(batch_size, 3))
        controls[:, :3] = rng.uniform(-1.0, 1.0, size=(batch_size, 3))
        controls[:, 3:] = rng.uniform(0.05, 1.0, size=(batch_size, 6))
    elif case == "wholebody_gait":
        states[:, :34] = rng.uniform(-0.2, 0.2, size=(batch_size, 34))
        states[:, 34:] = rng.uniform(-1.0, 1.0, size=(batch_size, 34))
        controls[:, :34] = rng.uniform(-100.0, 100.0, size=(batch_size, 34))
        controls[:, 34:54] = rng.uniform(0.05, 1.0, size=(batch_size, 20))
        controls[:, 54:60] = rng.uniform(-100.0, 100.0, size=(batch_size, 6))
        controls[:, 60:66] = rng.uniform(-0.2, 0.2, size=(batch_size, 6))
    else:
        raise ValueError(f"Unknown benchmark case: {case}")
    return [np.ascontiguousarray(values) for values in inputs]


def _mapped_output(dynamics: casadi.Function, inputs: list[np.ndarray]) -> np.ndarray:
    mapped = dynamics.map(inputs[0].shape[0], "serial")
    outputs = mapped.call([casadi.DM(values.T) for values in inputs])
    nonzeros = np.asarray(outputs[0].nonzeros(), dtype=np.float64)
    return nonzeros.reshape(inputs[0].shape[0], dynamics.nnz_out(0))


def _mapped_outputs(
    function: casadi.Function, inputs: list[np.ndarray]
) -> list[np.ndarray]:
    mapped = function.map(inputs[0].shape[0], "serial")
    outputs = mapped.call([casadi.DM(values.T) for values in inputs])
    return [
        np.asarray(output.nonzeros(), dtype=np.float64).reshape(
            inputs[0].shape[0], function.nnz_out(index)
        )
        for index, output in enumerate(outputs)
    ]


def _time_casadi_cpu(
    dynamics: casadi.Function,
    inputs: list[np.ndarray],
    cpu_cores: int,
    warmup: int,
    repeats: int,
) -> float:
    mode = "serial" if cpu_cores == 1 else "thread"
    mapped = dynamics.map(inputs[0].shape[0], mode, cpu_cores)
    mapped_inputs = [casadi.DM(values.T) for values in inputs]
    for _ in range(warmup):
        mapped.call(mapped_inputs)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        mapped.call(mapped_inputs)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return median(samples)


def _benchmark_gpu_case(
    args: argparse.Namespace,
    case: str,
    n_shooting: int,
    cusadi_root: Path,
    torch: Any,
) -> tuple[dict[str, Any], list[DynamicsBenchmarkResult]]:
    ocp = _prepare_case(case, n_shooting, cpu_cores=1)
    dynamics = ocp.nlp[0].dynamics_func
    if not isinstance(dynamics, casadi.Function):
        raise TypeError(f"Expected a CasADi Function, got {type(dynamics).__name__}")

    inputs_by_batch = {
        batch_size: _make_case_inputs(
            case, dynamics, batch_size, args.seed + batch_index
        )
        for batch_index, batch_size in enumerate(args.gpu_batch_sizes)
    }
    references = {
        batch_size: _mapped_output(dynamics, inputs)
        for batch_size, inputs in inputs_by_batch.items()
    }
    cpu_timings = {
        (batch_size, cpu_cores): _time_casadi_cpu(
            dynamics,
            inputs,
            cpu_cores,
            args.eval_warmup,
            args.eval_repeats,
        )
        for batch_size, inputs in inputs_by_batch.items()
        for cpu_cores in args.cpu_cores
    }

    results: list[DynamicsBenchmarkResult] = []
    used_operations: list[str] = []
    for gpu_block_size in args.gpu_block_sizes:
        named_dynamics = _named_dynamics(case, dynamics, gpu_block_size)
        _, used_operations = _generate_cuda_source(
            named_dynamics, cusadi_root, gpu_block_size
        )
        capability = torch.cuda.get_device_capability()
        _compile_cuda_source(
            cusadi_root, named_dynamics.name(), [capability[0] * 10 + capability[1]]
        )

        for batch_size, cpu_inputs in inputs_by_batch.items():
            cusadi_function = _load_cusadi_function(
                cusadi_root, named_dynamics, batch_size
            )
            gpu_inputs = [
                torch.from_numpy(values)
                .to(device="cuda", dtype=torch.float64)
                .contiguous()
                for values in cpu_inputs
            ]
            cusadi_function.evaluate(gpu_inputs)
            torch.cuda.synchronize()
            candidate = cusadi_function.outputs_sparse[0].detach().cpu().numpy()
            reference = references[batch_size]
            np.testing.assert_allclose(
                candidate, reference, rtol=args.rtol, atol=args.atol
            )
            max_absolute_error, max_relative_error = _error_metrics(
                reference, candidate
            )
            kernel_ms, resident_ms, with_transfers_ms = _time_cusadi_gpu(
                torch,
                cusadi_function,
                cpu_inputs,
                args.eval_warmup,
                args.eval_repeats,
            )
            for cpu_cores in args.cpu_cores:
                cpu_ms = cpu_timings[(batch_size, cpu_cores)]
                results.append(
                    DynamicsBenchmarkResult(
                        case=case,
                        n_shooting=n_shooting,
                        batch_size=batch_size,
                        cpu_cores=cpu_cores,
                        gpu_block_size=gpu_block_size,
                        gpu_grid_size=math.ceil(batch_size / gpu_block_size),
                        cuda_threads=batch_size,
                        max_absolute_error=max_absolute_error,
                        max_relative_error=max_relative_error,
                        casadi_cpu_ms=cpu_ms,
                        cusadi_kernel_ms=kernel_ms,
                        cusadi_gpu_resident_ms=resident_ms,
                        cusadi_with_transfers_ms=with_transfers_ms,
                        resident_speedup=cpu_ms / resident_ms,
                        end_to_end_speedup=cpu_ms / with_transfers_ms,
                    )
                )

    case_metadata = {
        "case": case,
        "description": CASE_DESCRIPTIONS[case],
        "n_shooting": n_shooting,
        "function_inputs": [
            {"name": dynamics.name_in(index), "nonzeros": dynamics.nnz_in(index)}
            for index in range(dynamics.n_in())
        ],
        "function_outputs": [
            {"name": dynamics.name_out(index), "nonzeros": dynamics.nnz_out(index)}
            for index in range(dynamics.n_out())
        ],
        "instructions": dynamics.n_instructions(),
        "work_size": dynamics.sz_w(),
        "operations": used_operations,
        "complete_ocp_solve_benchmarked": case not in DYNAMICS_ONLY_CASES,
        "complete_ocp_solve_limitation": DYNAMICS_ONLY_CASES.get(case),
    }
    return case_metadata, results


def _make_derivative_inputs(
    case: str,
    dynamics: casadi.Function,
    derivative_kind: str,
    batch_size: int,
    seed: int,
) -> list[np.ndarray]:
    inputs = _make_case_inputs(case, dynamics, batch_size, seed)
    if derivative_kind == "hessian":
        rng = np.random.default_rng(seed + 10_000)
        inputs.append(
            np.ascontiguousarray(
                rng.uniform(-1.0, 1.0, size=(batch_size, dynamics.nnz_out(0))),
                dtype=np.float64,
            )
        )
    return inputs


def _benchmark_gpu_derivatives_case(
    args: argparse.Namespace,
    case: str,
    n_shooting: int,
    cusadi_root: Path,
    torch: Any,
) -> tuple[list[dict[str, Any]], list[DerivativeBenchmarkResult]]:
    ocp = _prepare_case(case, n_shooting, cpu_cores=1)
    dynamics = ocp.nlp[0].dynamics_func
    if not isinstance(dynamics, casadi.Function):
        raise TypeError(f"Expected a CasADi Function, got {type(dynamics).__name__}")

    metadata: list[dict[str, Any]] = []
    results: list[DerivativeBenchmarkResult] = []
    variable_size = dynamics.nnz_in(1) + dynamics.nnz_in(2)
    for derivative_index, derivative_kind in enumerate(("jacobian", "hessian")):
        reference_function = _named_dynamics_derivative(
            case,
            dynamics,
            derivative_kind,
            args.gpu_block_sizes[0],
        )
        inputs_by_batch = {
            batch_size: _make_derivative_inputs(
                case,
                dynamics,
                derivative_kind,
                batch_size,
                args.seed + 1_000 * (derivative_index + 1) + batch_index,
            )
            for batch_index, batch_size in enumerate(args.derivative_batch_sizes)
        }
        references = {
            batch_size: _mapped_output(reference_function, inputs)
            for batch_size, inputs in inputs_by_batch.items()
        }
        cpu_timings = {
            (batch_size, cpu_cores): _time_casadi_cpu(
                reference_function,
                inputs,
                cpu_cores,
                args.eval_warmup,
                args.eval_repeats,
            )
            for batch_size, inputs in inputs_by_batch.items()
            for cpu_cores in args.cpu_cores
        }

        derivative_metadata = {
            "case": case,
            "derivative_kind": derivative_kind,
            "definition": (
                "d(f)/d(x,u)"
                if derivative_kind == "jacobian"
                else "d2(adjoint^T f)/d(x,u)2"
            ),
            "variable_size": variable_size,
            "output_rows": reference_function.size1_out(0),
            "output_columns": reference_function.size2_out(0),
            "output_nonzeros": reference_function.nnz_out(0),
            "instructions": reference_function.n_instructions(),
            "work_size": reference_function.sz_w(),
            "operations": [],
        }

        for gpu_block_size in args.gpu_block_sizes:
            derivative_function = _named_dynamics_derivative(
                case,
                dynamics,
                derivative_kind,
                gpu_block_size,
            )
            _, used_operations = _generate_cuda_source(
                derivative_function, cusadi_root, gpu_block_size
            )
            derivative_metadata["operations"] = used_operations
            capability = torch.cuda.get_device_capability()
            _compile_cuda_source(
                cusadi_root,
                derivative_function.name(),
                [capability[0] * 10 + capability[1]],
            )

            for batch_size, cpu_inputs in inputs_by_batch.items():
                cusadi_function = _load_cusadi_function(
                    cusadi_root, derivative_function, batch_size
                )
                gpu_inputs = [
                    torch.from_numpy(values)
                    .to(device="cuda", dtype=torch.float64)
                    .contiguous()
                    for values in cpu_inputs
                ]
                cusadi_function.evaluate(gpu_inputs)
                torch.cuda.synchronize()
                candidate = cusadi_function.outputs_sparse[0].detach().cpu().numpy()
                reference = references[batch_size]
                np.testing.assert_allclose(
                    candidate, reference, rtol=args.rtol, atol=args.atol
                )
                max_absolute_error, max_relative_error = _error_metrics(
                    reference, candidate
                )
                kernel_ms, resident_ms, with_transfers_ms = _time_cusadi_gpu(
                    torch,
                    cusadi_function,
                    cpu_inputs,
                    args.eval_warmup,
                    args.eval_repeats,
                )
                for cpu_cores in args.cpu_cores:
                    cpu_ms = cpu_timings[(batch_size, cpu_cores)]
                    results.append(
                        DerivativeBenchmarkResult(
                            case=case,
                            derivative_kind=derivative_kind,
                            variable_size=variable_size,
                            output_rows=derivative_function.size1_out(0),
                            output_columns=derivative_function.size2_out(0),
                            output_nonzeros=derivative_function.nnz_out(0),
                            instructions=derivative_function.n_instructions(),
                            work_size=derivative_function.sz_w(),
                            batch_size=batch_size,
                            cpu_cores=cpu_cores,
                            gpu_block_size=gpu_block_size,
                            gpu_grid_size=math.ceil(batch_size / gpu_block_size),
                            cuda_threads=batch_size,
                            max_absolute_error=max_absolute_error,
                            max_relative_error=max_relative_error,
                            casadi_cpu_ms=cpu_ms,
                            cusadi_kernel_ms=kernel_ms,
                            cusadi_gpu_resident_ms=resident_ms,
                            cusadi_with_transfers_ms=with_transfers_ms,
                            resident_speedup=cpu_ms / resident_ms,
                            end_to_end_speedup=cpu_ms / with_transfers_ms,
                        )
                    )
                del cusadi_function, gpu_inputs, candidate
                torch.cuda.empty_cache()
        metadata.append(derivative_metadata)
    return metadata, results


def _make_nlp_oracle_inputs(
    ocp: Any,
    function: casadi.Function,
    batch_size: int,
    seed: int,
) -> list[np.ndarray]:
    """Create trial points along a bounded search line and matrix-free seed vectors."""

    rng = np.random.default_rng(seed)
    initial_guess = np.asarray(ocp.init_vector, dtype=np.float64).reshape(-1)
    lower_bounds = np.asarray(ocp.bounds_vectors[0], dtype=np.float64).reshape(-1)
    upper_bounds = np.asarray(ocp.bounds_vectors[1], dtype=np.float64).reshape(-1)
    finite_width = np.where(
        np.isfinite(lower_bounds) & np.isfinite(upper_bounds),
        upper_bounds - lower_bounds,
        2.0 * np.maximum(np.abs(initial_guess), 1.0),
    )
    finite_width = np.where(finite_width > 0.0, finite_width, 0.0)
    direction = rng.normal(size=initial_guess.size) * 0.05 * finite_width
    alphas = np.linspace(0.0, 1.0, batch_size)
    trial_points = initial_guess[None, :] + alphas[:, None] * direction[None, :]
    trial_points = np.maximum(trial_points, lower_bounds[None, :])
    trial_points = np.minimum(trial_points, upper_bounds[None, :])

    inputs: list[np.ndarray] = []
    for index in range(function.n_in()):
        name = function.name_in(index)
        if name == "decision_vector":
            values = trial_points
        elif name == "direction":
            values = np.repeat(direction[None, :], batch_size, axis=0)
        elif name == "constraint_adjoint":
            adjoint = rng.normal(size=function.nnz_in(index))
            values = np.repeat(adjoint[None, :], batch_size, axis=0)
        elif name == "objective_factor":
            values = np.ones((batch_size, 1), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported NLP oracle input: {name}")
        inputs.append(np.ascontiguousarray(values, dtype=np.float64))
    return inputs


def _benchmark_gpu_nlp_oracles_case(
    args: argparse.Namespace,
    case: str,
    n_shooting: int,
    cusadi_root: Path,
    torch: Any,
) -> tuple[list[dict[str, Any]], list[NlpOracleBenchmarkResult]]:
    """Benchmark full-transcription products used by a matrix-free SQP iteration."""

    ocp = _prepare_case(case, n_shooting, cpu_cores=1)
    reference_functions, _ = _named_nlp_oracles(case, ocp, args.gpu_block_sizes[0])
    function_inputs: dict[str, dict[int, list[np.ndarray]]] = {}
    references: dict[str, dict[int, list[np.ndarray]]] = {}
    cpu_timings: dict[tuple[str, int, int], float] = {}
    for oracle_index, oracle_kind in enumerate(args.nlp_oracles):
        reference_function = reference_functions[oracle_kind]
        inputs_by_batch = {
            batch_size: _make_nlp_oracle_inputs(
                ocp,
                reference_function,
                batch_size,
                args.seed + 10_000 * (oracle_index + 1) + batch_index,
            )
            for batch_index, batch_size in enumerate(args.nlp_batch_sizes)
        }
        function_inputs[oracle_kind] = inputs_by_batch
        references[oracle_kind] = {
            batch_size: _mapped_outputs(reference_function, inputs)
            for batch_size, inputs in inputs_by_batch.items()
        }
        for batch_size, inputs in inputs_by_batch.items():
            for cpu_cores in args.cpu_cores:
                cpu_timings[(oracle_kind, batch_size, cpu_cores)] = _time_casadi_cpu(
                    reference_function,
                    inputs,
                    cpu_cores,
                    args.eval_warmup,
                    args.eval_repeats,
                )

    metadata: list[dict[str, Any]] = []
    results: list[NlpOracleBenchmarkResult] = []
    capability = torch.cuda.get_device_capability()
    n_variables = int(ocp.variables_vector.shape[0])
    _, _, constraints, _ = _nlp_expressions(ocp)
    n_constraints = int(constraints.shape[0])
    for gpu_block_size in args.gpu_block_sizes:
        functions, _ = _named_nlp_oracles(case, ocp, gpu_block_size)
        for oracle_kind in args.nlp_oracles:
            function = functions[oracle_kind]
            _, used_operations = _generate_cuda_source(
                function, cusadi_root, gpu_block_size
            )
            _compile_cuda_source(
                cusadi_root, function.name(), [capability[0] * 10 + capability[1]]
            )
            if gpu_block_size == args.gpu_block_sizes[0]:
                metadata.append(
                    {
                        "case": case,
                        "oracle_kind": oracle_kind,
                        "definition": {
                            "primal": "objective and complete discretized constraints",
                            "gradient": "d(objective)/d(z)",
                            "jvp": "d(constraints)/d(z) times direction",
                            "vjp": "constraint_adjoint^T times d(constraints)/d(z)",
                            "hvp": (
                                "d2(objective_factor*objective + constraint_adjoint^T*constraints)/d(z)2 "
                                "times direction"
                            ),
                        }[oracle_kind],
                        "decision_variables": n_variables,
                        "constraints": n_constraints,
                        "output_nonzeros": sum(
                            function.nnz_out(index) for index in range(function.n_out())
                        ),
                        "instructions": function.n_instructions(),
                        "work_size": function.sz_w(),
                        "operations": used_operations,
                    }
                )

            for batch_size, cpu_inputs in function_inputs[oracle_kind].items():
                cusadi_function = _load_cusadi_function(
                    cusadi_root, function, batch_size
                )
                gpu_inputs = [
                    torch.from_numpy(values)
                    .to(device="cuda", dtype=torch.float64)
                    .contiguous()
                    for values in cpu_inputs
                ]
                cusadi_function.evaluate(gpu_inputs)
                torch.cuda.synchronize()
                candidate_outputs = [
                    output.detach().cpu().numpy()
                    for output in cusadi_function.outputs_sparse
                ]
                reference_outputs = references[oracle_kind][batch_size]
                for candidate, reference in zip(candidate_outputs, reference_outputs):
                    np.testing.assert_allclose(
                        candidate, reference, rtol=args.rtol, atol=args.atol
                    )
                errors = [
                    _error_metrics(reference, candidate)
                    for reference, candidate in zip(
                        reference_outputs, candidate_outputs
                    )
                ]
                max_absolute_error = max(error[0] for error in errors)
                max_relative_error = max(error[1] for error in errors)
                kernel_ms, resident_ms, with_transfers_ms = _time_cusadi_gpu(
                    torch,
                    cusadi_function,
                    cpu_inputs,
                    args.eval_warmup,
                    args.eval_repeats,
                )
                for cpu_cores in args.cpu_cores:
                    cpu_ms = cpu_timings[(oracle_kind, batch_size, cpu_cores)]
                    results.append(
                        NlpOracleBenchmarkResult(
                            case=case,
                            oracle_kind=oracle_kind,
                            n_shooting=n_shooting,
                            decision_variables=n_variables,
                            constraints=n_constraints,
                            output_nonzeros=sum(
                                function.nnz_out(index)
                                for index in range(function.n_out())
                            ),
                            instructions=function.n_instructions(),
                            work_size=function.sz_w(),
                            batch_size=batch_size,
                            cpu_cores=cpu_cores,
                            gpu_block_size=gpu_block_size,
                            gpu_grid_size=math.ceil(batch_size / gpu_block_size),
                            cuda_threads=batch_size,
                            max_absolute_error=max_absolute_error,
                            max_relative_error=max_relative_error,
                            casadi_cpu_ms=cpu_ms,
                            cusadi_kernel_ms=kernel_ms,
                            cusadi_gpu_resident_ms=resident_ms,
                            cusadi_with_transfers_ms=with_transfers_ms,
                            resident_speedup=cpu_ms / resident_ms,
                            end_to_end_speedup=cpu_ms / with_transfers_ms,
                        )
                    )
                del cusadi_function, gpu_inputs, candidate_outputs
                torch.cuda.empty_cache()
    return metadata, results


def _print_solve_result(result: OcpSolveResult) -> None:
    solve = f"{result.solve_wall_s:.3f}s" if result.solve_wall_s is not None else "—"
    iterations = str(result.iterations) if result.iterations is not None else "—"
    print(
        f"CPU solve {result.case:20s} cores={result.cpu_cores:2d} "
        f"outcome={result.outcome:14s} wall={solve:>9s} iter={iterations}"
    )


def _print_dynamics_results(case: str, rows: list[DynamicsBenchmarkResult]) -> None:
    print(f"\nDynamics: {case}")
    print(
        f"{'batch':>8} {'CPU':>4} {'block':>6} {'CPU ms':>10} {'GPU ms':>10} "
        f"{'GPU+copy':>10} {'speedup':>9} {'e2e':>9}"
    )
    for row in rows:
        print(
            f"{row.batch_size:8d} {row.cpu_cores:4d} {row.gpu_block_size:6d} "
            f"{row.casadi_cpu_ms:10.4f} {row.cusadi_gpu_resident_ms:10.4f} "
            f"{row.cusadi_with_transfers_ms:10.4f} {row.resident_speedup:9.2f} "
            f"{row.end_to_end_speedup:9.2f}"
        )


def _print_derivative_results(case: str, rows: list[DerivativeBenchmarkResult]) -> None:
    for derivative_kind in ("jacobian", "hessian"):
        selected_rows = [row for row in rows if row.derivative_kind == derivative_kind]
        print(f"\nDerivative: {case} {derivative_kind}")
        print(
            f"{'batch':>8} {'CPU':>4} {'block':>6} {'CPU ms':>10} {'GPU ms':>10} "
            f"{'GPU+copy':>10} {'speedup':>9} {'e2e':>9}"
        )
        for row in selected_rows:
            print(
                f"{row.batch_size:8d} {row.cpu_cores:4d} {row.gpu_block_size:6d} "
                f"{row.casadi_cpu_ms:10.4f} {row.cusadi_gpu_resident_ms:10.4f} "
                f"{row.cusadi_with_transfers_ms:10.4f} {row.resident_speedup:9.2f} "
                f"{row.end_to_end_speedup:9.2f}"
            )


def _print_nlp_oracle_results(case: str, rows: list[NlpOracleBenchmarkResult]) -> None:
    for oracle_kind in NLP_ORACLE_KINDS:
        selected_rows = [row for row in rows if row.oracle_kind == oracle_kind]
        if not selected_rows:
            continue
        print(f"\nFull NLP oracle: {case} {oracle_kind}")
        print(
            f"{'batch':>8} {'CPU':>4} {'block':>6} {'CPU ms':>10} {'GPU ms':>10} "
            f"{'GPU+copy':>10} {'speedup':>9} {'e2e':>9}"
        )
        for row in selected_rows:
            print(
                f"{row.batch_size:8d} {row.cpu_cores:4d} {row.gpu_block_size:6d} "
                f"{row.casadi_cpu_ms:10.4f} {row.cusadi_gpu_resident_ms:10.4f} "
                f"{row.cusadi_with_transfers_ms:10.4f} {row.resident_speedup:9.2f} "
                f"{row.end_to_end_speedup:9.2f}"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", choices=CASE_NAMES, default=list(CASE_NAMES)
    )
    parser.add_argument(
        "--cpu-cores",
        type=_parse_int_list,
        default=_parse_int_list("4,8,16,32"),
        help="Comma-separated CPU thread counts used for OCP solves and mapped dynamics",
    )
    parser.add_argument(
        "--gpu-batch-sizes",
        type=_parse_int_list,
        default=_parse_int_list("256,2048,16384"),
        help="Comma-separated numbers of independent CUDA dynamics evaluations",
    )
    parser.add_argument(
        "--gpu-block-sizes",
        type=_parse_gpu_block_sizes,
        default=_parse_gpu_block_sizes("64,128,256"),
        help="Comma-separated CUDA threads per block",
    )
    parser.add_argument(
        "--gpu-derivatives",
        action="store_true",
        help="Also benchmark the full dynamics Jacobian and weighted Hessian on CPU and GPU",
    )
    parser.add_argument(
        "--derivative-batch-sizes",
        type=_parse_int_list,
        default=_parse_int_list("32,256,2048"),
        help="Comma-separated batch sizes for Jacobian and Hessian measurements",
    )
    parser.add_argument(
        "--gpu-nlp-oracles",
        action="store_true",
        help="Benchmark the complete discretized NLP and matrix-free derivative products",
    )
    parser.add_argument(
        "--nlp-oracles",
        nargs="+",
        choices=NLP_ORACLE_KINDS,
        default=list(NLP_ORACLE_KINDS),
        help="Full-NLP numerical oracles to benchmark",
    )
    parser.add_argument(
        "--nlp-batch-sizes",
        type=_parse_int_list,
        default=_parse_int_list("1,8,32,256"),
        help="Comma-separated trial-point or parallel-OCP batch sizes for full-NLP oracles",
    )
    parser.add_argument(
        "--n-shooting",
        type=_positive_int,
        help="Override every case-specific shooting size",
    )
    parser.add_argument("--solve-repetitions", type=_positive_int, default=1)
    parser.add_argument("--eval-warmup", type=_positive_int, default=5)
    parser.add_argument("--eval-repeats", type=_positive_int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=_positive_int, default=500)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-ocp-solves", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument(
        "--include-dynamics-only-ocp-solves",
        action="store_true",
        help="Run intentionally iteration-limited OCP solves for cases normally restricted to dynamics measurements",
    )
    parser.add_argument(
        "--cusadi-root", type=Path, default=os.environ.get("CUSADI_ROOT")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("cusadi_multi_dynamics_results.json")
    )

    parser.add_argument("--_cpu-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-case", choices=CASE_NAMES, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-n-shooting", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-cpu-cores", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--_worker-repetition", type=int, default=0, help=argparse.SUPPRESS
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args._cpu_worker:
        result = _cpu_worker(args)
        print(WORKER_RESULT_PREFIX + json.dumps(asdict(result), allow_nan=False))
        return 0

    if any(cpu_cores > (os.cpu_count() or cpu_cores) for cpu_cores in args.cpu_cores):
        raise ValueError(
            f"Requested CPU core count exceeds the {os.cpu_count()} logical CPUs available"
        )
    if not args.skip_gpu and args.cusadi_root is None:
        raise ValueError(
            "--cusadi-root or the CUSADI_ROOT environment variable is required for GPU measurements"
        )

    solve_results: list[OcpSolveResult] = []
    if not args.skip_ocp_solves:
        for case in args.cases:
            if (
                case in DYNAMICS_ONLY_CASES
                and not args.include_dynamics_only_ocp_solves
            ):
                print(f"CPU solve {case:20s} skipped: {DYNAMICS_ONLY_CASES[case]}")
                continue
            n_shooting = _case_n_shooting(case, args.n_shooting)
            for cpu_cores in args.cpu_cores:
                for repetition in range(args.solve_repetitions):
                    result = _run_cpu_solve(
                        args, case, n_shooting, cpu_cores, repetition
                    )
                    solve_results.append(result)
                    _print_solve_result(result)

    torch = None
    gpu_metadata: dict[str, Any] | None = None
    case_metadata: list[dict[str, Any]] = [
        {
            "case": case,
            "description": CASE_DESCRIPTIONS[case],
            "n_shooting": _case_n_shooting(case, args.n_shooting),
            "complete_ocp_solve_benchmarked": case not in DYNAMICS_ONLY_CASES,
            "complete_ocp_solve_limitation": DYNAMICS_ONLY_CASES.get(case),
        }
        for case in args.cases
    ]
    dynamics_results: list[DynamicsBenchmarkResult] = []
    derivative_metadata: list[dict[str, Any]] = []
    derivative_results: list[DerivativeBenchmarkResult] = []
    nlp_oracle_metadata: list[dict[str, Any]] = []
    nlp_oracle_results: list[NlpOracleBenchmarkResult] = []
    cusadi_root = (
        args.cusadi_root.expanduser().resolve()
        if args.cusadi_root is not None
        else None
    )
    if not args.skip_gpu:
        if platform.system() != "Linux":
            raise RuntimeError("CusADi GPU measurements require Linux")
        torch = _import_torch()
        device = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu_metadata = {
            "name": device.name,
            "compute_capability": ".".join(
                map(str, torch.cuda.get_device_capability())
            ),
            "multiprocessor_count": device.multi_processor_count,
            "max_threads_per_multiprocessor": device.max_threads_per_multi_processor,
            "total_memory_bytes": device.total_memory,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }
        for case in args.cases:
            n_shooting = _case_n_shooting(case, args.n_shooting)
            metadata, rows = _benchmark_gpu_case(
                args, case, n_shooting, cusadi_root, torch
            )
            case_metadata[args.cases.index(case)] = metadata
            dynamics_results.extend(rows)
            _print_dynamics_results(case, rows)
            if args.gpu_derivatives:
                if case in FULL_DERIVATIVE_EXCLUSIONS:
                    print(
                        f"GPU derivatives {case:17s} skipped: {FULL_DERIVATIVE_EXCLUSIONS[case]}"
                    )
                else:
                    case_derivative_metadata, derivative_rows = (
                        _benchmark_gpu_derivatives_case(
                            args,
                            case,
                            n_shooting,
                            cusadi_root,
                            torch,
                        )
                    )
                    derivative_metadata.extend(case_derivative_metadata)
                    derivative_results.extend(derivative_rows)
                    _print_derivative_results(case, derivative_rows)
            if args.gpu_nlp_oracles:
                case_nlp_metadata, nlp_rows = _benchmark_gpu_nlp_oracles_case(
                    args,
                    case,
                    n_shooting,
                    cusadi_root,
                    torch,
                )
                nlp_oracle_metadata.extend(case_nlp_metadata)
                nlp_oracle_results.extend(nlp_rows)
                _print_nlp_oracle_results(case, nlp_rows)

    payload = {
        "metadata": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "bioptim": bioptim.__version__,
            "casadi": casadi.__version__,
            "numpy": np.__version__,
            "logical_cpu_count": os.cpu_count(),
            "cpu_core_counts": args.cpu_cores,
            "native_thread_limits": list(THREAD_ENVIRONMENT_VARIABLES),
            "gpu": gpu_metadata,
            "bioptim_revision": _git_revision(Path(__file__).resolve().parents[4]),
            "cusadi_revision": (
                _git_revision(cusadi_root) if cusadi_root is not None else None
            ),
            "gpu_ocp_solve_supported": False,
            "gpu_ocp_solve_limitation": (
                "CusADi evaluates batched CasADi functions but does not provide an NLP solver. This benchmark can "
                "evaluate the complete discretized NLP and matrix-free derivative products on the GPU, but "
                "globalization and the sparse KKT solve still require a host-side solver."
            ),
        },
        "configuration": {
            "cases": args.cases,
            "case_default_n_shooting": DEFAULT_N_SHOOTING,
            "dynamics_only_cases": DYNAMICS_ONLY_CASES,
            "n_shooting_override": args.n_shooting,
            "solve_repetitions": args.solve_repetitions,
            "include_dynamics_only_ocp_solves": args.include_dynamics_only_ocp_solves,
            "gpu_batch_sizes": args.gpu_batch_sizes,
            "gpu_block_sizes": args.gpu_block_sizes,
            "gpu_derivatives": args.gpu_derivatives,
            "derivative_batch_sizes": args.derivative_batch_sizes,
            "gpu_nlp_oracles": args.gpu_nlp_oracles,
            "nlp_oracles": args.nlp_oracles,
            "nlp_batch_sizes": args.nlp_batch_sizes,
            "full_derivative_exclusions": FULL_DERIVATIVE_EXCLUSIONS,
            "eval_warmup": args.eval_warmup,
            "eval_repeats": args.eval_repeats,
            "tolerance": args.tolerance,
            "max_iterations": args.max_iterations,
            "rtol": args.rtol,
            "atol": args.atol,
            "seed": args.seed,
        },
        "cases": case_metadata,
        "ocp_solves": [asdict(result) for result in solve_results],
        "dynamics_benchmarks": [asdict(result) for result in dynamics_results],
        "derivative_functions": derivative_metadata,
        "derivative_benchmarks": [asdict(result) for result in derivative_results],
        "nlp_oracle_functions": nlp_oracle_metadata,
        "nlp_oracle_benchmarks": [asdict(result) for result in nlp_oracle_results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"\nWrote benchmark results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
