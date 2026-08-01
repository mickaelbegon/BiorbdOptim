"""Solve a Bioptim OCP with IPOPT and CPU or GPU CasADi-function callbacks.

CasADi keeps the complete NLP structure and IPOPT keeps its globalization and
sparse KKT solve on the CPU.  The numerical functions requested by IPOPT are
interchangeable: ordinary CasADi evaluation on the CPU or CusADi kernels on an
NVIDIA GPU.  The functions are the primal objective/constraints, objective
gradient, sparse constraint Jacobian, and sparse Hessian of the Lagrangian.

This is a hybrid solve, not an all-GPU NLP solver.  It deliberately measures a
single OCP (CusADi batch size one), including callback and transfer overhead.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import casadi
import numpy as np

import bioptim
from bioptim.examples.toy_examples.gpu.cusadi_multi_dynamics_benchmark import (
    CASE_NAMES,
    DEFAULT_N_SHOOTING,
    _case_n_shooting,
    _compile_cuda_source,
    _generate_cuda_source,
    _git_revision,
    _import_torch,
    _load_cusadi_function,
    _nlp_expressions,
    _positive_int,
    _prepare_case,
)


FUNCTION_ROLES = ("nlp", "grad_f", "jac_g", "hess_lag")


@dataclass
class HybridSolveResult:
    case: str
    backend: str
    n_shooting: int
    decision_variables: int
    constraints: int
    outcome: str
    success: bool
    return_status: str
    iterations: int
    nlp_calls: int
    gradient_calls: int
    jacobian_calls: int
    hessian_calls: int
    oracle_wall_s: float
    solve_wall_s: float
    solver_wall_s: float | None
    cost: float
    max_constraint_violation: float
    compilation_s: float = 0.0
    max_cpu_gpu_decision_difference: float | None = None


def _named_ipopt_functions(
    case: str,
    ocp: Any,
    gpu_block_size: int,
    hessian_approximation: str = "exact",
) -> tuple[dict[str, casadi.Function], Any]:
    """Build functions with the exact signatures expected by CasADi nlpsol."""

    decision_vector, objective, constraints, constraint_bounds = _nlp_expressions(ocp)
    n_variables = int(decision_vector.shape[0])
    n_constraints = int(constraints.shape[0])
    transcription = casadi.Function(
        f"{case}_IpoptTranscription",
        [decision_vector],
        [objective, constraints],
    )

    x = casadi.SX.sym("x", n_variables)
    p = casadi.SX.sym("p", 0)
    objective_at_x, constraints_at_x = transcription(x)
    objective_gradient = casadi.gradient(objective_at_x, x)
    constraint_jacobian = casadi.jacobian(constraints_at_x, x)
    prefix = f"{case}_block_{gpu_block_size}_Ipopt"
    functions = {
        "nlp": casadi.Function(
            f"{prefix}Nlp",
            [x, p],
            [objective_at_x, constraints_at_x],
            ["x", "p"],
            ["f", "g"],
        ),
        "grad_f": casadi.Function(
            f"{prefix}GradF",
            [x, p],
            [objective_at_x, objective_gradient],
            ["x", "p"],
            ["f", "grad_f_x"],
        ),
        "jac_g": casadi.Function(
            f"{prefix}JacG",
            [x, p],
            [constraints_at_x, constraint_jacobian],
            ["x", "p"],
            ["g", "jac_g_x"],
        ),
    }
    if hessian_approximation == "exact":
        objective_factor = casadi.SX.sym("lam_f")
        constraint_multipliers = casadi.SX.sym("lam_g", n_constraints)
        lagrangian = objective_factor * objective_at_x + casadi.dot(
            constraint_multipliers, constraints_at_x
        )
        upper_hessian = casadi.triu(casadi.hessian(lagrangian, x)[0])
        functions["hess_lag"] = casadi.Function(
            f"{prefix}HessLag",
            [x, p, objective_factor, constraint_multipliers],
            [upper_hessian],
            ["x", "p", "lam_f", "lam_g"],
            ["triu_hess_gamma_x_x"],
        )
    return functions, constraint_bounds


class _FunctionBackend:
    def __init__(self, functions: dict[str, casadi.Function]):
        self.functions = functions
        self.function_roles = {
            function.name(): role for role, function in functions.items()
        }
        self.calls = {role: 0 for role in functions}
        self.wall_s = 0.0

    def reset_statistics(self) -> None:
        self.calls = {role: 0 for role in self.functions}
        self.wall_s = 0.0

    def _evaluate_sparse(
        self,
        function: casadi.Function,
        inputs: list[np.ndarray],
    ) -> list[np.ndarray]:
        raise NotImplementedError

    def evaluate(
        self,
        function: casadi.Function,
        inputs: list[np.ndarray],
    ) -> list[np.ndarray]:
        start = time.perf_counter()
        outputs = self._evaluate_sparse(function, inputs)
        self.wall_s += time.perf_counter() - start
        self.calls[self.function_roles[function.name()]] += 1
        return outputs


class _CasadiCpuBackend(_FunctionBackend):
    def _evaluate_sparse(
        self,
        function: casadi.Function,
        inputs: list[np.ndarray],
    ) -> list[np.ndarray]:
        outputs = function.call([casadi.DM(value) for value in inputs])
        return [np.asarray(output.nonzeros(), dtype=np.float64) for output in outputs]


class _CusadiGpuBackend(_FunctionBackend):
    def __init__(
        self,
        functions: dict[str, casadi.Function],
        gpu_functions: dict[str, Any],
        torch: Any,
    ):
        super().__init__(functions)
        self.gpu_functions = gpu_functions
        self.torch = torch

    def _evaluate_sparse(
        self,
        function: casadi.Function,
        inputs: list[np.ndarray],
    ) -> list[np.ndarray]:
        gpu_inputs = [
            self.torch.from_numpy(
                np.ascontiguousarray(value, dtype=np.float64).reshape(1, -1)
            )
            .to(device="cuda", dtype=self.torch.float64)
            .contiguous()
            for value in inputs
        ]
        gpu_function = self.gpu_functions[self.function_roles[function.name()]]
        gpu_function.evaluate(gpu_inputs)
        return [
            output.detach().cpu().numpy()[0].copy()
            for output in gpu_function.outputs_sparse
        ]


class _BackendCallback(casadi.Callback):
    """Expose one CPU/GPU numerical function through CasADi's Function ABI."""

    def __init__(
        self,
        name: str,
        function: casadi.Function,
        backend: _FunctionBackend,
    ):
        casadi.Callback.__init__(self)
        self.function = function
        self.backend = backend
        self.derivative_callbacks: list[casadi.Callback] = []
        self.construct(name, {})

    def get_n_in(self) -> int:
        return self.function.n_in()

    def get_n_out(self) -> int:
        return self.function.n_out()

    def get_name_in(self, index: int) -> str:
        return self.function.name_in(index)

    def get_name_out(self, index: int) -> str:
        return self.function.name_out(index)

    def get_sparsity_in(self, index: int) -> casadi.Sparsity:
        return self.function.sparsity_in(index)

    def get_sparsity_out(self, index: int) -> casadi.Sparsity:
        return self.function.sparsity_out(index)

    def eval(self, arguments: list[casadi.DM]) -> list[casadi.DM]:
        inputs = [
            np.asarray(argument.nonzeros(), dtype=np.float64) for argument in arguments
        ]
        outputs = self.backend.evaluate(self.function, inputs)
        return [
            casadi.DM(self.function.sparsity_out(index), output)
            for index, output in enumerate(outputs)
        ]

    def has_jacobian(self) -> bool:
        return self.backend.function_roles[self.function.name()] == "nlp"

    def get_jacobian(
        self,
        name: str,
        input_names: list[str],
        output_names: list[str],
        options: dict[str, Any],
    ) -> casadi.Function:
        derivative = _NlpJacobianCallback(
            name,
            input_names,
            output_names,
            self.backend,
        )
        self.derivative_callbacks.append(derivative)
        return derivative


class _NlpJacobianCallback(casadi.Callback):
    """Derivative bridge CasADi needs when creating its internal nlp_grad."""

    def __init__(
        self,
        name: str,
        input_names: list[str],
        output_names: list[str],
        backend: _FunctionBackend,
    ):
        casadi.Callback.__init__(self)
        self.input_names = input_names
        self.output_names = output_names
        self.backend = backend
        self.nlp = backend.functions["nlp"]
        self.grad_f = backend.functions["grad_f"]
        self.jac_g = backend.functions["jac_g"]
        self.construct(name, {})

    def get_n_in(self) -> int:
        return 4

    def get_n_out(self) -> int:
        return 4

    def get_name_in(self, index: int) -> str:
        return self.input_names[index]

    def get_name_out(self, index: int) -> str:
        return self.output_names[index]

    def get_sparsity_in(self, index: int) -> casadi.Sparsity:
        if index < self.nlp.n_in():
            return self.nlp.sparsity_in(index)
        return self.nlp.sparsity_out(index - self.nlp.n_in())

    def get_sparsity_out(self, index: int) -> casadi.Sparsity:
        if index == 0:
            return self.grad_f.sparsity_out(1).T
        if index == 1:
            return casadi.Sparsity(self.nlp.size1_out(0), self.nlp.size1_in(1))
        if index == 2:
            return self.jac_g.sparsity_out(1)
        return casadi.Sparsity(self.nlp.size1_out(1), self.nlp.size1_in(1))

    def eval(self, arguments: list[casadi.DM]) -> list[casadi.DM]:
        x = np.asarray(arguments[0].nonzeros(), dtype=np.float64)
        p = np.asarray(arguments[1].nonzeros(), dtype=np.float64)
        gradient = self.backend.evaluate(self.grad_f, [x, p])[1]
        jacobian = self.backend.evaluate(self.jac_g, [x, p])[1]
        return [
            casadi.DM(self.get_sparsity_out(0), gradient),
            casadi.DM(self.get_sparsity_out(1)),
            casadi.DM(self.get_sparsity_out(2), jacobian),
            casadi.DM(self.get_sparsity_out(3)),
        ]


def _maximum_violation(
    values: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> float:
    lower_violation = np.maximum(lower_bounds - values, 0.0)
    upper_violation = np.maximum(values - upper_bounds, 0.0)
    return (
        float(np.max(np.maximum(lower_violation, upper_violation)))
        if values.size
        else 0.0
    )


def _solve_with_ipopt(
    case: str,
    n_shooting: int,
    ocp: Any,
    functions: dict[str, casadi.Function],
    backend: _FunctionBackend,
    constraint_bounds: Any,
    tolerance: float,
    max_iterations: int,
    linear_solver: str,
    hessian_approximation: str,
    compilation_s: float = 0.0,
) -> tuple[HybridSolveResult, np.ndarray]:
    backend_name = type(backend).__name__.removeprefix("_").removesuffix("Backend")
    callback_prefix = f"{case}_{backend_name.lower()}"
    callbacks = {
        role: _BackendCallback(
            f"{callback_prefix}_{role}",
            function,
            backend,
        )
        for role, function in functions.items()
    }
    options = {
        "grad_f": callbacks["grad_f"],
        "jac_g": callbacks["jac_g"],
        "ipopt.print_level": 0,
        "ipopt.tol": tolerance,
        "ipopt.constr_viol_tol": tolerance,
        "ipopt.max_iter": max_iterations,
        "ipopt.linear_solver": linear_solver,
        "ipopt.hessian_approximation": hessian_approximation,
        "print_time": False,
        "error_on_fail": False,
    }
    if hessian_approximation == "exact":
        options["hess_lag"] = callbacks["hess_lag"]
    solver = casadi.nlpsol(
        f"{callback_prefix}_solver",
        "ipopt",
        callbacks["nlp"],
        options,
    )

    initial_guess = np.asarray(ocp.init_vector, dtype=np.float64).reshape(-1)
    variable_lower = np.asarray(ocp.bounds_vectors[0], dtype=np.float64).reshape(-1)
    variable_upper = np.asarray(ocp.bounds_vectors[1], dtype=np.float64).reshape(-1)
    constraint_lower = np.asarray(constraint_bounds.min, dtype=np.float64).reshape(-1)
    constraint_upper = np.asarray(constraint_bounds.max, dtype=np.float64).reshape(-1)

    backend.reset_statistics()
    start = time.perf_counter()
    solution = solver(
        x0=initial_guess,
        lbx=variable_lower,
        ubx=variable_upper,
        lbg=constraint_lower,
        ubg=constraint_upper,
    )
    solve_wall_s = time.perf_counter() - start
    statistics = solver.stats()
    decision = np.asarray(solution["x"], dtype=np.float64).reshape(-1)
    constraint_values = np.asarray(solution["g"], dtype=np.float64).reshape(-1)
    violation = _maximum_violation(
        constraint_values,
        constraint_lower,
        constraint_upper,
    )
    success = bool(statistics["success"])
    outcome = (
        "success" if success and violation <= 10.0 * tolerance else "solver_failure"
    )
    solver_wall_s = statistics.get("t_wall_total")
    if solver_wall_s is not None and not np.isfinite(solver_wall_s):
        solver_wall_s = None
    result = HybridSolveResult(
        case=case,
        backend=backend_name,
        n_shooting=n_shooting,
        decision_variables=initial_guess.size,
        constraints=constraint_lower.size,
        outcome=outcome,
        success=success,
        return_status=str(statistics["return_status"]),
        iterations=int(statistics["iter_count"]),
        nlp_calls=backend.calls["nlp"],
        gradient_calls=backend.calls["grad_f"],
        jacobian_calls=backend.calls["jac_g"],
        hessian_calls=backend.calls.get("hess_lag", 0),
        oracle_wall_s=backend.wall_s,
        solve_wall_s=solve_wall_s,
        solver_wall_s=float(solver_wall_s) if solver_wall_s is not None else None,
        cost=float(solution["f"]),
        max_constraint_violation=violation,
        compilation_s=compilation_s,
    )
    return result, decision


def _validate_backends(
    cpu_backend: _FunctionBackend,
    gpu_backend: _FunctionBackend,
    functions: dict[str, casadi.Function],
    initial_guess: np.ndarray,
    n_constraints: int,
    hessian_approximation: str,
    rtol: float,
    atol: float,
) -> None:
    rng = np.random.default_rng(42)
    inputs = {
        "nlp": [initial_guess, np.zeros(0)],
        "grad_f": [initial_guess, np.zeros(0)],
        "jac_g": [initial_guess, np.zeros(0)],
    }
    if hessian_approximation == "exact":
        inputs["hess_lag"] = [
            initial_guess,
            np.zeros(0),
            np.ones(1),
            rng.normal(size=n_constraints),
        ]
    for role, function in functions.items():
        if role == "hess_lag" and hessian_approximation != "exact":
            continue
        references = cpu_backend.evaluate(function, inputs[role])
        candidates = gpu_backend.evaluate(function, inputs[role])
        for reference, candidate in zip(references, candidates):
            np.testing.assert_allclose(
                candidate,
                reference,
                rtol=rtol,
                atol=atol,
            )
    cpu_backend.reset_statistics()
    gpu_backend.reset_statistics()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASE_NAMES, default=["pendulum"])
    parser.add_argument(
        "--n-shooting",
        type=_positive_int,
        default=10,
        help="Shooting intervals per case",
    )
    parser.add_argument("--gpu-block-size", type=_positive_int, default=256)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=_positive_int, default=500)
    parser.add_argument("--linear-solver", default="mumps")
    parser.add_argument(
        "--hessian-approximation",
        choices=("exact", "limited-memory"),
        default="exact",
        help="IPOPT Hessian mode; limited-memory avoids compiling the exact Hessian kernel",
    )
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recompile CUDA even when the generated source is unchanged",
    )
    parser.add_argument(
        "--cusadi-root",
        type=Path,
        default=os.environ.get("CUSADI_ROOT"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cusadi_hybrid_ocp_results.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.gpu_block_size < 32
        or args.gpu_block_size > 1024
        or args.gpu_block_size % 32
    ):
        raise ValueError(
            "--gpu-block-size must be a multiple of 32 between 32 and 1024"
        )
    if not args.skip_gpu and args.cusadi_root is None:
        raise ValueError(
            "--cusadi-root or the CUSADI_ROOT environment variable is required"
        )

    torch = None
    gpu_metadata = None
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
            "total_memory_bytes": device.total_memory,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }

    results: list[HybridSolveResult] = []
    function_metadata: list[dict[str, Any]] = []
    for case in args.cases:
        n_shooting = _case_n_shooting(case, args.n_shooting)
        ocp = _prepare_case(case, n_shooting, cpu_cores=1)
        functions, constraint_bounds = _named_ipopt_functions(
            case,
            ocp,
            args.gpu_block_size,
            args.hessian_approximation,
        )
        function_metadata.append(
            {
                "case": case,
                "decision_variables": int(ocp.variables_vector.shape[0]),
                "constraints": functions["jac_g"].size1_out(1),
                "functions": {
                    role: {
                        "instructions": function.n_instructions(),
                        "work_size": function.sz_w(),
                        "output_nonzeros": [
                            function.nnz_out(index) for index in range(function.n_out())
                        ],
                    }
                    for role, function in functions.items()
                },
            }
        )

        cpu_backend = _CasadiCpuBackend(functions)
        cpu_result, cpu_decision = _solve_with_ipopt(
            case,
            n_shooting,
            ocp,
            functions,
            cpu_backend,
            constraint_bounds,
            args.tolerance,
            args.max_iterations,
            args.linear_solver,
            args.hessian_approximation,
        )
        results.append(cpu_result)
        print(
            f"CPU oracle {case:20s} outcome={cpu_result.outcome:14s} "
            f"wall={cpu_result.solve_wall_s:9.3f}s iter={cpu_result.iterations}"
        )

        if not args.skip_gpu:
            compile_start = time.perf_counter()
            operations: dict[str, list[str]] = {}
            reused_libraries: dict[str, bool] = {}
            capability = torch.cuda.get_device_capability()
            architecture = 10 * capability[0] + capability[1]
            gpu_roles = ["nlp", "grad_f", "jac_g"]
            if args.hessian_approximation == "exact":
                gpu_roles.append("hess_lag")
            for role in gpu_roles:
                function = functions[role]
                source_path = cusadi_root / "codegen" / f"{function.name()}.cu"
                previous_source = (
                    source_path.read_text(encoding="utf-8")
                    if source_path.is_file()
                    else None
                )
                generated_source, operations[role] = _generate_cuda_source(
                    function,
                    cusadi_root,
                    args.gpu_block_size,
                )
                library = cusadi_root / "build" / f"lib{function.name()}.so"
                source_unchanged = (
                    previous_source is not None
                    and generated_source.read_text(encoding="utf-8") == previous_source
                )
                reuse_library = (
                    library.is_file() and source_unchanged and not args.rebuild
                )
                reused_libraries[role] = reuse_library
                if not reuse_library:
                    _compile_cuda_source(
                        cusadi_root,
                        function.name(),
                        [architecture],
                    )
            compilation_s = time.perf_counter() - compile_start
            function_metadata[-1]["operations"] = operations
            function_metadata[-1]["reused_libraries"] = reused_libraries
            gpu_functions = {
                role: _load_cusadi_function(cusadi_root, functions[role], 1)
                for role in gpu_roles
            }
            function_metadata[-1]["gpu_functions"] = gpu_roles
            gpu_backend = _CusadiGpuBackend(
                functions,
                gpu_functions,
                torch,
            )
            initial_guess = np.asarray(ocp.init_vector, dtype=np.float64).reshape(-1)
            _validate_backends(
                cpu_backend,
                gpu_backend,
                functions,
                initial_guess,
                int(functions["jac_g"].size1_out(1)),
                args.hessian_approximation,
                args.rtol,
                args.atol,
            )
            gpu_result, gpu_decision = _solve_with_ipopt(
                case,
                n_shooting,
                ocp,
                functions,
                gpu_backend,
                constraint_bounds,
                args.tolerance,
                args.max_iterations,
                args.linear_solver,
                args.hessian_approximation,
                compilation_s=compilation_s,
            )
            decision_difference = float(np.max(np.abs(cpu_decision - gpu_decision)))
            cpu_result.max_cpu_gpu_decision_difference = decision_difference
            gpu_result.max_cpu_gpu_decision_difference = decision_difference
            results.append(gpu_result)
            print(
                f"GPU oracle {case:20s} outcome={gpu_result.outcome:14s} "
                f"wall={gpu_result.solve_wall_s:9.3f}s iter={gpu_result.iterations} "
                f"compile={compilation_s:.3f}s"
            )

    payload = {
        "metadata": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "bioptim": bioptim.__version__,
            "casadi": casadi.__version__,
            "numpy": np.__version__,
            "gpu": gpu_metadata,
            "bioptim_revision": _git_revision(Path(__file__).resolve().parents[4]),
            "cusadi_revision": (
                _git_revision(cusadi_root) if cusadi_root is not None else None
            ),
            "architecture": (
                "CasADi nlpsol/IPOPT with CPU KKT solve and interchangeable CPU CasADi or GPU CusADi "
                "objective, constraints, gradient, sparse Jacobian, and sparse Lagrangian-Hessian callbacks"
            ),
            "gpu_batch_size": 1,
        },
        "configuration": {
            "cases": args.cases,
            "case_default_n_shooting": DEFAULT_N_SHOOTING,
            "n_shooting": args.n_shooting,
            "gpu_block_size": args.gpu_block_size,
            "tolerance": args.tolerance,
            "max_iterations": args.max_iterations,
            "linear_solver": args.linear_solver,
            "hessian_approximation": args.hessian_approximation,
            "rtol": args.rtol,
            "atol": args.atol,
            "rebuild": args.rebuild,
        },
        "functions": function_metadata,
        "solves": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote hybrid OCP results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
