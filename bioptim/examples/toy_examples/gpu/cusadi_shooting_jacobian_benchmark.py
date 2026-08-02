"""Benchmark an exact multiple-shooting Jacobian assembled from batched JVPs.

The full discretized constraint graph is kept on the host.  Bioptim already
groups its constraints by shooting node, so this benchmark extracts one local
constraint block, differentiates it in a seed direction, and evaluates all
node/column pairs in one CusADi batch.  The resulting directional derivatives
are assembled into the sparse global Jacobian on the CPU.

This avoids generating one CUDA translation unit for the complete NLP
Jacobian.  It assumes a single phase with homogeneous shooting blocks; the
primal blocks and global Jacobian-vector products are checked before timings
are reported.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from statistics import median
from typing import Any

import casadi
import numpy as np

import bioptim
from bioptim import OrderingStrategy
from bioptim.examples.toy_examples.gpu.cusadi_multi_dynamics_benchmark import (
    CASE_DESCRIPTIONS,
    _nlp_expressions,
    _prepare_case,
    _time_casadi_cpu,
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


SUPPORTED_CASES = (
    "pendulum",
    "cube",
    "joint_acceleration",
    "contact_inequality",
    "static_arm",
    "muscle_contact",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _gpu_block_size(value: str) -> int:
    parsed = _positive_int(value)
    if parsed < 32 or parsed > 1024 or parsed % 32:
        raise argparse.ArgumentTypeError(
            "GPU block size must be a multiple of 32 between 32 and 1024"
        )
    return parsed


def _node_constraint_expressions(
    ocp: Any,
) -> tuple[casadi.SX | casadi.MX, list[casadi.SX | casadi.MX]]:
    """Return the full constraints and their node-major Bioptim blocks."""

    decision_vector, _, constraints, _ = _nlp_expressions(ocp)
    if len(ocp.nlp) != 1:
        raise NotImplementedError("Only single-phase OCPs are supported")

    nlp = ocp.nlp[0]
    interface = ocp.ocp_solver
    penalty_groups: list[dict[int, casadi.SX | casadi.MX]] = []
    for penalties in (nlp.g_internal, nlp.g):
        values, _ = interface.get_all_penalties(nlp, penalties, get_bounds=True)
        penalty_groups.append(values)

    blocks = []
    for node in range(nlp.ns + 1):
        parts = [
            values[node]
            for values in penalty_groups
            if node in values and values[node].numel()
        ]
        blocks.append(casadi.vertcat(*parts) if parts else ocp.cx())

    nonempty_blocks = [block for block in blocks if block.numel()]
    assembled = casadi.vertcat(*nonempty_blocks)
    if assembled.shape != constraints.shape:
        raise NotImplementedError(
            "The NLP contains phase-level or terminal constraints outside the shooting blocks"
        )

    difference = casadi.Function(
        "shooting_constraint_partition_error",
        [decision_vector],
        [assembled - constraints],
    )
    initial_guess = np.asarray(ocp.init_vector, dtype=np.float64).reshape(-1)
    np.testing.assert_allclose(
        np.asarray(difference(initial_guess), dtype=np.float64),
        0.0,
        rtol=0.0,
        atol=1e-12,
    )
    return constraints, nonempty_blocks


def _dependency_columns(
    decision_vector: casadi.SX | casadi.MX,
    block: casadi.SX | casadi.MX,
) -> tuple[np.ndarray, casadi.Sparsity]:
    block_function = casadi.Function(
        "shooting_block_dependency", [decision_vector], [block]
    )
    global_sparsity = block_function.sparsity_jac(0, 0)
    _, columns = global_sparsity.get_triplet()
    dependencies = np.asarray(sorted(set(columns)), dtype=np.int64)
    if dependencies.size == 0:
        raise RuntimeError("The shooting constraint block has no decision-variable dependency")
    return dependencies, global_sparsity


def _local_block_functions(
    case: str,
    decision_vector: casadi.SX | casadi.MX,
    block: casadi.SX | casadi.MX,
    dependencies: np.ndarray,
    gpu_block_size: int,
) -> tuple[casadi.Function, casadi.Function, casadi.Sparsity]:
    """Lower one node block to SX and form its directional derivative."""

    local_variables = casadi.SX.sym("local_variables", dependencies.size)
    replacement = casadi.SX.zeros(decision_vector.sparsity())
    replacement[dependencies.tolist()] = local_variables
    local_constraints = casadi.substitute(block, decision_vector, replacement)
    primal = casadi.Function(
        f"{case}_ShootingBlockPrimal",
        [local_variables],
        [local_constraints],
        ["local_variables"],
        ["constraints"],
    )

    direction = casadi.SX.sym("direction", dependencies.size)
    jvp = casadi.Function(
        f"{case}_block_{gpu_block_size}_ShootingBlockJvp",
        [local_variables, direction],
        [casadi.jtimes(local_constraints, local_variables, direction)],
        ["local_variables", "direction"],
        ["constraint_jvp"],
    )
    local_sparsity = primal.sparsity_jac(0, 0)
    return primal, jvp, local_sparsity


def _candidate(ocp: Any, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    initial = np.asarray(ocp.init_vector, dtype=np.float64).reshape(-1)
    lower = np.asarray(ocp.bounds_vectors[0], dtype=np.float64).reshape(-1)
    upper = np.asarray(ocp.bounds_vectors[1], dtype=np.float64).reshape(-1)
    scale = np.maximum(np.abs(initial), 1.0)
    width = np.where(
        np.isfinite(lower) & np.isfinite(upper), upper - lower, 2.0 * scale
    )
    width = np.where(width > 0.0, width, 0.0)
    point = initial + 0.02 * width * rng.normal(size=initial.size)
    return np.minimum(np.maximum(point, lower), upper)


def _batched_jvp_inputs(
    candidate: np.ndarray, dependency_columns: list[np.ndarray]
) -> list[np.ndarray]:
    local_size = dependency_columns[0].size
    identity = np.eye(local_size, dtype=np.float64)
    variables = np.concatenate(
        [np.repeat(candidate[columns][None, :], local_size, axis=0) for columns in dependency_columns],
        axis=0,
    )
    directions = np.tile(identity, (len(dependency_columns), 1))
    return [
        np.ascontiguousarray(variables, dtype=np.float64),
        np.ascontiguousarray(directions, dtype=np.float64),
    ]


def _dense_local_jacobians(
    jvp_values: np.ndarray, n_blocks: int, local_size: int, constraint_size: int
) -> np.ndarray:
    return jvp_values.reshape(n_blocks, local_size, constraint_size).transpose(0, 2, 1)


def _assemble_sparse_jacobian(
    dense_blocks: np.ndarray,
    dependency_columns: list[np.ndarray],
    local_sparsity: casadi.Sparsity,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_rows, local_columns = local_sparsity.get_triplet()
    local_rows = np.asarray(local_rows, dtype=np.int64)
    local_columns = np.asarray(local_columns, dtype=np.int64)
    constraint_size = dense_blocks.shape[1]
    rows = np.concatenate(
        [local_rows + block * constraint_size for block in range(len(dependency_columns))]
    )
    columns = np.concatenate(
        [dependency_columns[block][local_columns] for block in range(len(dependency_columns))]
    )
    values = np.concatenate(
        [dense_blocks[block, local_rows, local_columns] for block in range(len(dependency_columns))]
    )
    return rows, columns, values


def _median_assembly_ms(
    jvp_values: np.ndarray,
    dependency_columns: list[np.ndarray],
    local_sparsity: casadi.Sparsity,
    constraint_size: int,
    warmup: int,
    repeats: int,
) -> float:
    local_size = dependency_columns[0].size

    def assemble() -> None:
        dense = _dense_local_jacobians(
            jvp_values, len(dependency_columns), local_size, constraint_size
        )
        _assemble_sparse_jacobian(dense, dependency_columns, local_sparsity)

    for _ in range(warmup):
        assemble()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        assemble()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return median(samples)


def _validate_partition_and_jacobian(
    decision_vector: casadi.SX | casadi.MX,
    constraints: casadi.SX | casadi.MX,
    blocks: list[casadi.SX | casadi.MX],
    primal: casadi.Function,
    dense_jacobians: np.ndarray,
    dependency_columns: list[np.ndarray],
    candidate: np.ndarray,
    seed: int,
    rtol: float,
    atol: float,
) -> dict[str, float]:
    global_constraints = casadi.Function(
        "global_constraints_for_block_validation", [decision_vector], [constraints]
    )
    reference_constraints = np.asarray(
        global_constraints(candidate), dtype=np.float64
    ).reshape(-1)
    block_constraints = primal.map(len(blocks), "serial")(
        casadi.DM(np.column_stack([candidate[columns] for columns in dependency_columns]))
    )
    block_constraints = np.asarray(block_constraints, dtype=np.float64).reshape(-1, order="F")
    np.testing.assert_allclose(
        block_constraints, reference_constraints, rtol=rtol, atol=atol
    )

    rng = np.random.default_rng(seed)
    direction = rng.normal(size=candidate.size)
    symbolic_direction = casadi.SX.sym("global_direction", candidate.size)
    global_jvp = casadi.Function(
        "global_constraint_jvp_for_block_validation",
        [decision_vector, symbolic_direction],
        [casadi.jtimes(constraints, decision_vector, symbolic_direction)],
    )
    reference_jvp = np.asarray(
        global_jvp(candidate, direction), dtype=np.float64
    ).reshape(-1)
    assembled_jvp = np.concatenate(
        [
            dense_jacobians[block] @ direction[columns]
            for block, columns in enumerate(dependency_columns)
        ]
    )
    np.testing.assert_allclose(assembled_jvp, reference_jvp, rtol=rtol, atol=atol)
    primal_error = _error_metrics(reference_constraints, block_constraints)[0]
    jvp_error = _error_metrics(reference_jvp, assembled_jvp)[0]
    return {
        "max_primal_partition_error": primal_error,
        "max_global_jvp_error": jvp_error,
    }


class _ShootingJacobianBackend:
    """Evaluate and assemble one full sparse Jacobian on CPU or GPU."""

    def __init__(
        self,
        jvp: casadi.Function,
        dependency_columns: list[np.ndarray],
        local_sparsity: casadi.Sparsity,
        constraint_size: int,
        decision_size: int,
        cpu_cores: int,
        torch: Any | None = None,
        gpu_function: Any | None = None,
    ):
        self.jvp = jvp
        self.dependency_columns = dependency_columns
        self.local_sparsity = local_sparsity
        self.constraint_size = constraint_size
        self.local_size = dependency_columns[0].size
        self.batch_size = len(dependency_columns) * self.local_size
        self.torch = torch
        self.gpu_function = gpu_function
        self.mapped_jvp = (
            None
            if gpu_function is not None
            else jvp.map(
                self.batch_size,
                "serial" if cpu_cores == 1 else "thread",
                cpu_cores,
            )
        )

        empty_blocks = np.zeros(
            (len(dependency_columns), constraint_size, self.local_size),
            dtype=np.float64,
        )
        rows, columns, _ = _assemble_sparse_jacobian(
            empty_blocks, dependency_columns, local_sparsity
        )
        self.csc_order = np.lexsort((rows, columns))
        sorted_rows = rows[self.csc_order]
        sorted_columns = columns[self.csc_order]
        self.sparsity = casadi.Sparsity.triplet(
            len(dependency_columns) * constraint_size,
            decision_size,
            sorted_rows.tolist(),
            sorted_columns.tolist(),
        )
        sparsity_rows, sparsity_columns = self.sparsity.get_triplet()
        if not np.array_equal(sparsity_rows, sorted_rows) or not np.array_equal(
            sparsity_columns, sorted_columns
        ):
            raise RuntimeError("CasADi reordered the assembled CSC sparsity unexpectedly")
        self.calls = 0
        self.wall_s = 0.0

    def reset_statistics(self) -> None:
        self.calls = 0
        self.wall_s = 0.0

    def evaluate(self, decision: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        inputs = _batched_jvp_inputs(decision, self.dependency_columns)
        if self.gpu_function is None:
            output = self.mapped_jvp.call(
                [casadi.DM(values.T) for values in inputs]
            )[0]
            jvp_values = np.asarray(output.nonzeros(), dtype=np.float64).reshape(
                self.batch_size, self.jvp.nnz_out(0)
            )
        else:
            gpu_inputs = [
                self.torch.from_numpy(values)
                .to(device="cuda", dtype=self.torch.float64)
                .contiguous()
                for values in inputs
            ]
            self.gpu_function.evaluate(gpu_inputs)
            self.torch.cuda.synchronize()
            jvp_values = (
                self.gpu_function.outputs_sparse[0].detach().cpu().numpy()
            )

        dense_blocks = _dense_local_jacobians(
            jvp_values,
            len(self.dependency_columns),
            self.local_size,
            self.constraint_size,
        )
        _, _, values = _assemble_sparse_jacobian(
            dense_blocks, self.dependency_columns, self.local_sparsity
        )
        self.calls += 1
        self.wall_s += time.perf_counter() - start
        return values[self.csc_order]


class _ShootingJacobianCallback(casadi.Callback):
    """Expose the assembled block Jacobian through CasADi's ``jac_g`` ABI."""

    def __init__(
        self,
        name: str,
        decision_size: int,
        constraint_function: casadi.Function,
        backend: _ShootingJacobianBackend,
    ):
        casadi.Callback.__init__(self)
        self.decision_size = decision_size
        self.constraint_function = constraint_function
        self.backend = backend
        self.construct(name, {})

    def get_n_in(self) -> int:
        return 2

    def get_n_out(self) -> int:
        return 2

    def get_name_in(self, index: int) -> str:
        return ("x", "p")[index]

    def get_name_out(self, index: int) -> str:
        return ("g", "jac_g_x")[index]

    def get_sparsity_in(self, index: int) -> casadi.Sparsity:
        return (
            casadi.Sparsity.dense(self.decision_size, 1)
            if index == 0
            else casadi.Sparsity.dense(0, 1)
        )

    def get_sparsity_out(self, index: int) -> casadi.Sparsity:
        return (
            self.constraint_function.sparsity_out(0)
            if index == 0
            else self.backend.sparsity
        )

    def eval(self, arguments: list[casadi.DM]) -> list[casadi.DM]:
        decision = np.asarray(arguments[0], dtype=np.float64).reshape(-1)
        constraints = self.constraint_function(arguments[0])
        jacobian_values = self.backend.evaluate(decision)
        return [
            constraints,
            casadi.DM(self.backend.sparsity, jacobian_values),
        ]


def _maximum_constraint_violation(
    values: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    return float(
        np.max(
            np.maximum(
                np.maximum(lower - values, 0.0),
                np.maximum(values - upper, 0.0),
            )
        )
    )


def _solve_with_block_jacobian(
    label: str,
    ocp: Any,
    decision_vector: casadi.SX | casadi.MX,
    objective: casadi.SX | casadi.MX,
    constraints: casadi.SX | casadi.MX,
    constraint_bounds: Any,
    backend: _ShootingJacobianBackend,
    tolerance: float,
    max_iterations: int,
) -> tuple[dict[str, Any], np.ndarray]:
    constraint_function = casadi.Function(
        f"{label}_block_constraints", [decision_vector], [constraints]
    )
    jacobian_callback = _ShootingJacobianCallback(
        f"{label}_block_jac_g",
        int(decision_vector.shape[0]),
        constraint_function,
        backend,
    )
    options = {
        "jac_g": jacobian_callback,
        "ipopt.print_level": 0,
        "ipopt.tol": tolerance,
        "ipopt.constr_viol_tol": tolerance,
        "ipopt.max_iter": max_iterations,
        "ipopt.linear_solver": "mumps",
        "ipopt.hessian_approximation": "limited-memory",
        "print_time": False,
        "error_on_fail": False,
    }
    solver_start = time.perf_counter()
    solver = casadi.nlpsol(
        f"{label}_block_solver",
        "ipopt",
        {"x": decision_vector, "f": objective, "g": constraints},
        options,
    )
    solver_setup_s = time.perf_counter() - solver_start

    initial = np.asarray(ocp.init_vector, dtype=np.float64).reshape(-1)
    variable_lower = np.asarray(
        ocp.bounds_vectors[0], dtype=np.float64
    ).reshape(-1)
    variable_upper = np.asarray(
        ocp.bounds_vectors[1], dtype=np.float64
    ).reshape(-1)
    constraint_lower = np.asarray(
        constraint_bounds.min, dtype=np.float64
    ).reshape(-1)
    constraint_upper = np.asarray(
        constraint_bounds.max, dtype=np.float64
    ).reshape(-1)

    backend.reset_statistics()
    solve_start = time.perf_counter()
    solution = solver(
        x0=initial,
        lbx=variable_lower,
        ubx=variable_upper,
        lbg=constraint_lower,
        ubg=constraint_upper,
    )
    solve_s = time.perf_counter() - solve_start
    statistics = solver.stats()
    decision = np.asarray(solution["x"], dtype=np.float64).reshape(-1)
    constraint_values = np.asarray(solution["g"], dtype=np.float64).reshape(-1)
    violation = _maximum_constraint_violation(
        constraint_values, constraint_lower, constraint_upper
    )
    result = {
        "backend": label,
        "success": bool(statistics["success"]),
        "return_status": str(statistics["return_status"]),
        "iterations": int(statistics["iter_count"]),
        "cost": float(solution["f"]),
        "max_constraint_violation": violation,
        "solver_setup_s": solver_setup_s,
        "solve_s": solve_s,
        "solver_setup_and_solve_s": solver_setup_s + solve_s,
        "jacobian_calls": backend.calls,
        "jacobian_wall_s": backend.wall_s,
    }
    return result, decision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=SUPPORTED_CASES, default="contact_inequality")
    parser.add_argument("--n-shooting", type=_positive_int, default=100)
    parser.add_argument("--cpu-cores", type=_positive_int, default=4)
    parser.add_argument("--gpu-block-size", type=_gpu_block_size, default=256)
    parser.add_argument("--warmup", type=_positive_int, default=3)
    parser.add_argument("--repeats", type=_positive_int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument(
        "--solve-ocp",
        action="store_true",
        help="Solve the complete OCP with CPU and GPU shooting-block jac_g callbacks",
    )
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=_positive_int, default=500)
    parser.add_argument(
        "--rebuild", action="store_true", help="Force recompilation of an unchanged CUDA source"
    )
    parser.add_argument("--cusadi-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("cusadi_shooting_jacobian_results.json")
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    start = time.perf_counter()
    ocp = _prepare_case(
        args.case,
        args.n_shooting,
        cpu_cores=1,
        ordering_strategy=OrderingStrategy.TIME_MAJOR,
    )
    decision_vector, objective, _, constraint_bounds = _nlp_expressions(ocp)
    constraints, blocks = _node_constraint_expressions(ocp)
    build_ocp_s = time.perf_counter() - start

    dependency_columns = []
    global_sparsities = []
    for block in blocks:
        dependencies, global_sparsity = _dependency_columns(decision_vector, block)
        dependency_columns.append(dependencies)
        global_sparsities.append(global_sparsity)
    local_sizes = {columns.size for columns in dependency_columns}
    block_sizes = {int(block.shape[0]) for block in blocks}
    if len(local_sizes) != 1 or len(block_sizes) != 1:
        raise NotImplementedError("The shooting blocks are not homogeneous")

    primal, jvp, local_sparsity = _local_block_functions(
        args.case,
        decision_vector,
        blocks[0],
        dependency_columns[0],
        args.gpu_block_size,
    )
    expected_patterns = [
        (
            tuple(global_sparsity.get_triplet()[0]),
            tuple(
                np.searchsorted(columns, global_sparsity.get_triplet()[1]).tolist()
            ),
        )
        for columns, global_sparsity in zip(dependency_columns, global_sparsities)
    ]
    if any(pattern != expected_patterns[0] for pattern in expected_patterns[1:]):
        raise NotImplementedError("The local shooting Jacobian sparsity changes by node")

    candidate = _candidate(ocp, args.seed)
    cpu_inputs = _batched_jvp_inputs(candidate, dependency_columns)
    batch_size = cpu_inputs[0].shape[0]
    cpu_reference = jvp.map(batch_size, "serial").call(
        [casadi.DM(values.T) for values in cpu_inputs]
    )[0]
    cpu_reference = np.asarray(cpu_reference.nonzeros(), dtype=np.float64).reshape(
        batch_size, jvp.nnz_out(0)
    )
    dense_reference = _dense_local_jacobians(
        cpu_reference,
        len(blocks),
        dependency_columns[0].size,
        int(blocks[0].shape[0]),
    )
    validation = _validate_partition_and_jacobian(
        decision_vector,
        constraints,
        blocks,
        primal,
        dense_reference,
        dependency_columns,
        candidate,
        args.seed + 1,
        args.rtol,
        args.atol,
    )

    cpu_ms = _time_casadi_cpu(
        jvp, cpu_inputs, args.cpu_cores, args.warmup, args.repeats
    )
    gpu_preparation_start = time.perf_counter()
    source_path = args.cusadi_root / "codegen" / f"{jvp.name()}.cu"
    previous_source = (
        source_path.read_text(encoding="utf-8") if source_path.is_file() else None
    )
    codegen_start = time.perf_counter()
    generated_source, operations = _generate_cuda_source(
        jvp, args.cusadi_root, args.gpu_block_size
    )
    codegen_s = time.perf_counter() - codegen_start
    torch = _import_torch()
    capability = torch.cuda.get_device_capability()
    compile_start = time.perf_counter()
    library = args.cusadi_root / "build" / f"lib{jvp.name()}.so"
    source_unchanged = (
        previous_source is not None
        and generated_source.read_text(encoding="utf-8") == previous_source
    )
    reused_library = library.is_file() and source_unchanged and not args.rebuild
    if not reused_library:
        _compile_cuda_source(
            args.cusadi_root, jvp.name(), [capability[0] * 10 + capability[1]]
        )
    compilation_s = time.perf_counter() - compile_start
    load_start = time.perf_counter()
    gpu_function = _load_cusadi_function(args.cusadi_root, jvp, batch_size)
    library_load_s = time.perf_counter() - load_start
    gpu_preparation_s = time.perf_counter() - gpu_preparation_start
    gpu_inputs = [
        torch.from_numpy(values).to(device="cuda", dtype=torch.float64).contiguous()
        for values in cpu_inputs
    ]
    gpu_function.evaluate(gpu_inputs)
    torch.cuda.synchronize()
    gpu_values = gpu_function.outputs_sparse[0].detach().cpu().numpy()
    np.testing.assert_allclose(gpu_values, cpu_reference, rtol=args.rtol, atol=args.atol)
    gpu_error = _error_metrics(cpu_reference, gpu_values)
    absolute_gpu_error = np.abs(gpu_values - cpu_reference)
    scaled_gpu_error = absolute_gpu_error / (
        args.atol + args.rtol * np.abs(cpu_reference)
    )
    kernel_ms, resident_ms, transfer_ms = _time_cusadi_gpu(
        torch, gpu_function, cpu_inputs, args.warmup, args.repeats
    )
    assembly_ms = _median_assembly_ms(
        gpu_values,
        dependency_columns,
        local_sparsity,
        int(blocks[0].shape[0]),
        args.warmup,
        args.repeats,
    )
    dense_gpu = _dense_local_jacobians(
        gpu_values,
        len(blocks),
        dependency_columns[0].size,
        int(blocks[0].shape[0]),
    )
    rows, columns, values = _assemble_sparse_jacobian(
        dense_gpu, dependency_columns, local_sparsity
    )
    shared_columns = [
        int(column)
        for column in sorted(
            set(dependency_columns[0]).intersection(
                *(set(block_columns) for block_columns in dependency_columns[1:])
            )
        )
    ]
    nonshared_columns = [
        np.asarray(
            [column for column in block_columns if column not in shared_columns],
            dtype=np.int64,
        )
        for block_columns in dependency_columns
    ]
    nonshared_ranges = [
        [int(block_columns.min()), int(block_columns.max())]
        for block_columns in nonshared_columns
    ]
    nonshared_contiguous = all(
        np.array_equal(
            block_columns,
            np.arange(block_columns.min(), block_columns.max() + 1),
        )
        for block_columns in nonshared_columns
    )

    ocp_solves: list[dict[str, Any]] = []
    solve_comparison: dict[str, float] = {}
    if args.solve_ocp:
        cpu_backend = _ShootingJacobianBackend(
            jvp,
            dependency_columns,
            local_sparsity,
            int(blocks[0].shape[0]),
            int(decision_vector.shape[0]),
            args.cpu_cores,
        )
        gpu_backend = _ShootingJacobianBackend(
            jvp,
            dependency_columns,
            local_sparsity,
            int(blocks[0].shape[0]),
            int(decision_vector.shape[0]),
            args.cpu_cores,
            torch=torch,
            gpu_function=gpu_function,
        )
        cpu_solve, cpu_decision = _solve_with_block_jacobian(
            "CasadiCpuBlocks",
            ocp,
            decision_vector,
            objective,
            constraints,
            constraint_bounds,
            cpu_backend,
            args.tolerance,
            args.max_iterations,
        )
        gpu_solve, gpu_decision = _solve_with_block_jacobian(
            "CusadiGpuBlocks",
            ocp,
            decision_vector,
            objective,
            constraints,
            constraint_bounds,
            gpu_backend,
            args.tolerance,
            args.max_iterations,
        )
        cpu_solve["total_including_ocp_build_s"] = (
            build_ocp_s + cpu_solve["solver_setup_and_solve_s"]
        )
        gpu_solve["total_including_ocp_build_and_gpu_preparation_s"] = (
            build_ocp_s
            + gpu_preparation_s
            + gpu_solve["solver_setup_and_solve_s"]
        )
        ocp_solves.extend((cpu_solve, gpu_solve))
        solve_comparison = {
            "max_cpu_gpu_decision_difference": float(
                np.max(np.abs(cpu_decision - gpu_decision))
            ),
            "solve_speedup": cpu_solve["solve_s"] / gpu_solve["solve_s"],
            "solver_setup_and_solve_speedup": (
                cpu_solve["solver_setup_and_solve_s"]
                / gpu_solve["solver_setup_and_solve_s"]
            ),
            "total_speedup_including_build_and_gpu_preparation": (
                cpu_solve["total_including_ocp_build_s"]
                / gpu_solve[
                    "total_including_ocp_build_and_gpu_preparation_s"
                ]
            ),
        }
        print(
            f"Complete OCP: CPU {cpu_solve['solve_s']:.3f} s, "
            f"GPU {gpu_solve['solve_s']:.3f} s, "
            f"speedup {solve_comparison['solve_speedup']:.2f}x"
        )

    result = {
        "metadata": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "bioptim": bioptim.__version__,
            "casadi": casadi.__version__,
            "numpy": np.__version__,
            "bioptim_revision": _git_revision(Path(__file__).resolve().parents[4]),
            "cusadi_revision": _git_revision(args.cusadi_root),
            "gpu": {
                "name": torch.cuda.get_device_name(),
                "compute_capability": f"{capability[0]}.{capability[1]}",
            },
        },
        "configuration": {
            "case": args.case,
            "description": CASE_DESCRIPTIONS[args.case],
            "n_shooting": args.n_shooting,
            "cpu_cores": args.cpu_cores,
            "gpu_block_size": args.gpu_block_size,
            "ordering_strategy": "TIME_MAJOR",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "solve_ocp": args.solve_ocp,
            "tolerance": args.tolerance,
            "max_iterations": args.max_iterations,
        },
        "structure": {
            "decision_variables": int(decision_vector.shape[0]),
            "constraints": int(constraints.shape[0]),
            "shooting_blocks": len(blocks),
            "constraints_per_block": int(blocks[0].shape[0]),
            "local_variables_per_block": int(dependency_columns[0].size),
            "local_jacobian_nonzeros": local_sparsity.nnz(),
            "assembled_jacobian_nonzeros": int(values.size),
            "jvp_batch_size": batch_size,
            "primal_instructions": primal.n_instructions(),
            "jvp_instructions": jvp.n_instructions(),
            "jvp_work_size": jvp.sz_w(),
            "operations": operations,
            "reused_cuda_library": reused_library,
            "shared_decision_columns": shared_columns,
            "first_nonshared_block_column_range": nonshared_ranges[0],
            "last_nonshared_block_column_range": nonshared_ranges[-1],
            "nonshared_block_columns_are_contiguous": nonshared_contiguous,
            "maximum_nonshared_block_column_span": max(
                end - start + 1 for start, end in nonshared_ranges
            ),
            "assembled_row_range": [int(rows.min()), int(rows.max())],
            "assembled_column_range": [int(columns.min()), int(columns.max())],
        },
        "validation": {
            **validation,
            "max_cpu_gpu_absolute_error": gpu_error[0],
            "max_cpu_gpu_raw_relative_error": gpu_error[1],
            "max_cpu_gpu_allclose_scaled_error": float(np.max(scaled_gpu_error)),
        },
        "timings": {
            "ocp_and_block_build_s": build_ocp_s,
            "cuda_codegen_s": codegen_s,
            "cuda_compilation_s": compilation_s,
            "cuda_library_load_s": library_load_s,
            "gpu_preparation_s": gpu_preparation_s,
            "casadi_cpu_jvp_ms": cpu_ms,
            "cusadi_kernel_ms": kernel_ms,
            "cusadi_gpu_resident_ms": resident_ms,
            "cusadi_with_transfers_ms": transfer_ms,
            "host_sparse_assembly_ms": assembly_ms,
            "gpu_with_transfers_and_assembly_ms": transfer_ms + assembly_ms,
            "resident_speedup": cpu_ms / resident_ms,
            "end_to_end_speedup_before_assembly": cpu_ms / transfer_ms,
            "end_to_end_speedup_after_assembly": cpu_ms / (transfer_ms + assembly_ms),
            "setup_plus_one_jacobian_s": (
                build_ocp_s
                + gpu_preparation_s
                + (transfer_ms + assembly_ms) / 1000.0
            ),
        },
        "ocp_solves": ocp_solves,
        "ocp_solve_comparison": solve_comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(
        f"{args.case}: {len(blocks)} blocks x {dependency_columns[0].size} directions "
        f"= {batch_size} JVPs; CPU {cpu_ms:.3f} ms, GPU+copy {transfer_ms:.3f} ms, "
        f"assembly {assembly_ms:.3f} ms"
    )
    print(f"Wrote shooting-block Jacobian results to {args.output}")


if __name__ == "__main__":
    main()
