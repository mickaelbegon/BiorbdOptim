"""
Benchmark a bioptim-generated dynamics function with CusADi.

The benchmark extracts the ``ForwardDyn`` CasADi function from the basic
pendulum OCP, asks CusADi to generate and compile a CUDA kernel for it, checks
the GPU output against CasADi on the CPU, and compares execution times for
several batch sizes.

This is an experimental benchmark, not a GPU backend for bioptim solvers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import casadi
import numpy as np

from bioptim.examples.getting_started.basic_ocp import prepare_ocp
from bioptim.examples.utils import ExampleUtils


@dataclass
class BatchResult:
    batch_size: int
    max_absolute_error: float
    max_relative_error: float
    casadi_cpu_ms: float
    cusadi_kernel_ms: float
    cusadi_gpu_resident_ms: float
    cusadi_with_transfers_ms: float
    resident_speedup: float
    end_to_end_speedup: float


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _parse_batch_sizes(value: str) -> list[int]:
    try:
        batch_sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from exc
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise argparse.ArgumentTypeError("batch sizes must contain positive integers")
    return batch_sizes


def _run(command: list[str], cwd: Path | None = None) -> str:
    process = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(process.stdout, end="")
    if process.returncode:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")
    return process.stdout


def _git_revision(repository: Path) -> str | None:
    if not (repository / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _build_pendulum_dynamics() -> casadi.Function:
    model_path = Path(ExampleUtils.folder) / "models" / "pendulum.bioMod"
    ocp = prepare_ocp(
        biorbd_model_path=str(model_path),
        final_time=1.0,
        n_shooting=10,
        use_sx=True,
        n_threads=1,
        expand_dynamics=True,
    )
    dynamics = ocp.nlp[0].dynamics_func
    if not isinstance(dynamics, casadi.Function):
        raise TypeError(f"Expected a CasADi Function, got {type(dynamics).__name__}")
    return dynamics


def _load_module(module_name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cusadi_codegen(cusadi_root: Path) -> tuple[types.ModuleType, dict[int, str]]:
    """
    Load only the CusADi code generator.

    Importing the public CusADi package imports PyTorch immediately. Loading the
    two code-generation modules directly keeps ``--codegen-only`` usable on a
    machine without CUDA or PyTorch.
    """

    operations_path = cusadi_root / "src" / "CusadiOperations.py"
    generator_path = cusadi_root / "src" / "generateCUDACode.py"
    if not operations_path.is_file() or not generator_path.is_file():
        raise FileNotFoundError(
            f"{cusadi_root} does not look like a CusADi checkout; expected src/CusadiOperations.py "
            "and src/generateCUDACode.py"
        )

    operations = _load_module("_bioptim_cusadi_operations", operations_path)
    src_stub = types.ModuleType("src")
    for name in dir(operations):
        if not name.startswith("__"):
            setattr(src_stub, name, getattr(operations, name))
    src_stub.CUSADI_ROOT_DIR = str(cusadi_root)

    previous_src = sys.modules.get("src")
    sys.modules["src"] = src_stub
    try:
        generator = _load_module("_bioptim_cusadi_codegen", generator_path)
    finally:
        if previous_src is None:
            sys.modules.pop("src", None)
        else:
            sys.modules["src"] = previous_src

    operation_names = {
        value: name for name, value in vars(casadi).items() if name.startswith("OP_") and isinstance(value, int)
    }
    supported_operations = {
        operation_id: operation_names.get(operation_id, str(operation_id)) for operation_id in operations.OP_CUDA_DICT
    }
    return generator, supported_operations


def _generate_cuda_source(dynamics: casadi.Function, cusadi_root: Path) -> tuple[Path, list[str]]:
    generator, supported_operations = _load_cusadi_codegen(cusadi_root)
    used_operation_ids = {dynamics.instruction_id(index) for index in range(dynamics.n_instructions())}
    unsupported = sorted(used_operation_ids - supported_operations.keys())
    if unsupported:
        names = [
            next(
                (name for name, value in vars(casadi).items() if name.startswith("OP_") and value == operation_id),
                str(operation_id),
            )
            for operation_id in unsupported
        ]
        raise RuntimeError(f"CusADi does not implement the following CasADi operations: {', '.join(names)}")

    function_directory = cusadi_root / "src" / "casadi_functions"
    codegen_directory = cusadi_root / "codegen"
    function_directory.mkdir(parents=True, exist_ok=True)
    codegen_directory.mkdir(parents=True, exist_ok=True)

    dynamics.save(str(function_directory / f"{dynamics.name()}.casadi"))
    cuda_source = codegen_directory / f"{dynamics.name()}.cu"
    generator.generateCUDACodeDouble(
        dynamics,
        filepath=str(cuda_source),
        benchmarking=False,
        debug_mode=False,
    )
    used_operation_names = sorted(supported_operations[operation_id] for operation_id in used_operation_ids)
    return cuda_source, used_operation_names


def _write_cmake_project(cusadi_root: Path, function_name: str, cuda_architecture: int) -> None:
    """
    Write a minimal architecture-aware build for the generated kernel.

    CusADi currently adds ``-arch=sm_86`` unconditionally in its generated
    CMake project. Selecting the architecture reported by PyTorch makes the
    benchmark work on Turing, Ampere, Ada, and newer CUDA devices.
    """

    cmake_contents = f"""cmake_minimum_required(VERSION 3.18)
project(BioptimCusADIBenchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES {cuda_architecture})

add_library({function_name} SHARED codegen/{function_name}.cu)
target_compile_options(
    {function_name}
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
"""
    (cusadi_root / "CMakeLists.txt").write_text(cmake_contents, encoding="utf-8")


def _compile_cuda_source(cusadi_root: Path, function_name: str, cuda_architecture: int) -> Path:
    if shutil.which("cmake") is None:
        raise RuntimeError("cmake is required to compile the generated CUDA source")
    if shutil.which("nvcc") is None:
        raise RuntimeError("nvcc is required; install the NVIDIA CUDA toolkit and add it to PATH")

    _write_cmake_project(cusadi_root, function_name, cuda_architecture)
    build_directory = cusadi_root / "build"
    _run(
        [
            "cmake",
            "-S",
            str(cusadi_root),
            "-B",
            str(build_directory),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architecture}",
        ]
    )
    _run(["cmake", "--build", str(build_directory), "--parallel"])

    library = build_directory / f"lib{function_name}.so"
    if not library.is_file():
        raise FileNotFoundError(f"CusADi build completed but {library} was not produced")
    return library


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the GPU benchmark; install a CUDA-enabled build") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot access a CUDA device. This benchmark requires Linux with an NVIDIA GPU and driver."
        )
    return torch


def _load_cusadi_function(cusadi_root: Path, dynamics: casadi.Function, batch_size: int) -> Any:
    original_path = list(sys.path)
    previous_src = sys.modules.pop("src", None)
    sys.path.insert(0, str(cusadi_root))
    try:
        from src import CusadiFunction
    finally:
        sys.path[:] = original_path
        if previous_src is not None:
            sys.modules["src"] = previous_src
    return CusadiFunction(dynamics, batch_size)


def _make_inputs(dynamics: casadi.Function, batch_size: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs = [
        np.ascontiguousarray(rng.uniform(-1.0, 1.0, size=(batch_size, dynamics.nnz_in(index))), dtype=np.float64)
        for index in range(dynamics.n_in())
    ]

    # The basic pendulum dynamics inputs are [t_span, q/qdot, tau, p, a, d].
    if dynamics.n_in() >= 3 and dynamics.nnz_in(0) == 2 and dynamics.nnz_in(1) == 4:
        inputs[0][:, 0] = rng.uniform(0.0, 1.0, size=batch_size)
        inputs[0][:, 1] = 0.01
        inputs[1][:, :2] *= np.pi
        inputs[1][:, 2:] *= 5.0
        inputs[2] *= 100.0
    return inputs


def _casadi_mapped_output(dynamics: casadi.Function, inputs: list[np.ndarray]) -> np.ndarray:
    batch_size = inputs[0].shape[0]
    mapped = dynamics.map(batch_size, "serial")
    mapped_inputs = [casadi.DM(values.T) for values in inputs]
    output = mapped.call(mapped_inputs)[0]
    return np.asarray(output, dtype=np.float64).T


def _time_casadi_cpu(
    dynamics: casadi.Function,
    inputs: list[np.ndarray],
    warmup: int,
    repeats: int,
) -> float:
    batch_size = inputs[0].shape[0]
    mapped = dynamics.map(batch_size, "serial")
    mapped_inputs = [casadi.DM(values.T) for values in inputs]
    for _ in range(warmup):
        mapped.call(mapped_inputs)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        mapped.call(mapped_inputs)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return median(samples)


def _time_cusadi_gpu(
    torch: Any,
    cusadi_function: Any,
    cpu_inputs: list[np.ndarray],
    warmup: int,
    repeats: int,
) -> tuple[float, float, float]:
    device = torch.device("cuda")
    gpu_inputs = [torch.from_numpy(values).to(device=device, dtype=torch.float64).contiguous() for values in cpu_inputs]

    for _ in range(warmup):
        cusadi_function.evaluate(gpu_inputs)
    torch.cuda.synchronize()

    kernel_samples = []
    resident_samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        cusadi_function.evaluate(gpu_inputs)
        torch.cuda.synchronize()
        resident_samples.append((time.perf_counter_ns() - start) / 1e6)
        kernel_samples.append(float(cusadi_function.eval_time) * 1e3)

    transfer_samples = []
    cpu_tensors = [torch.from_numpy(values) for values in cpu_inputs]
    for _ in range(repeats):
        start = time.perf_counter_ns()
        transferred_inputs = [values.to(device=device, dtype=torch.float64).contiguous() for values in cpu_tensors]
        cusadi_function.evaluate(transferred_inputs)
        _ = [values.cpu().numpy() for values in cusadi_function.outputs_sparse]
        torch.cuda.synchronize()
        transfer_samples.append((time.perf_counter_ns() - start) / 1e6)

    return median(kernel_samples), median(resident_samples), median(transfer_samples)


def _error_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    absolute_error = np.abs(candidate - reference)
    relative_error = absolute_error / np.maximum(np.abs(reference), np.finfo(np.float64).eps)
    return float(np.max(absolute_error)), float(np.max(relative_error))


def _benchmark_batch(
    torch: Any,
    dynamics: casadi.Function,
    cusadi_root: Path,
    batch_size: int,
    warmup: int,
    repeats: int,
    seed: int,
    rtol: float,
    atol: float,
) -> BatchResult:
    cpu_inputs = _make_inputs(dynamics, batch_size, seed)
    reference = _casadi_mapped_output(dynamics, cpu_inputs)
    cusadi_function = _load_cusadi_function(cusadi_root, dynamics, batch_size)
    gpu_inputs = [torch.from_numpy(values).to(device="cuda", dtype=torch.float64).contiguous() for values in cpu_inputs]
    cusadi_function.evaluate(gpu_inputs)
    torch.cuda.synchronize()
    candidate = cusadi_function.outputs_sparse[0].detach().cpu().numpy()

    np.testing.assert_allclose(candidate, reference, rtol=rtol, atol=atol)
    max_absolute_error, max_relative_error = _error_metrics(reference, candidate)

    casadi_cpu_ms = _time_casadi_cpu(dynamics, cpu_inputs, warmup, repeats)
    kernel_ms, resident_ms, with_transfers_ms = _time_cusadi_gpu(
        torch,
        cusadi_function,
        cpu_inputs,
        warmup,
        repeats,
    )
    return BatchResult(
        batch_size=batch_size,
        max_absolute_error=max_absolute_error,
        max_relative_error=max_relative_error,
        casadi_cpu_ms=casadi_cpu_ms,
        cusadi_kernel_ms=kernel_ms,
        cusadi_gpu_resident_ms=resident_ms,
        cusadi_with_transfers_ms=with_transfers_ms,
        resident_speedup=casadi_cpu_ms / resident_ms,
        end_to_end_speedup=casadi_cpu_ms / with_transfers_ms,
    )


def _print_results(results: list[BatchResult]) -> None:
    header = (
        f"{'batch':>8} {'max abs err':>13} {'CPU ms':>11} {'kernel ms':>11} "
        f"{'GPU ms':>11} {'GPU+copy ms':>13} {'speedup':>10} {'e2e speedup':>13}"
    )
    print("\n" + header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.batch_size:8d} "
            f"{result.max_absolute_error:13.3e} "
            f"{result.casadi_cpu_ms:11.4f} "
            f"{result.cusadi_kernel_ms:11.4f} "
            f"{result.cusadi_gpu_resident_ms:11.4f} "
            f"{result.cusadi_with_transfers_ms:13.4f} "
            f"{result.resident_speedup:10.2f} "
            f"{result.end_to_end_speedup:13.2f}"
        )


def _write_results(
    output_path: Path,
    args: argparse.Namespace,
    dynamics: casadi.Function,
    used_operations: list[str],
    cusadi_root: Path,
    torch: Any,
    results: list[BatchResult],
) -> None:
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload = {
        "metadata": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "casadi": casadi.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": device.name,
            "gpu_compute_capability": ".".join(map(str, torch.cuda.get_device_capability())),
            "bioptim_revision": _git_revision(Path(__file__).resolve().parents[4]),
            "cusadi_revision": _git_revision(cusadi_root),
        },
        "function": {
            "name": dynamics.name(),
            "inputs": [
                {
                    "name": dynamics.name_in(index),
                    "nonzeros": dynamics.nnz_in(index),
                }
                for index in range(dynamics.n_in())
            ],
            "outputs": [
                {
                    "name": dynamics.name_out(index),
                    "nonzeros": dynamics.nnz_out(index),
                }
                for index in range(dynamics.n_out())
            ],
            "instructions": dynamics.n_instructions(),
            "work_size": dynamics.sz_w(),
            "operations": used_operations,
        },
        "configuration": {
            "batch_sizes": args.batch_sizes,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "rtol": args.rtol,
            "atol": args.atol,
        },
        "results": [asdict(result) for result in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote benchmark results to {output_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cusadi-root",
        type=Path,
        default=os.environ.get("CUSADI_ROOT"),
        required=os.environ.get("CUSADI_ROOT") is None,
        help="Path to a CusADi checkout (or set CUSADI_ROOT)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_parse_batch_sizes,
        default=_parse_batch_sizes("1,32,256,2048,16384"),
        help="Comma-separated number of independent dynamics evaluations",
    )
    parser.add_argument("--warmup", type=_positive_int, default=5, help="Warm-up calls per batch size")
    parser.add_argument("--repeats", type=_positive_int, default=20, help="Timed calls per batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rtol", type=float, default=1e-9, help="Relative numerical tolerance")
    parser.add_argument("--atol", type=float, default=1e-9, help="Absolute numerical tolerance")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cusadi_benchmark_results.json"),
        help="JSON result path",
    )
    parser.add_argument(
        "--codegen-only",
        action="store_true",
        help="Stop after compatibility checking and CUDA source generation; no GPU is required",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    cusadi_root = args.cusadi_root.expanduser().resolve()
    dynamics = _build_pendulum_dynamics()
    cuda_source, used_operations = _generate_cuda_source(dynamics, cusadi_root)

    print(f"CasADi function: {dynamics.name()}")
    print(f"Instructions: {dynamics.n_instructions()}")
    print(f"Operations supported by CusADi: {', '.join(used_operations)}")
    print(f"Generated CUDA source: {cuda_source}")
    if args.codegen_only:
        return

    if platform.system() != "Linux":
        raise RuntimeError("The GPU benchmark requires Linux; use --codegen-only on other platforms")
    torch = _import_torch()
    capability = torch.cuda.get_device_capability()
    cuda_architecture = capability[0] * 10 + capability[1]
    _compile_cuda_source(cusadi_root, dynamics.name(), cuda_architecture)

    results = [
        _benchmark_batch(
            torch=torch,
            dynamics=dynamics,
            cusadi_root=cusadi_root,
            batch_size=batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
            rtol=args.rtol,
            atol=args.atol,
        )
        for batch_size in args.batch_sizes
    ]
    _print_results(results)
    _write_results(
        output_path=args.output,
        args=args,
        dynamics=dynamics,
        used_operations=used_operations,
        cusadi_root=cusadi_root,
        torch=torch,
        results=results,
    )


if __name__ == "__main__":
    main()
