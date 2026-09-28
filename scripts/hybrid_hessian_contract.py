"""Reusable, solver-free validation for an IPOPT Lagrangian-Hessian callback.

This module is deliberately independent from the implementation of ``H_other``
or the compiled continuity scatter.  It guards their public boundary: a
candidate must be indistinguishable from CasADi's native ``nlp_hess_l`` in ABI,
sparsity and numeric response to both ``sigma`` and ``lambda_g``.
"""

from __future__ import annotations

import statistics
import time

import casadi as ca
import numpy as np


def _same_sparsity(left, right) -> bool:
    return left == right and left.shape == right.shape and left.nnz() == right.nnz()


def check_abi(reference: ca.Function, candidate: ca.Function) -> dict:
    """Compare all details IPOPT uses to install a replacement callback."""

    compatible = reference.n_in() == candidate.n_in() and reference.n_out() == candidate.n_out()
    inputs = []
    if compatible:
        for index in range(reference.n_in()):
            inputs.append(
                {
                    "index": index,
                    "name_equal": reference.name_in(index) == candidate.name_in(index),
                    "sparsity_equal": _same_sparsity(reference.sparsity_in(index), candidate.sparsity_in(index)),
                }
            )
    outputs = []
    if compatible:
        for index in range(reference.n_out()):
            outputs.append(
                {
                    "index": index,
                    "name_equal": reference.name_out(index) == candidate.name_out(index),
                    "sparsity_equal": _same_sparsity(reference.sparsity_out(index), candidate.sparsity_out(index)),
                }
            )
    passed = compatible and all(item["name_equal"] and item["sparsity_equal"] for item in inputs + outputs)
    return {"passed": passed, "input_count": reference.n_in(), "output_count": reference.n_out(), "inputs": inputs, "outputs": outputs}


def _random_dm(sparsity: ca.Sparsity, rng: np.random.Generator) -> ca.DM:
    """Create a random DM preserving even an empty 0x0 callback input."""

    return ca.DM(sparsity, rng.normal(size=sparsity.nnz()))


def _nonzeros(value) -> np.ndarray:
    return np.asarray(value.nonzeros(), dtype=float).reshape(-1)


def _median_seconds(function: ca.Function, values, repeats: int) -> float:
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(*values)
        elapsed.append(time.perf_counter() - start)
    return statistics.median(elapsed)


def audit(
    reference: ca.Function,
    candidate: ca.Function,
    *,
    selected_lambda_rows: np.ndarray | list[int] | tuple[int, ...],
    samples: int = 3,
    seed: int = 20260928,
    repeats: int = 3,
    tolerance: float = 1e-10,
) -> dict:
    """Audit ABI, Hessian values, sigma response and selected multipliers.

    ``selected_lambda_rows`` are the canonical STATE_CONTINUITY rows supplied
    by the sparse projection plan. They are checked as valid and perturbed
    separately; this catches accidental use of a packet-local multiplier order
    in place of IPOPT's global ``lambda_g`` order.
    """

    abi = check_abi(reference, candidate)
    if not abi["passed"]:
        return {"abi": abi, "passed": False, "reason": "ABI/sparsity mismatch"}
    if reference.n_in() != 4:
        return {"abi": abi, "passed": False, "reason": "expected IPOPT hess_lag four-input ABI"}
    lambda_shape = reference.sparsity_in(3)
    lambda_rows = np.asarray(selected_lambda_rows, dtype=int).reshape(-1)
    if lambda_rows.size == 0 or np.any(lambda_rows < 0) or np.any(lambda_rows >= lambda_shape.numel()):
        return {"abi": abi, "passed": False, "reason": "invalid canonical lambda_g rows"}

    rng = np.random.default_rng(seed)
    maximum_error = 0.0
    multiplier_response_error = 0.0
    rows_checked = np.unique(lambda_rows)
    for _ in range(samples):
        values = [_random_dm(reference.sparsity_in(index), rng) for index in range(reference.n_in())]
        native = _nonzeros(reference(*values))
        actual = _nonzeros(candidate(*values))
        maximum_error = max(maximum_error, float(np.max(np.abs(native - actual), initial=0.0)))

        # Perturb exactly the global IPOPT multipliers used by the projected
        # continuity term. Do not assume contiguity or packet ordering.
        perturbed = list(values)
        lam_values = _nonzeros(values[3]).copy()
        lam_values[rows_checked] += rng.normal(size=rows_checked.size)
        perturbed[3] = ca.DM(reference.sparsity_in(3), lam_values)
        native_delta = _nonzeros(reference(*perturbed)) - native
        actual_delta = _nonzeros(candidate(*perturbed)) - actual
        multiplier_response_error = max(
            multiplier_response_error, float(np.max(np.abs(native_delta - actual_delta), initial=0.0))
        )

    timing_values = [_random_dm(reference.sparsity_in(index), rng) for index in range(reference.n_in())]
    native_s = _median_seconds(reference, timing_values, repeats)
    candidate_s = _median_seconds(candidate, timing_values, repeats)
    return {
        "abi": abi,
        "samples": samples,
        "tolerance": tolerance,
        "max_abs_error": maximum_error,
        "max_abs_selected_lambda_response_error": multiplier_response_error,
        "native_median_s": native_s,
        "candidate_median_s": candidate_s,
        "speedup": native_s / candidate_s if candidate_s else None,
        "passed": maximum_error <= tolerance and multiplier_response_error <= tolerance,
    }
