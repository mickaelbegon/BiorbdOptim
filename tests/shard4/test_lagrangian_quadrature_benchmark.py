"""Regression checks for the self-contained pendulum quadrature benchmark."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _benchmark_module():
    benchmark_path = (
        Path(__file__).parents[2]
        / "bioptim/examples/discrete_mechanics_and_optimal_control/benchmark_lagrangian_quadrature_pendulum.py"
    )
    module_name = "lagrangian_quadrature_benchmark"
    spec = importlib.util.spec_from_file_location(module_name, benchmark_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_affine_discrete_lagrangian_matches_high_order_quadrature():
    """The analytical reference must agree with independent high-order integration."""
    benchmark = _benchmark_module()
    q1, q2, time_step = 0.2, 1.16, 0.4
    nodes, weights = np.polynomial.legendre.leggauss(32)
    q_nodes = q1 + (nodes + 1) / 2 * (q2 - q1)
    numerical_reference = time_step / 2 * np.dot(weights, benchmark.pendulum_lagrangian(q_nodes, (q2 - q1) / time_step))

    np.testing.assert_allclose(
        benchmark.exact_affine_discrete_lagrangian(q1, q2, time_step),
        numerical_reference,
        atol=1e-14,
        rtol=1e-14,
    )


def test_gauss_legendre_2_has_fifth_local_error_order_and_outperforms_current_rules():
    """The two-node candidate should outperform midpoint and trapezoidal rules."""
    benchmark = _benchmark_module()
    results = benchmark.run_benchmark(time_steps=(0.2, 0.1, 0.05))
    errors = {rule: [result.absolute_error for result in results if result.rule == rule] for rule in benchmark.RULES}

    assert errors["gauss_legendre_2"][-1] < errors["midpoint"][-1]
    assert errors["gauss_legendre_2"][-1] < errors["trapezoidal"][-1]
    assert errors["gauss_legendre_2"][-1] < errors["simpson"][-1]

    observed_orders = benchmark.observed_convergence_rates(results)
    gauss_legendre_orders = [observed_orders[("gauss_legendre_2", time_step)] for time_step in (0.1, 0.05)]
    np.testing.assert_array_less(4.5, gauss_legendre_orders)
