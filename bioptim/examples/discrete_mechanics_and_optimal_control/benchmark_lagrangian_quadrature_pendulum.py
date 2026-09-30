"""
Benchmark the quadrature rules used to build a pendulum discrete Lagrangian.

The benchmark deliberately evaluates the action on an affine configuration segment.
This is the path assumed by the existing endpoint-based discrete Lagrangians, for
which the velocity is ``(q2 - q1) / h``.  Its action has an analytical expression,
so the reported errors isolate the quadrature error instead of the error induced by
the discrete trajectory itself.

``gauss_legendre_2`` reproduces the formula selected for
``QuadratureRule.GAUSS_LEGENDRE_2`` without importing the public enum.  The script
also reports Simpson's rule to make the fourth-order alternatives easy to compare.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class QuadratureBenchmarkResult:
    """One quadrature-rule measurement for a pendulum action segment."""

    rule: str
    time_step: float
    discrete_lagrangian: float
    reference_discrete_lagrangian: float
    absolute_error: float
    relative_error: float


def pendulum_lagrangian(
    q: float | np.ndarray,
    qdot: float,
    mass: float = 1.0,
    length: float = 1.0,
    gravity: float = 9.81,
) -> float | np.ndarray:
    """Return ``T - V`` for a simple pendulum with its zero potential at ``q = 0``."""
    return 0.5 * mass * length**2 * qdot**2 - mass * gravity * length * (1 - np.cos(q))


def exact_affine_discrete_lagrangian(
    q1: float,
    q2: float,
    time_step: float,
    mass: float = 1.0,
    length: float = 1.0,
    gravity: float = 9.81,
) -> float:
    """Return the exact action along the affine segment from ``q1`` to ``q2``.

    The stable ``sinc`` form includes the degenerate case ``q1 == q2``.  This is
    the high-precision reference for every rule in this example.
    """
    qdot = (q2 - q1) / time_step
    mean_cosine = np.cos((q1 + q2) / 2) * np.sinc((q2 - q1) / (2 * np.pi))
    kinetic = 0.5 * mass * length**2 * qdot**2
    mean_potential = mass * gravity * length * (1 - mean_cosine)
    return float(time_step * (kinetic - mean_potential))


def rectangle_left(q1: float, q2: float, time_step: float) -> float:
    """Approximate the action with the left endpoint rule."""
    return float(time_step * pendulum_lagrangian(q1, (q2 - q1) / time_step))


def rectangle_right(q1: float, q2: float, time_step: float) -> float:
    """Approximate the action with the right endpoint rule."""
    return float(time_step * pendulum_lagrangian(q2, (q2 - q1) / time_step))


def midpoint(q1: float, q2: float, time_step: float) -> float:
    """Approximate the action with the midpoint rule."""
    return float(time_step * pendulum_lagrangian((q1 + q2) / 2, (q2 - q1) / time_step))


def trapezoidal(q1: float, q2: float, time_step: float) -> float:
    """Approximate the action with the endpoint trapezoidal rule."""
    qdot = (q2 - q1) / time_step
    return float(time_step / 2 * (pendulum_lagrangian(q1, qdot) + pendulum_lagrangian(q2, qdot)))


def simpson(q1: float, q2: float, time_step: float) -> float:
    """Approximate the action with Simpson's rule (three Lagrangian evaluations)."""
    qdot = (q2 - q1) / time_step
    q_mid = (q1 + q2) / 2
    return float(
        time_step
        / 6
        * (pendulum_lagrangian(q1, qdot) + 4 * pendulum_lagrangian(q_mid, qdot) + pendulum_lagrangian(q2, qdot))
    )


def gauss_legendre_2(q1: float, q2: float, time_step: float) -> float:
    """Approximate the action with two interior Gauss--Legendre evaluations.

    The two nodes are ``1/2 +/- sqrt(3)/6`` of the segment.  Like Simpson's
    rule, this is a fourth-order quadrature rule for a smooth integrand, but it
    requires only two Lagrangian evaluations.
    """
    qdot = (q2 - q1) / time_step
    nodes = 0.5 + np.array((-np.sqrt(3) / 6, np.sqrt(3) / 6))
    q_nodes = q1 + nodes * (q2 - q1)
    return float(time_step / 2 * np.sum(pendulum_lagrangian(q_nodes, qdot)))


RULES: dict[str, Callable[[float, float, float], float]] = {
    "rectangle_left": rectangle_left,
    "rectangle_right": rectangle_right,
    "midpoint": midpoint,
    "trapezoidal": trapezoidal,
    "simpson": simpson,
    "gauss_legendre_2": gauss_legendre_2,
}


def run_benchmark(
    time_steps: tuple[float, ...] = (0.4, 0.2, 0.1, 0.05),
    initial_angle: float = 0.2,
    angular_velocity: float = 2.4,
) -> list[QuadratureBenchmarkResult]:
    """Benchmark all rules on increasingly short affine pendulum segments.

    Keeping the angular velocity constant while reducing ``h`` means that
    ``q2 - q1 = O(h)``.  The displayed local action errors consequently have
    observed exponents close to 2 for endpoint rectangles, 3 for
    midpoint/trapezoidal, and 5 for Simpson/Gauss--Legendre 2.  These are one
    order higher than the usual quadrature-rule labels because the interval itself
    is being refined.
    """
    results = []
    for time_step in time_steps:
        q1 = initial_angle
        q2 = initial_angle + angular_velocity * time_step
        reference = exact_affine_discrete_lagrangian(q1, q2, time_step)
        for rule_name, rule in RULES.items():
            approximation = rule(q1, q2, time_step)
            absolute_error = abs(approximation - reference)
            results.append(
                QuadratureBenchmarkResult(
                    rule=rule_name,
                    time_step=time_step,
                    discrete_lagrangian=approximation,
                    reference_discrete_lagrangian=reference,
                    absolute_error=absolute_error,
                    relative_error=absolute_error / max(abs(reference), np.finfo(float).eps),
                )
            )
    return results


def observed_convergence_rates(
    results: list[QuadratureBenchmarkResult],
) -> dict[tuple[str, float], float]:
    """Return the local order observed when refining from ``h`` to the next smaller step.

    The returned rate is associated with the smaller time step and therefore can be
    displayed next to the corresponding benchmark result.  No rate is returned for
    the largest time step of each rule.
    """
    rates = {}
    for rule_name in RULES:
        rule_results = sorted(
            (result for result in results if result.rule == rule_name),
            key=lambda result: -result.time_step,
        )
        for coarse, fine in zip(rule_results[:-1], rule_results[1:]):
            rates[(rule_name, fine.time_step)] = np.log(coarse.absolute_error / fine.absolute_error) / np.log(
                coarse.time_step / fine.time_step
            )
    return rates


def format_results(results: list[QuadratureBenchmarkResult]) -> str:
    """Format benchmark results as a compact, copyable table."""
    rates = observed_convergence_rates(results)
    header = "rule                 h       Ld approximation    abs. error      rel. error   observed order"
    rows = [header, "-" * len(header)]
    for result in results:
        rate = rates.get((result.rule, result.time_step))
        rate_text = "-" if rate is None else f"{rate:.6f}"
        rows.append(
            f"{result.rule:20} {result.time_step:5.3f} {result.discrete_lagrangian:17.10e} "
            f"{result.absolute_error:13.6e} {result.relative_error:13.6e} {rate_text:>16}"
        )
    return "\n".join(rows)


def main() -> None:
    """Run the benchmark from the command line."""
    print(format_results(run_benchmark()))


if __name__ == "__main__":
    main()
