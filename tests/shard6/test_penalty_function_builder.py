from types import SimpleNamespace

import pytest

from bioptim.limits.penalty_function_builder import (
    DerivativePenaltyFunctionBuilder,
    PenaltyFunctionBuilder,
    PointwisePenaltyFunctionBuilder,
    TrapezoidalPenaltyFunctionBuilder,
)
from bioptim.misc.enums import QuadratureRule


@pytest.mark.parametrize(
    "integration_rule,derivative,expected_builder",
    [
        (QuadratureRule.DEFAULT, False, PointwisePenaltyFunctionBuilder),
        (QuadratureRule.DEFAULT, True, DerivativePenaltyFunctionBuilder),
        (QuadratureRule.APPROXIMATE_TRAPEZOIDAL, False, TrapezoidalPenaltyFunctionBuilder),
        (QuadratureRule.TRAPEZOIDAL, True, TrapezoidalPenaltyFunctionBuilder),
    ],
)
def test_penalty_function_builder_selects_the_historical_symbolic_mode(
    monkeypatch, integration_rule, derivative, expected_builder
):
    result = object()
    penalty = SimpleNamespace(integration_rule=integration_rule, derivative=derivative)

    def build_for(current_builder):
        return lambda *unused: result if current_builder is expected_builder else None

    for builder in (
        PointwisePenaltyFunctionBuilder,
        DerivativePenaltyFunctionBuilder,
        TrapezoidalPenaltyFunctionBuilder,
    ):
        monkeypatch.setattr(builder, "build", staticmethod(build_for(builder)))

    assert PenaltyFunctionBuilder.build(penalty, object(), object(), object()) is result
