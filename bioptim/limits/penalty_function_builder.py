from dataclasses import dataclass
from typing import Any

from casadi import Function

from ..misc.enums import ControlType, QuadratureRule


@dataclass(frozen=True)
class PenaltyFunctionBuildContext:
    """Symbolic inputs shared by every penalty construction mode."""

    name: str
    time: Any
    phases_dt: Any
    x: Any
    u: Any
    p: Any
    a: Any
    d: Any
    target: Any
    exponent: int


@dataclass(frozen=True)
class PenaltyFunctionBuildResult:
    """The unweighted CasADi function and its target residual."""

    function: Function
    residual: Any


class PenaltyFunctionBuilder:
    """Select the symbolic construction strategy for a penalty.

    The three construction modes are orthogonal to the penalty nature and to
    custom callbacks: a pointwise function, a temporal derivative, or a
    trapezoidal integral. Their order deliberately matches the legacy logic.
    """

    @staticmethod
    def build(penalty, controller, sub_fcn, context: PenaltyFunctionBuildContext) -> PenaltyFunctionBuildResult:
        if penalty.integration_rule in (QuadratureRule.APPROXIMATE_TRAPEZOIDAL, QuadratureRule.TRAPEZOIDAL):
            return TrapezoidalPenaltyFunctionBuilder.build(penalty, controller, sub_fcn, context)
        if penalty.derivative:
            return DerivativePenaltyFunctionBuilder.build(penalty, controller, sub_fcn, context)
        return PointwisePenaltyFunctionBuilder.build(sub_fcn, context)


class PointwisePenaltyFunctionBuilder:
    """Build the regular pointwise penalty function."""

    @staticmethod
    def build(sub_fcn, context: PenaltyFunctionBuildContext) -> PenaltyFunctionBuildResult:
        function = Function(
            context.name,
            [context.time, context.phases_dt, context.x, context.u, context.p, context.a, context.d],
            [sub_fcn],
            ["t", "dt", "x", "u", "p", "a", "d"],
            ["val"],
        )
        residual = (
            function(context.time, context.phases_dt, context.x, context.u, context.p, context.a, context.d)
            - context.target
        ) ** context.exponent
        return PenaltyFunctionBuildResult(function, residual)


class DerivativePenaltyFunctionBuilder:
    """Build a penalty on the temporal difference between two nodes."""

    @staticmethod
    def build(penalty, controller, sub_fcn, context: PenaltyFunctionBuildContext) -> PenaltyFunctionBuildResult:
        x_start = controller.states_scaled.cx_start
        x_end = controller.states_scaled.cx_end
        u_start = controller.controls_scaled.cx_start
        u_end = (
            controller.controls_scaled.cx_start
            if penalty.control_types[0] in (ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE)
            else controller.controls_scaled.cx_end
        )
        p_start = controller.parameters_scaled.cx
        a_start = controller.algebraic_states_scaled.cx_start
        a_end = controller.algebraic_states_scaled.cx_end
        d_start = controller.numerical_timeseries.cx_start
        d_end = controller.numerical_timeseries.cx_end

        function_at_node = Function(
            context.name,
            [context.time, context.phases_dt, x_start, u_start, p_start, a_start, d_start],
            [sub_fcn],
            ["t", "dt", "x", "u", "p", "a", "d"],
            ["val"],
        )
        function = Function(
            context.name,
            [context.time, context.phases_dt, context.x, context.u, context.p, context.a, context.d],
            [
                function_at_node(context.time, context.phases_dt, x_end, u_end, context.p, a_end, d_end)
                - function_at_node(context.time, context.phases_dt, x_start, u_start, context.p, a_start, d_start)
            ],
            ["t", "dt", "x", "u", "p", "a", "d"],
            ["val"],
        )
        residual = (
            function(context.time, context.phases_dt, context.x, context.u, context.p, context.a, context.d)
            - context.target
        ) ** context.exponent
        return PenaltyFunctionBuildResult(function, residual)


class TrapezoidalPenaltyFunctionBuilder:
    """Build a trapezoidal or approximate-trapezoidal penalty function."""

    @staticmethod
    def build(penalty, controller, sub_fcn, context: PenaltyFunctionBuildContext) -> PenaltyFunctionBuildResult:
        p_start = controller.parameters_scaled.cx
        x_start = controller.states_scaled.cx_start
        a_start = controller.algebraic_states_scaled.cx_start
        a_end = controller.algebraic_states_scaled.cx_end
        d_start = controller.numerical_timeseries.cx_start
        d_end = controller.numerical_timeseries.cx_end

        if penalty.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
            x_end = controller.states_scaled.cx_end
        elif penalty.integration_rule == QuadratureRule.TRAPEZOIDAL:
            u_integrate = context.u.reshape((-1, 2))
            if penalty.control_types[0] in (ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE):
                u_integrate = u_integrate[:, 0]
            elif penalty.control_types[0] not in (ControlType.LINEAR_CONTINUOUS,):
                raise NotImplementedError(f"Control type {penalty.control_types[0]} not implemented yet")

            x_end = controller.integrate(
                t_span=controller.t_span.cx,
                x0=controller.states.cx_start,
                u=u_integrate,
                p=controller.parameters.cx,
                a=controller.algebraic_states.cx_start,
                d=controller.numerical_timeseries.cx_start,
            )["xf"]
        else:
            raise NotImplementedError(f"Integration rule {penalty.integration_rule} not implemented yet")

        u_start = controller.controls_scaled.cx_start
        if penalty.control_types[0] in (ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE):
            u_end = controller.controls_scaled.cx_start
        elif penalty.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
            u_end = controller.controls_scaled.cx_start
        else:
            u_end = controller.controls_scaled.cx_end

        function_at_node = Function(
            context.name,
            [context.time, context.phases_dt, x_start, u_start, p_start, a_start, d_start],
            [sub_fcn],
        )
        value_at_start = function_at_node(
            context.time, context.phases_dt, x_start, u_start, p_start, a_start, d_start
        )
        value_at_end = function_at_node(
            context.time + controller.dt.cx,
            context.phases_dt,
            x_end,
            u_end,
            p_start,
            a_end,
            d_end,
        )
        function = Function(
            context.name,
            [context.time, context.phases_dt, context.x, context.u, context.p, context.a, context.d],
            [(value_at_start + value_at_end) / 2],
            ["t", "dt", "x", "u", "p", "a", "d"],
            ["val"],
        )
        residual = (
            (value_at_start - context.target[:, 0]) ** context.exponent
            + (value_at_end - context.target[:, 1]) ** context.exponent
        ) / 2
        return PenaltyFunctionBuildResult(function, residual)
