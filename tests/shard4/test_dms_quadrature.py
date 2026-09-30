from types import SimpleNamespace

from casadi import Function, SX
import numpy as np
import numpy.testing as npt
import pytest

from bioptim import ControlType, OdeSolver, QuadratureRule
from bioptim.dynamics.integrator import RK1, RK2, RK4, RK8
from bioptim.misc.enums import DefectType

from .test_cost_function_integration import prepare_ocp
from ..utils import TestUtils


def _rk_integrator(integrator_class, rhs, number_of_finite_elements=1, control_type=ControlType.CONSTANT):
    cx = SX
    t_span = cx.sym("t_span", 2, 1)
    states = cx.sym("states", 1, 1)
    controls = cx.sym("controls", 1, 2 if control_type == ControlType.LINEAR_CONTINUOUS else 1)
    parameters = cx.sym("parameters", 0, 1)
    algebraic_states = cx.sym("algebraic_states", 0, 1)
    numerical_timeseries = cx.sym("numerical_timeseries", 0, 1)
    dynamics = Function(
        "dynamics",
        [t_span, states, controls, parameters, algebraic_states, numerical_timeseries],
        [rhs(t_span, states, controls)],
        ["t_span", "x", "u", "p", "a", "d"],
        ["xdot"],
    )
    return integrator_class(
        {
            "t": t_span,
            "x": states,
            "u": controls,
            "param": parameters,
            "a": algebraic_states,
            "d": numerical_timeseries,
            "ode": dynamics,
            "implicit_ode": None,
        },
        {
            "model": SimpleNamespace(nb_quaternions=0),
            "ode_index": 0,
            "cx": cx,
            "defects_type": DefectType.NOT_APPLICABLE,
            "control_type": control_type,
            "duplicate_starting_point": False,
            "number_of_finite_elements": number_of_finite_elements,
        },
    )


@pytest.mark.parametrize(
    "integrator_class, expected",
    [(RK1, 0.0), (RK2, 0.25), (RK4, 1 / 3)],
)
def test_dms_quadrature_uses_integrator_stages_for_state_cost(integrator_class, expected):
    """The Lagrange quadrature follows the RK stages for xdot=1 and x(0)=0."""

    integrator = _rk_integrator(integrator_class, lambda _, __, ___: 1)
    stages = integrator.quadrature_stages(
        SX.zeros(1, 1), SX.zeros(1, 1), SX.zeros(0, 1), SX.zeros(0, 1), SX.zeros(0, 1)
    )
    quadrature = Function(
        "quadrature",
        [integrator.t_span_sym],
        [sum(weight * state**2 for _, state, _, weight in stages)],
    )

    npt.assert_allclose(float(quadrature([0, 1])), expected)


def test_dms_quadrature_uses_linear_controls_and_interpolates_targets():
    """The public INTEGRATOR rule evaluates the RK4 stages and linearly interpolated targets."""

    bioptim_folder = TestUtils.bioptim_folder()
    target = np.array([[0.0, 2.0, 4.0], [0.0, 0.0, 0.0]])
    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
        n_shooting=2,
        integration_rule=QuadratureRule.INTEGRATOR,
        objective="torque",
        control_type=ControlType.LINEAR_CONTINUOUS,
        target=target,
        ode_solver=OdeSolver.RK4(n_integration_steps=1),
    )

    weighted_function = ocp.nlp[0].J[0].weighted_function[0]
    x = np.zeros((weighted_function.size1_in("x"), 1))
    u = np.array([[1.0], [0.0], [2.0], [0.0]])
    actual = np.asarray(
        weighted_function(0.0, 1.0, x, u, [], [], [], np.ones((2, 1)), target[:, :2])
    ).reshape(-1)

    npt.assert_allclose(actual, [1 / 3, 0.0])


def test_dms_quadrature_rejects_rk8_until_its_tableau_fix_is_integrated():
    integrator = _rk_integrator(RK8, lambda _, __, ___: 0)

    with pytest.raises(NotImplementedError, match="RK8"):
        integrator.quadrature_stages(
            SX.zeros(1, 1), SX.zeros(1, 1), SX.zeros(0, 1), SX.zeros(0, 1), SX.zeros(0, 1)
        )
