import pytest
from casadi import SX, MX, vertcat, Function
import numpy as np

from bioptim import NonLinearProgram, PhaseDynamics
from bioptim.interfaces.solver_interface import SolverInterface


@pytest.fixture
def nlp_sx():
    # Create a dummy NonLinearProgram object with necessary attributes
    nlp = NonLinearProgram(None)
    nlp.X = [SX([[1], [2], [3]])]
    nlp.X_scaled = [SX([[4], [5], [6]])]
    # Add more attributes as needed
    return nlp


@pytest.fixture
def nlp_mx():
    # Create a dummy NonLinearProgram object with necessary attributes
    nlp = NonLinearProgram(PhaseDynamics.SHARED_DURING_THE_PHASE, use_sx=False)
    nlp.X = [MX(np.array([[1], [2], [3]]))]
    nlp.X_scaled = [MX(np.array([[4], [5], [6]]))]
    # Add more attributes as needed
    return nlp


@pytest.fixture
def nlp_control_sx():
    nlp = NonLinearProgram(PhaseDynamics.SHARED_DURING_THE_PHASE)
    nlp.U_scaled = [SX([[1], [2], [3]])]
    return nlp


@pytest.fixture
def nlp_control_mx():
    nlp = NonLinearProgram(PhaseDynamics.SHARED_DURING_THE_PHASE, use_sx=False)
    nlp.U_scaled = [MX(np.array([[1], [2], [3]]))]
    return nlp


def test_one_shot_initial_guess_override_is_consumed_once():
    interface = SolverInterface(ocp=None)
    default = np.zeros((3, 1))

    interface.set_next_initial_guess_override([1.0, 2.0, 3.0])

    np.testing.assert_array_equal(
        interface.consume_initial_guess_override(default),
        np.array([[1.0], [2.0], [3.0]]),
    )
    assert interface.consume_initial_guess_override(default) is default


def test_one_shot_initial_guess_override_validates_values_and_dimension():
    interface = SolverInterface(ocp=None)

    with pytest.raises(ValueError, match="finite and non-empty"):
        interface.set_next_initial_guess_override([np.nan])
    interface.set_next_initial_guess_override([1.0, 2.0])
    with pytest.raises(ValueError, match="expected 3"):
        interface.consume_initial_guess_override(np.zeros((3, 1)))
