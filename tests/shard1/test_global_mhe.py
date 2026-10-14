"""
Test for file IO
"""

from types import SimpleNamespace

from bioptim import Solver, PhaseDynamics, SolutionMerge
from bioptim.misc.enums import SolverType
import numpy as np
import numpy.testing as npt
import pytest

from ..utils import TestUtils


@pytest.mark.parametrize("solver_type", [SolverType.IPOPT, SolverType.ALPAQA])
def test_cyclic_nmpc_primal_warm_start_backend_contract(solver_type):
    from bioptim.examples.toy_examples.moving_horizon_estimation import cyclic_nmpc as ocp_module

    nmpc = ocp_module.prepare_nmpc(
        model_path=TestUtils.bioptim_folder() + "/examples/models/arm2.bioMod",
        cycle_len=20,
        cycle_duration=1,
        max_torque=50,
        expand_dynamics=True,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
    )
    nmpc.ocp_solver = SimpleNamespace(opts=SimpleNamespace(type=solver_type))
    nmpc.nlp[0].x_bounds["q"][0, 0] = -np.pi
    states = {"q": np.arange(63).reshape(3, 21) / 100, "qdot": np.zeros((3, 21))}

    nmpc.advance_window_initial_guess_states(SimpleNamespace(decision_states=lambda **_: states))

    expected_q = states["q"].copy()
    if solver_type == SolverType.ALPAQA:
        expected_q[0, 0] = -np.pi
    npt.assert_array_equal(nmpc.nlp[0].x_init["q"].init, expected_q)
    npt.assert_array_equal(nmpc.nlp[0].x_init["qdot"].init, states["qdot"])


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_cyclic_nmpc(phase_dynamics):
    def update_functions(_nmpc, cycle_idx, _sol):
        return cycle_idx < n_cycles  # True if there are still some cycle to perform

    from bioptim.examples.toy_examples.moving_horizon_estimation import cyclic_nmpc as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    n_cycles = 3
    cycle_len = 20
    nmpc = ocp_module.prepare_nmpc(
        model_path=bioptim_folder + "/examples/models/arm2.bioMod",
        cycle_len=cycle_len,
        cycle_duration=1,
        max_torque=50,
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
    )
    sol = nmpc.solve(update_functions, solver=Solver.IPOPT())

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_equal(q.shape, (3, n_cycles * cycle_len + 1))
    npt.assert_almost_equal(q[:, 0], np.array((-3.14159265, 0.12979378, 2.77623291)))
    npt.assert_almost_equal(q[:, -1], np.array((3.14159265, 0.12979378, 2.77623291)))

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((6.28268908, -11.63289399, 0.37215021)))
    npt.assert_almost_equal(qdot[:, -1], np.array((6.28268908, -12.14519356, -0.21986407)))

    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((0.01984925, 17.53758229, -1.92204945)))
    npt.assert_almost_equal(tau[:, -1], np.array((-0.01984925, -6.81104298, -1.80560018)))
