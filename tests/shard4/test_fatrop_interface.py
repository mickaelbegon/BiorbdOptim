from types import SimpleNamespace

from casadi import DM
import numpy as np
import numpy.testing as npt

from bioptim.interfaces.fatrop_interface import FatropInterface


def test_fatrop_solver_call_limits_compensate_relative_bound_relaxation():
    interface = object.__new__(FatropInterface)
    interface.opts = SimpleNamespace(bound_tightening_factor=1e-8)
    interface.limits = {
        "lbx": DM([7035.752, -np.inf, 2.0]),
        "ubx": DM([7035.762, 4.0, 2.0]),
        "x0": DM([7035.762, 5.0, 2.0]),
        "lbg": DM.zeros(0, 1),
        "ubg": DM.zeros(0, 1),
    }

    call_limits = interface.solver_call_limits()

    npt.assert_allclose(
        np.asarray(call_limits["lbx"]).reshape(-1),
        [7035.752 + 7035.752e-8, -np.inf, 2.0],
    )
    npt.assert_allclose(
        np.asarray(call_limits["ubx"]).reshape(-1),
        [7035.762 - 7035.762e-8, 4.0 - 4e-8, 2.0],
    )
    npt.assert_allclose(
        np.asarray(call_limits["x0"]).reshape(-1),
        np.asarray(call_limits["ubx"]).reshape(-1),
    )
    npt.assert_allclose(
        np.asarray(interface.limits["ubx"]).reshape(-1),
        [7035.762, 4.0, 2.0],
    )
