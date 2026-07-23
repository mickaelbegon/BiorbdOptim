from types import SimpleNamespace

import numpy as np
import pytest

from bioptim import Solver
from bioptim.interfaces.alpaqa_interface import maximum_bound_violation
from bioptim.misc.enums import SolverType


def test_alpaqa_public_api_and_nested_options():
    solver = Solver.ALPAQA()
    solver.set_convergence_tolerance(1e-6)
    solver.set_constraint_tolerance(1e-5)
    solver.set_maximum_iterations(500)
    solver.set_alm_maximum_iterations(50)
    solver.set_lbfgs_memory(20)
    solver.set_print_level(1)

    assert solver.type == SolverType.ALPAQA
    assert solver.as_dict(SimpleNamespace(options_common={})) == {
        "alpaqa": {
            "alm": {
                "tolerance": 1e-6,
                "dual_tolerance": 1e-5,
                "max_iter": 50,
                "print_interval": 1,
            },
            "panoc": {"max_iter": 500, "print_interval": 1},
            "lbfgs": {"memory": 20},
        }
    }


def test_alpaqa_advanced_options_and_common_options():
    solver = Solver.ALPAQA()
    solver.set_initial_penalty(10.0)
    solver.set_penalty_update_factor(20.0)
    solver.set_maximum_penalty(1e8)
    solver.set_initial_tolerance(1e-2)
    solver.set_maximum_wall_time(0.02)
    solver.set_option_unsafe(12, "panoc.max_no_progress")

    options = solver.as_dict(SimpleNamespace(options_common={"verbose": True}))
    assert options["alpaqa"]["alm"]["max_time"] == "0.02s"
    assert options["alpaqa"]["panoc"]["max_no_progress"] == 12
    assert options["verbose"] is True


@pytest.mark.parametrize(
    "call, error",
    [
        (lambda s: s.set_option_unsafe(1, "unknown"), ValueError),
        (lambda s: s.set_option_unsafe(1, "alm.unknown"), ValueError),
        (lambda s: s.set_maximum_iterations(-1), ValueError),
        (lambda s: s.set_maximum_iterations(1.5), TypeError),
        (lambda s: s.set_convergence_tolerance("small"), TypeError),
        (lambda s: s.set_lbfgs_memory(0), ValueError),
    ],
)
def test_alpaqa_rejects_invalid_options(call, error):
    with pytest.raises(error):
        call(Solver.ALPAQA())


def test_alpaqa_rejects_unverified_features():
    solver = Solver.ALPAQA()
    with pytest.raises(NotImplementedError, match="C compilation"):
        solver.set_c_compile(True)


def test_maximum_bound_violation():
    assert maximum_bound_violation([1.1, 2.0], [0.0, 2.0], [1.0, 2.0]) == pytest.approx(
        0.1
    )
    assert maximum_bound_violation([1.0], [1.0], [1.0]) == 0.0
    assert maximum_bound_violation([np.nan], [0.0], [2.0]) == float("inf")
    with pytest.raises(ValueError, match="identical dimensions"):
        maximum_bound_violation([1.0], [0.0, 0.0], [2.0, 2.0])
