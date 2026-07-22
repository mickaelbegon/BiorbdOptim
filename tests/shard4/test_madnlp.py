import numpy as np
import pytest

cas = pytest.importorskip("casadi")

if not getattr(cas, "has_nlpsol", lambda _: False)("madnlp"):
    pytest.skip("CasADi MadNLP plugin is not available", allow_module_level=True)

from bioptim import Solver


def test_madnlp_casadi_plugin_and_warm_start_inputs():
    """Exercise the option shape and standard nlpsol multiplier inputs."""
    x = cas.MX.sym("x", 2)
    nlp = {"x": x, "f": (x[0] - 1) ** 2 + (x[1] - 2) ** 2, "g": x[0] + x[1]}
    options = Solver.MADNLP()
    options.set_convergence_tolerance(1e-8)
    options.set_maximum_iterations(100)
    options.set_print_level("ERROR")
    solver = cas.nlpsol(
        "madnlp_test",
        "madnlp",
        nlp,
        options.as_dict(type("I", (), {"options_common": {}})()),
    )

    limits = {"x0": [0, 0], "lbx": [-10, -10], "ubx": [10, 10], "lbg": 3, "ubg": 3}
    first = solver(**limits)
    second = solver(**limits, lam_x0=first["lam_x"], lam_g0=first["lam_g"])
    assert solver.stats()["success"]
    np.testing.assert_allclose(np.array(second["x"]).squeeze(), [1, 2], atol=1e-6)
    assert float(second["f"]) == pytest.approx(0.0, abs=1e-10)
    assert np.array(second["g"]).size == 1
    assert np.array(second["lam_x"]).size == 2
    assert np.array(second["lam_g"]).size == 1
