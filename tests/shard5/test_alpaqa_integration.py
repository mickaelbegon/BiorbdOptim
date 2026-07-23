import casadi as cas
import numpy as np
import pytest

from bioptim import Solver
from bioptim.interfaces.alpaqa_interface import AlpaqaInterface, alpaqa_plugin_available

ALPAQA_AVAILABLE = alpaqa_plugin_available()


def test_missing_alpaqa_plugin_is_reported_at_solve_time():
    if ALPAQA_AVAILABLE:
        pytest.skip("The installed CasADi build includes alpaqa")
    interface = AlpaqaInterface(ocp=None)
    with pytest.raises(RuntimeError, match="alpaqa plugin is not available"):
        interface.solve()


@pytest.mark.skipif(
    not ALPAQA_AVAILABLE, reason="CasADi was not compiled with alpaqa support"
)
def test_minimal_alpaqa_nlp():
    xy = cas.MX.sym("xy", 2)
    nlp = {"x": xy, "f": (xy[0] - 1) ** 2 + (xy[1] - 2) ** 2, "g": xy[0] + xy[1]}
    options = Solver.ALPAQA().as_dict(type("Interface", (), {"options_common": {}})())
    solver = cas.nlpsol("solver", "alpaqa", nlp, options)
    result = solver(x0=[0, 0], lbx=[-10, -10], ubx=[10, 10], lbg=3, ubg=3)
    np.testing.assert_allclose(np.asarray(result["x"]).reshape(-1), [1, 2], atol=1e-5)
    assert float(result["f"]) == pytest.approx(0.0, abs=1e-10)
    assert float(result["g"]) == pytest.approx(3.0, abs=1e-6)
    stats = solver.stats()
    assert stats["success"] is True
    assert stats["unified_return_status"] == "SOLVER_RET_SUCCESS"
    assert "t_wall_total" in stats
