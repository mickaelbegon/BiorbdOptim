import subprocess
import sys
import textwrap

import numpy as np
import pytest

cas = pytest.importorskip("casadi")

if not getattr(cas, "has_nlpsol", lambda _: False)("madnlp"):
    pytest.skip("CasADi MadNLP plugin is not available", allow_module_level=True)

from bioptim import Solver
from bioptim.examples.getting_started.basic_ocp import prepare_ocp
from bioptim.examples.utils import ExampleUtils


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


def test_madnlp_linear_solver_backends():
    """Verify that libMad selects the requested backend instead of falling back to MUMPS."""
    script = textwrap.dedent(
        """
        import casadi as cas
        from bioptim import Solver

        x = cas.MX.sym("x", 2)
        nlp = {"x": x, "f": (x[0] - 1) ** 2 + (x[1] - 2) ** 2, "g": x[0] + x[1]}
        for index, linear_solver in enumerate(("mumps", "umfpack", "lapack_cpu")):
            options = Solver.MADNLP()
            options.set_convergence_tolerance(1e-8)
            options.set_print_level("INFO")
            options.set_linear_solver(linear_solver)
            solver = cas.nlpsol(
                f"madnlp_backend_{index}",
                "madnlp",
                nlp,
                options.as_dict(type("I", (), {"options_common": {}})()),
            )
            result = solver(x0=[0, 0], lbg=3, ubg=3)
            assert solver.stats()["success"]
            assert abs(float(result["f"])) <= 1e-10
        """
    )
    process = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    output = process.stdout + process.stderr

    assert process.returncode == 0, output
    assert "running with MUMPS" in output
    assert "running with umfpack" in output
    assert "running with Lapack-CPU" in output


def test_madnlp_solves_bioptim_pendulum_ocp():
    ocp = prepare_ocp(
        ExampleUtils.folder + "/models/pendulum.bioMod",
        final_time=1,
        n_shooting=20,
        n_threads=1,
    )
    solver = Solver.MADNLP()
    solver.set_convergence_tolerance(1e-6)
    solver.set_constraint_tolerance(1e-6)
    solver.set_maximum_iterations(500)
    solver.set_print_level("ERROR")

    solution = ocp.solve(solver)

    assert solution.status == 0
    # MadNLP's converged local solution varies by a few ppm across supported platforms.
    assert float(solution.cost) == pytest.approx(68.046875234144, rel=1e-5)
    assert np.max(np.abs(np.asarray(solution.constraints))) <= 1e-8
    assert solution.iterations > 0
    assert solution.inf_pr <= 1e-8
    assert solution.inf_du <= 1e-8
