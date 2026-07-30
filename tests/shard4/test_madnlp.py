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


def test_madnlp_compiled_casadi_plugin(tmp_path, monkeypatch):
    """Generate the NLP evaluator once and solve it through CasADi's importer."""
    monkeypatch.chdir(tmp_path)
    x = cas.SX.sym("x", 2)
    nlp = {
        "x": x,
        "f": (x[0] - 1) ** 2 + (x[1] - 2) ** 2,
        "g": x[0] + x[1],
    }
    options = Solver.MADNLP()
    options.set_convergence_tolerance(1e-8)
    options.set_maximum_iterations(100)
    options.set_print_level("ERROR")
    options.set_c_compile(True)
    solver_options = options.as_dict(
        type("I", (), {"options_common": {}})()
    )

    cas.nlpsol("madnlp_codegen", "madnlp", nlp, solver_options).generate_dependencies(
        "madnlp_nlp.c"
    )
    solver = cas.nlpsol(
        "madnlp_compiled",
        "madnlp",
        cas.Importer("madnlp_nlp.c", "shell"),
        solver_options,
    )
    result = solver(
        x0=[0, 0],
        lbx=[-10, -10],
        ubx=[10, 10],
        lbg=3,
        ubg=3,
    )

    assert solver.stats()["success"]
    np.testing.assert_allclose(np.array(result["x"]).squeeze(), [1, 2], atol=1e-6)
    assert float(result["f"]) == pytest.approx(0.0, abs=1e-10)


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


def test_madnlp_solves_compiled_bioptim_pendulum_ocp(tmp_path, monkeypatch):
    """Exercise Bioptim's generated-oracle path with the MadNLP plugin."""
    monkeypatch.chdir(tmp_path)
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
    solver.set_c_compile(True)

    solution = ocp.solve(solver)

    assert solution.status == 0
    assert float(solution.cost) == pytest.approx(68.046875234144, rel=1e-5)
    assert np.max(np.abs(np.asarray(solution.constraints))) <= 1e-8
    assert solution.iterations > 0
    assert solution.inf_pr <= 1e-8
    assert solution.inf_du <= 1e-8
    assert (tmp_path / "nlp.c").is_file()
