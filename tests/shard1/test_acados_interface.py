"""
Test for file IO.
It tests the results of an optimal control problem with acados regarding the proper functioning of :
- the handling of mayer and lagrange obj
"""

import os
import shutil
from sys import platform
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest

from bioptim import (
    TorqueBiorbdModel,
    Axis,
    ObjectiveList,
    ObjectiveFcn,
    OdeSolver,
    ConstraintList,
    ConstraintFcn,
    Node,
    MovingHorizonEstimator,
    DynamicsOptions,
    InterpolationType,
    Solver,
    BoundsList,
    PhaseDynamics,
    SolutionMerge,
)
from tests.utils import TestUtils


def test_acados_v055_diagnostics_snapshot():
    from bioptim.interfaces.acados_interface import _collect_acados_diagnostics

    statistics = np.array(
        [
            [0.0, 1.0],
            [1e-1, 1e-6],
            [2e-1, 2e-6],
            [3e-1, 3e-6],
            [4e-1, 4e-6],
            [0.0, 0.0],
            [2.0, 1.0],
            [0.0, 0.75],
        ]
    )

    class FakeAcadosSolver:
        def __init__(self, statistics_table=statistics):
            self.statistics_table = statistics_table

        def get_stats(self, field):
            values = {
                "statistics": self.statistics_table,
                "nlp_iter": 1,
                "sqp_iter": 1,
                "qpscaling_status": 0,
                "time_tot": 0.12,
                "time_qp": 0.03,
            }
            if field not in values:
                raise ValueError(f"{field} unavailable")
            return values[field]

        def get_residuals(self, recompute=False):
            assert recompute is False
            return np.array([1e-6, 2e-6, 3e-6, 4e-6])

    diagnostics = _collect_acados_diagnostics(
        FakeAcadosSolver(),
        status=4,
        nlp_solver_type="SQP",
        qp_solver="PARTIAL_CONDENSING_HPIPM",
    )

    assert diagnostics["status_label"] == "ACADOS_QP_FAILURE"
    assert diagnostics["successful"] is False
    assert diagnostics["residuals"] == {
        "stationarity": 1e-6,
        "dynamics": 2e-6,
        "inequality": 3e-6,
        "complementarity": 4e-6,
    }
    npt.assert_equal(diagnostics["qp_status"], [0.0, 0.0])
    npt.assert_equal(diagnostics["qp_iterations"], [2.0, 1.0])
    npt.assert_equal(diagnostics["step_sizes"], [0.0, 0.75])
    assert diagnostics["timings"]["time_tot"] == 0.12
    assert diagnostics["timings"]["time_qp"] == 0.03
    assert "time_sim" in diagnostics["unavailable_statistics"]

    feasible_qp_statistics = np.zeros((14, 2))
    feasible_qp_statistics[[5, 7, 9], :] = [[0, 0], [1, 0], [0, 2]]
    feasible_qp_statistics[[6, 8, 10], :] = [[2, 3], [4, 5], [6, 7]]
    feasible_qp_statistics[11, :] = [1.0, 0.5]
    feasible_qp_diagnostics = _collect_acados_diagnostics(
        FakeAcadosSolver(feasible_qp_statistics),
        status=0,
        nlp_solver_type="SQP_WITH_FEASIBLE_QP",
        qp_solver="PARTIAL_CONDENSING_HPIPM",
    )
    npt.assert_equal(feasible_qp_diagnostics["qp_status"], [[0, 1, 0], [0, 0, 2]])
    npt.assert_equal(feasible_qp_diagnostics["qp_iterations_per_solve"], [[2, 4, 6], [3, 5, 7]])
    npt.assert_equal(feasible_qp_diagnostics["qp_iterations"], [12, 15])
    npt.assert_equal(feasible_qp_diagnostics["step_sizes"], [1.0, 0.5])


def test_acados_v055_codegen_configuration(tmp_path, monkeypatch):
    pytest.importorskip("acados_template")
    from acados_template import AcadosOcp

    from bioptim.interfaces.acados_interface import _configure_acados_codegen

    monkeypatch.setenv("ACADOS_SOURCE_DIR", str(tmp_path / "source"))
    acados_root = tmp_path / "installed_acados"
    generated_code = tmp_path / "generated_code"

    solver = Solver.ACADOS()
    solver.set_acados_dir(str(acados_root))
    solver.set_c_generated_code_path(str(generated_code))

    acados_ocp = AcadosOcp()
    _configure_acados_codegen(acados_ocp, solver)

    assert acados_ocp.code_gen_options.acados_include_path == str(acados_root / "include")
    assert acados_ocp.code_gen_options.acados_lib_path == str(acados_root / "lib")
    assert acados_ocp.code_gen_options.code_export_directory == str(generated_code)
    assert acados_ocp.code_gen_options.json_file == str(generated_code / "acados_ocp.json")


def test_acados_v055_runtime_parameter_updates(tmp_path, monkeypatch):
    if platform == "win32":
        return

    from bioptim import OptimalControlProgram

    monkeypatch.chdir(tmp_path)
    bioptim_folder = TestUtils.bioptim_folder()
    model = TorqueBiorbdModel(bioptim_folder + "/examples/models/cube_acados.bioMod")
    n_shooting = 5
    runtime_data = np.arange(4 * (n_shooting + 1), dtype=float).reshape((2, 2, n_shooting + 1))
    dynamics = DynamicsOptions(
        ode_solver=OdeSolver.RK4(),
        expand_dynamics=True,
        numerical_data_timeseries={"runtime_data": runtime_data},
    )

    x_bounds = BoundsList()
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * model.nb_tau, [100] * model.nb_tau
    ocp = OptimalControlProgram(
        model,
        n_shooting,
        1,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        use_sx=True,
    )

    solver = Solver.ACADOS()
    first_solution = ocp.solve(solver)
    assert first_solution.status == 0
    acados_solver = ocp.ocp_solver.ocp_solver
    expected_runtime_data = np.vstack((runtime_data[:, 0, :], runtime_data[:, 1, :]))
    for stage in range(n_shooting + 1):
        npt.assert_equal(acados_solver.get(stage, "p"), expected_runtime_data[:, stage])

    ocp.nlp[0].numerical_data_timeseries["runtime_data"][1, 0, 2] = 42.0
    ocp.nlp[0].numerical_data_timeseries["runtime_data"][0, 1, 4] = -3.0
    with patch.object(acados_solver, "set_params_sparse", wraps=acados_solver.set_params_sparse) as sparse_update:
        second_solution = ocp.solve(solver)

    assert second_solution.status == 0
    assert sparse_update.call_count == 2
    npt.assert_equal(acados_solver.get(2, "p"), [2.0, 42.0, 8.0, 20.0])
    npt.assert_equal(acados_solver.get(4, "p"), [4.0, 16.0, -3.0, 22.0])


def test_acados_v055_runtime_parameter_values_do_not_invalidate_code_reuse(tmp_path, monkeypatch):
    if platform == "win32":
        return

    from bioptim import OptimalControlProgram

    monkeypatch.chdir(tmp_path)
    bioptim_folder = TestUtils.bioptim_folder()
    generated_code = tmp_path / "runtime_parameter_reuse"
    n_shooting = 5

    def solve_with_runtime_data(runtime_data):
        model = TorqueBiorbdModel(bioptim_folder + "/examples/models/cube_acados.bioMod")
        dynamics = DynamicsOptions(
            ode_solver=OdeSolver.RK4(),
            expand_dynamics=True,
            numerical_data_timeseries={"runtime_data": runtime_data},
        )
        x_bounds = BoundsList()
        x_bounds["q"] = model.bounds_from_ranges("q")
        x_bounds["qdot"] = model.bounds_from_ranges("qdot")
        u_bounds = BoundsList()
        u_bounds["tau"] = [-100] * model.nb_tau, [100] * model.nb_tau
        ocp = OptimalControlProgram(
            model,
            n_shooting,
            1,
            dynamics=dynamics,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            use_sx=True,
        )
        solver = Solver.ACADOS()
        solver.set_acados_model_name("runtime_parameter_reuse")
        solver.set_c_generated_code_path(str(generated_code))
        solver.set_check_reuse_possible(True)
        solution = ocp.solve(solver)
        assert solution.status == 0
        return ocp

    first_data = np.zeros((2, 2, n_shooting + 1))
    first_ocp = solve_with_runtime_data(first_data)
    assert first_ocp.ocp_solver.ocp_solver.generated

    second_data = np.arange(4 * (n_shooting + 1), dtype=float).reshape((2, 2, n_shooting + 1))
    reused_ocp = solve_with_runtime_data(second_data)
    assert not reused_ocp.ocp_solver.ocp_solver.generated

    expected_runtime_data = np.vstack((second_data[:, 0, :], second_data[:, 1, :]))
    for stage in range(n_shooting + 1):
        npt.assert_equal(reused_ocp.ocp_solver.ocp_solver.get(stage, "p"), expected_runtime_data[:, stage])


def test_acados_v055_primal_dual_warm_start(tmp_path, monkeypatch):
    if platform == "win32":
        return

    from bioptim.examples.toy_examples.acados import cube as ocp_module

    monkeypatch.chdir(tmp_path)
    bioptim_folder = TestUtils.bioptim_folder()
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube_acados.bioMod",
        n_shooting=5,
        tf=1,
        expand_dynamics=True,
    )
    solver = Solver.ACADOS()

    first_solution = ocp.solve(solver=solver)
    assert first_solution.status == 0
    solver_state = first_solution.solver_state
    assert solver_state["solver"] == "ACADOS"
    assert solver_state["format_version"] == 1
    assert solver_state["n_horizon"] == 5
    assert set(solver_state["iterate"]) == {"x", "u", "pi", "lam", "sl", "su"}

    acados_solver = ocp.ocp_solver.ocp_solver
    expected_iterate = {field: values.copy() for field, values in solver_state["iterate"].items()}
    for field, values in expected_iterate.items():
        assert values.ndim == 1
        assert values.shape[0] == acados_solver.get_dim_flat(field)
        if values.size:
            acados_solver.set_flat(field, np.zeros_like(values))

    # The Solution owns detached arrays, not views into the mutable Acados capsule.
    for field, values in expected_iterate.items():
        npt.assert_equal(first_solution.solver_state["iterate"][field], values)

    iterate_before_solve = {}
    original_solve = acados_solver.solve

    def solve_after_recording_iterate():
        for field in expected_iterate:
            iterate_before_solve[field] = acados_solver.get_flat(field)
        return original_solve()

    with patch.object(acados_solver, "solve", side_effect=solve_after_recording_iterate):
        second_solution = ocp.solve(solver=solver, warm_start=first_solution)

    assert second_solution.status == 0
    for field, values in expected_iterate.items():
        npt.assert_equal(iterate_before_solve[field], values)

    copied_solution = first_solution.copy(skip_data=True)
    assert copied_solution.solver_state is not first_solution.solver_state
    for field, values in expected_iterate.items():
        npt.assert_equal(copied_solution.solver_state["iterate"][field], values)


def test_acados_v055_code_reuse_and_reset(tmp_path, monkeypatch):
    if platform == "win32":
        return

    from bioptim.examples.toy_examples.acados import cube as ocp_module

    monkeypatch.chdir(tmp_path)
    bioptim_folder = TestUtils.bioptim_folder()
    generated_code = tmp_path / "generated_code"

    def solve_ocp():
        ocp = ocp_module.prepare_ocp(
            biorbd_model_path=bioptim_folder + "/examples/models/cube_acados.bioMod",
            n_shooting=10,
            tf=2,
            expand_dynamics=True,
        )
        solver = Solver.ACADOS()
        solver.set_acados_model_name("bioptim_code_reuse")
        solver.set_c_generated_code_path(str(generated_code))
        solver.set_check_reuse_possible(True)
        ocp.solve(solver=solver)
        return ocp, solver

    first_ocp, _ = solve_ocp()
    assert first_ocp.ocp_solver.ocp_solver.generated

    reused_ocp, reused_solver = solve_ocp()
    assert not reused_ocp.ocp_solver.ocp_solver.generated

    reused_solver.set_reset_solver_before_solve(True)
    reset_method = reused_ocp.ocp_solver.ocp_solver.reset
    with patch.object(reused_ocp.ocp_solver.ocp_solver, "reset", wraps=reset_method) as reset_mock:
        reused_ocp.solve(solver=reused_solver)
        reset_mock.assert_called_once_with()


def test_acados_v055_code_reuse_requires_stable_model_name():
    if platform == "win32":
        return

    from bioptim.examples.toy_examples.acados import cube as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube_acados.bioMod",
        n_shooting=10,
        tf=2,
        expand_dynamics=True,
    )
    solver = Solver.ACADOS()
    solver.set_check_reuse_possible(True)

    with pytest.raises(RuntimeError, match="code reuse requires a stable model name"):
        ocp.solve(solver=solver)


@pytest.mark.parametrize("solver_mode", ["ANDERSON", "SQP_WITH_FEASIBLE_QP"])
def test_acados_v055_solver_modes(solver_mode):
    if platform == "win32":
        return

    from bioptim.examples.toy_examples.acados import cube as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube_acados.bioMod",
        n_shooting=10,
        tf=2,
        expand_dynamics=True,
    )
    solver = Solver.ACADOS()

    if solver_mode == "ANDERSON":
        solver.set_with_anderson_acceleration(True)
        solver.set_anderson_activation_threshold(10.0)
    else:
        solver.set_nlp_solver_type("SQP_WITH_FEASIBLE_QP")
        solver.set_byrd_omojokon_slack_relaxation_factor(1.01)

    sol = ocp.solve(solver=solver)
    assert sol.status == 0
    diagnostics = sol.solver_diagnostics
    assert diagnostics["status"] == 0
    assert diagnostics["status_label"] == "ACADOS_SUCCESS"
    assert diagnostics["successful"] is True
    assert diagnostics["nlp_solver_type"] == (
        "SQP_WITH_FEASIBLE_QP" if solver_mode == "SQP_WITH_FEASIBLE_QP" else "SQP"
    )
    assert diagnostics["residuals"] is not None
    assert set(diagnostics["residuals"]) == {
        "stationarity",
        "dynamics",
        "inequality",
        "complementarity",
    }
    assert diagnostics["sqp_iterations"] == sol.iterations
    assert diagnostics["timings"]["time_tot"] == sol.solver_time_to_optimize
    assert diagnostics["raw_statistics"].ndim == 2
    if solver_mode == "SQP_WITH_FEASIBLE_QP":
        assert diagnostics["qp_status"].ndim == 2
        assert diagnostics["qp_iterations_per_solve"].shape[1] == 3

    copied_solution = sol.copy(skip_data=True)
    assert copied_solution.status == sol.status
    assert copied_solution.solver_diagnostics is not diagnostics
    npt.assert_equal(
        copied_solution.solver_diagnostics["raw_statistics"],
        diagnostics["raw_statistics"],
    )

    acados_options = ocp.ocp_solver.acados_ocp.solver_options
    if solver_mode == "ANDERSON":
        assert acados_options.with_anderson_acceleration is True
        assert acados_options.anderson_activation_threshold == 10.0
    else:
        assert acados_options.nlp_solver_type == "SQP_WITH_FEASIBLE_QP"
        assert acados_options.byrd_omojokon_slack_relaxation_factor == 1.01

    os.remove("./acados_ocp.json")
    shutil.rmtree("./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_no_obj(cost_type):
    if platform == "win32":
        return

    from bioptim.examples.toy_examples.acados import cube as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube_acados.bioMod",
        n_shooting=10,
        tf=2,
        expand_dynamics=True,
    )
    solver = Solver.ACADOS()
    solver.set_cost_type(cost_type)
    sol = ocp.solve(solver=solver)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_one_mayer(cost_type):
    if platform == "win32":
        return

    from bioptim.examples.toy_examples.acados import cube as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube_acados.bioMod",
        n_shooting=10,
        tf=2,
        expand_dynamics=True,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=[0], target=np.array([[1.0]]).T)
    ocp.update_objectives(objective_functions)

    solver = Solver.ACADOS()
    solver.set_cost_type(cost_type)
    sol = ocp.solve(solver=solver)

    # Check end state value
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = states["q"]
    npt.assert_almost_equal(q[0, -1], 1.0)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_mayer_first_node(cost_type):
    if platform == "win32":
        return

    from bioptim.examples.toy_examples.acados import cube as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube_acados.bioMod",
        n_shooting=10,
        tf=2,
        expand_dynamics=True,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, node=Node.START, key="q", index=[0], target=np.array([[1.0]]).T
    )
    ocp.update_objectives(objective_functions)

    solver = Solver.ACADOS()
    solver.set_cost_type(cost_type)
    sol = ocp.solve(solver=solver)

    # Check end state value
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = states["q"]
    npt.assert_almost_equal(q[0, 0], 0.999999948505021)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_several_mayer(cost_type):
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.toy_examples.acados import cube as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube_acados.bioMod",
        n_shooting=10,
        tf=2,
        expand_dynamics=True,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=[0, 1], target=np.array([[1.0, 2.0]]).T)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=[2], target=np.array([[3.0]]))
    ocp.update_objectives(objective_functions)

    solver = Solver.ACADOS()
    solver.set_cost_type(cost_type)
    sol = ocp.solve(solver=solver)

    # Check end state value
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = states["q"]
    npt.assert_almost_equal(q[0, -1], 1.0)
    npt.assert_almost_equal(q[1, -1], 2.0)
    npt.assert_almost_equal(q[2, -1], 3.0)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_one_lagrange(cost_type):
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.toy_examples.acados import cube as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    n_shooting = 10
    target = np.expand_dims(np.arange(0, n_shooting + 1), axis=0)
    target[0, -1] = n_shooting - 2
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube_acados.bioMod",
        n_shooting=n_shooting,
        tf=2,
        expand_dynamics=True,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        key="q",
        node=Node.ALL,
        weight=10,
        index=[0],
        target=target,
        multi_thread=False,
    )
    ocp.update_objectives(objective_functions)

    solver = Solver.ACADOS()
    solver.set_cost_type(cost_type)
    sol = ocp.solve(solver=solver)

    # Check end state value
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = states["q"]
    npt.assert_almost_equal(q[0, :], target[0, :].squeeze())

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_one_lagrange_and_one_mayer(cost_type):
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.toy_examples.acados import cube as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    n_shooting = 10
    target = np.expand_dims(np.arange(0, n_shooting + 1), axis=0)
    target[0, -1] = n_shooting - 2
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube_acados.bioMod",
        n_shooting=n_shooting,
        tf=2,
        expand_dynamics=True,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        key="q",
        node=Node.ALL,
        weight=10,
        index=[0],
        target=target,
        multi_thread=False,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=[0], target=target[:, -1:], multi_thread=False
    )
    ocp.update_objectives(objective_functions)

    solver = Solver.ACADOS()
    solver.set_cost_type(cost_type)
    sol = ocp.solve(solver=solver)

    # Check end state value
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = states["q"]
    npt.assert_almost_equal(q[0, :], target[0, :].squeeze(), decimal=6)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_control_lagrange_and_state_mayer(cost_type):
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.toy_examples.acados import cube as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    n_shooting = 10
    target = np.array([[2]])
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube_acados.bioMod",
        n_shooting=n_shooting,
        tf=2,
        expand_dynamics=True,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", multi_thread=False)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=[0], target=target, weight=1000, multi_thread=False
    )
    ocp.update_objectives(objective_functions)

    solver = Solver.ACADOS()
    solver.set_cost_type(cost_type)
    sol = ocp.solve(solver=solver)

    # Check end state value
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = states["q"]
    npt.assert_almost_equal(q[0, -1], target.squeeze())

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_options(cost_type):
    if platform == "win32" or platform == "darwin":
        print("Tests for ACADOS options on Windows and Mac are skipped")
        return

    from bioptim.examples.toy_examples.acados import pendulum as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
        final_time=0.6,
        n_shooting=200,
        expand_dynamics=True,
    )

    tols = [1e-1, 1e1]
    iter = []
    for tol in tols:
        solver = Solver.ACADOS()
        solver.set_cost_type(cost_type)
        solver.set_nlp_solver_tol_stat(tol)
        sol = ocp.solve(solver=solver)
        iter += [sol.iterations]

    # Check that tol impacted convergence
    for i in range(len(tols) - 1):
        npt.assert_array_less(iter[i + 1], iter[i])

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


def test_acados_fail_external():
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.toy_examples.acados import pendulum as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
        final_time=1,
        n_shooting=2,
        expand_dynamics=True,
    )

    solver = Solver.ACADOS()
    solver.set_cost_type("EXTERNAL")

    with pytest.raises(RuntimeError, match="EXTERNAL is not interfaced yet, please use NONLINEAR_LS"):
        sol = ocp.solve(solver=solver)


def test_acados_fail_lls():
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.toy_examples.acados import static_arm as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/arm26.bioMod",
        final_time=1,
        n_shooting=2,
        use_sx=True,
        expand_dynamics=True,
    )

    solver = Solver.ACADOS()
    solver.set_cost_type("LINEAR_LS")

    with pytest.raises(
        RuntimeError, match="SUPERIMPOSE_MARKERS is an incompatible objective term with LINEAR_LS cost type"
    ):
        sol = ocp.solve(solver=solver)


@pytest.mark.parametrize("problem_type_custom", [True, False])
def test_acados_custom_dynamics(problem_type_custom):
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from tests import test_utils_ocp as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube_acados.bioMod",
        problem_type_custom=problem_type_custom,
        ode_solver=OdeSolver.RK4(),
        use_sx=True,
        expand_dynamics=True,
    )
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2")
    ocp.update_constraints(constraints)
    sol = ocp.solve(solver=Solver.ACADOS())

    # Check some results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((2, 0, 0)), decimal=6)
    npt.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))

    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((0, 9.81, 2.27903226)))
    npt.assert_almost_equal(tau[:, -1], np.array((0, 9.81, -2.27903226)))


def test_acados_one_parameter():
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.getting_started import custom_parameters as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    target_g = np.zeros((3, 1))
    target_g[2] = -9.81
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
        final_time=1,
        n_shooting=100,
        optim_gravity=True,
        optim_mass=False,
        min_g=np.array([-1, -1, -10]),
        max_g=np.array([1, 1, -5]),
        min_m=10,
        max_m=30,
        target_g=target_g,
        target_m=20,
        use_sx=True,
        expand_dynamics=True,
    )
    model = ocp.nlp[0].model
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Mayer.TRACK_STATE, key="q", target=np.array([[0, 3.14]]).T, weight=100000)
    objectives.add(ObjectiveFcn.Mayer.TRACK_STATE, key="qdot", target=np.array([[0, 0]]).T, weight=100)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", index=1, weight=10, multi_thread=False)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=0.000000010, multi_thread=False)
    ocp.update_objectives(objectives)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, 0] = 0

    u_bounds = BoundsList()
    u_bounds["tau"] = [-300] * model.nb_q, [300] * model.nb_q

    ocp.update_bounds(x_bounds, u_bounds)

    solver = Solver.ACADOS()
    solver.set_nlp_solver_tol_eq(1e-3)
    sol = ocp.solve(solver=solver)

    # Check some results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]
    gravity = sol.parameters["gravity_xyz"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((0, 0)), decimal=6)
    npt.assert_almost_equal(q[:, -1], np.array((0, 3.14)), decimal=6)

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0)), decimal=6)
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0)), decimal=6)

    # parameters
    npt.assert_almost_equal(gravity[-1], -9.80995, decimal=4)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


def test_acados_several_parameter():
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.getting_started import custom_parameters as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    target_g = np.zeros((3, 1))
    target_g[2] = -9.81
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
        final_time=1,
        n_shooting=100,
        optim_gravity=True,
        optim_mass=True,
        min_g=np.array([-1, -1, -10]),
        max_g=np.array([1, 1, -5]),
        min_m=10,
        max_m=30,
        target_g=target_g,
        target_m=20,
        use_sx=True,
        expand_dynamics=True,
    )
    model = ocp.nlp[0].model
    objectives = ObjectiveList()
    objectives.add(
        ObjectiveFcn.Mayer.TRACK_STATE, key="q", target=np.array([[0, 3.14]]).T, weight=100000, multi_thread=False
    )
    objectives.add(
        ObjectiveFcn.Mayer.TRACK_STATE, key="qdot", target=np.array([[0, 0]]).T, weight=100, multi_thread=False
    )
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", index=1, weight=10, multi_thread=False)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=0.000000010, multi_thread=False)
    ocp.update_objectives(objectives)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, 0] = 0

    u_bounds = BoundsList()
    u_bounds["tau"] = [-300] * model.nb_q, [300] * model.nb_q

    ocp.update_bounds(x_bounds, u_bounds)

    solver = Solver.ACADOS()
    solver.set_nlp_solver_tol_eq(1e-3)
    sol = ocp.solve(solver=solver)

    # Check some results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]
    gravity, mass = sol.parameters["gravity_xyz"], sol.parameters["mass"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((0, 0)), decimal=6)
    npt.assert_almost_equal(q[:, -1], np.array((0, 3.14)), decimal=6)

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0)), decimal=6)
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0)), decimal=6)

    # parameters
    npt.assert_almost_equal(gravity[-1], np.array([-9.80996]), decimal=4)
    npt.assert_almost_equal(mass[0], 20, decimal=6)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


def test_acados_one_end_constraints():
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.toy_examples.acados import cube as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube_acados.bioMod",
        n_shooting=10,
        tf=2,
        expand_dynamics=True,
    )

    model = ocp.nlp[0].model
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_STATE, index=0, key="q", weight=100, target=np.array([[1]]), multi_thread=False
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, multi_thread=False)
    ocp.update_objectives(objective_functions)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["q"][1:, [0, -1]] = 0
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0
    x_bounds["q"][0, 0] = 0

    ocp.update_bounds(x_bounds=x_bounds)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2")
    ocp.update_constraints(constraints)

    sol = ocp.solve(solver=Solver.ACADOS())

    # Check some results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # final position
    npt.assert_almost_equal(q[:, -1], np.array((2, 0, 0)))

    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((2.72727272, 9.81, 0)))
    npt.assert_almost_equal(tau[:, -1], np.array((-2.72727272, 9.81, 0)))


def test_acados_constraints_all():
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.toy_examples.tracking import track_marker_on_segment as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube_and_line.bioMod",
        n_shooting=30,
        final_time=2,
        initialize_near_solution=True,
        constr=False,
        use_sx=True,
        expand_dynamics=True,
    )

    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_MARKER_WITH_SEGMENT_AXIS, node=Node.ALL, marker="m1", segment="seg_rt", axis=Axis.X
    )
    ocp.update_constraints(constraints)

    sol = ocp.solve(solver=Solver.ACADOS())

    # Check some results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # final position
    npt.assert_almost_equal(q[:, 0], np.array([2.28988221, 0, 0, 2.95087911e-01]), decimal=6)
    npt.assert_almost_equal(q[:, -1], np.array((2.28215749, 0, 1.57, 6.62470772e-01)), decimal=6)

    npt.assert_almost_equal(qdot[:, 0], np.array([0, 0, 0, 0]), decimal=6)
    npt.assert_almost_equal(qdot[:, -1], np.array([0, 0, 0, 0]), decimal=6)

    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((0.04483914, 9.90739842, 2.24951691, 0.78496612)), decimal=6)
    npt.assert_almost_equal(tau[:, -1], np.array((0.15945561, 10.03978178, -2.36075327, 0.07267697)), decimal=6)


def test_acados_constraints_end_all():
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.toy_examples.tracking import track_marker_on_segment as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube_and_line.bioMod",
        n_shooting=30,
        final_time=2,
        initialize_near_solution=True,
        constr=False,
        use_sx=True,
        expand_dynamics=True,
    )

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m5")
    constraints.add(
        ConstraintFcn.TRACK_MARKER_WITH_SEGMENT_AXIS, node=Node.ALL_SHOOTING, marker="m1", segment="seg_rt", axis=Axis.X
    )
    ocp.update_constraints(constraints)

    sol = ocp.solve(solver=Solver.ACADOS())

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # final position
    npt.assert_almost_equal(q[:, 0], np.array([2.01701330, 0, 0, 3.20057865e-01]), decimal=6)
    npt.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57, 7.85398168e-01)), decimal=6)

    npt.assert_almost_equal(qdot[:, 0], np.array([0, 0, 0, 0]), decimal=6)
    npt.assert_almost_equal(qdot[:, -1], np.array([0, 0, 0, 0]), decimal=6)

    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((0.04648408, 9.88616194, 2.24285498, 0.864213)), decimal=6)
    npt.assert_almost_equal(tau[:, -1], np.array((0.19389194, 9.99905781, -2.37713652, -0.19858311)), decimal=6)


def test_acados_phase_dynamics_reject():
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.getting_started import basic_ocp as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
        final_time=1,
        n_shooting=10,
        phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        expand_dynamics=True,
    )

    with pytest.raises(RuntimeError, match=f"ACADOS necessitate phase_dynamics==PhaseDynamics.SHARED_DURING_THE_PHASE"):
        ocp.solve(solver=Solver.ACADOS())


@pytest.mark.parametrize("failing", ["u_bounds", "x_bounds"])
def test_acados_bounds_not_implemented(failing):
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return
    bioptim_folder = TestUtils.bioptim_folder()
    bio_model_path = bioptim_folder + "/examples/models/cart_pendulum.bioMod"
    bio_model = TorqueBiorbdModel(bio_model_path)

    nq = bio_model.nb_q
    ntau = bio_model.nb_tau

    n_cycles = 3
    window_len = 5
    window_duration = 0.2
    if failing == "u_bounds":
        x_bounds = BoundsList()
        x_bounds.add("q", min_bound=np.zeros((nq, 1)), max_bound=np.zeros((nq, 1)))
        x_bounds.add("qdot", min_bound=np.zeros((nq, 1)), max_bound=np.zeros((nq, 1)))
        u_bounds = BoundsList()
        u_bounds.add(
            "tau",
            min_bound=np.zeros((ntau, 1)),
            max_bound=np.zeros((ntau, 1)),
            interpolation=InterpolationType.CONSTANT,
        )
    elif failing == "x_bounds":
        x_bounds = BoundsList()
        x_bounds.add(
            "q", min_bound=np.zeros((nq, 1)), max_bound=np.zeros((nq, 1)), interpolation=InterpolationType.CONSTANT
        )
        x_bounds.add(
            "qdot", min_bound=np.zeros((nq, 1)), max_bound=np.zeros((nq, 1)), interpolation=InterpolationType.CONSTANT
        )
        u_bounds = BoundsList()
        u_bounds.add("tau", min_bound=np.zeros((ntau, 1)), max_bound=np.zeros((ntau, 1)))
    else:
        raise ValueError("Wrong value for failing")

    mhe = MovingHorizonEstimator(
        bio_model,
        window_len,
        window_duration,
        dynamics=DynamicsOptions(expand_dynamics=True),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        n_threads=4,
    )

    def update_functions(mhe, t, _):
        return t < n_cycles

    with pytest.raises(
        NotImplementedError,
        match=f"ACADOS must declare an InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT for the {failing}",
    ):
        mhe.solve(update_functions, Solver.ACADOS())
