import numpy as np
import numpy.testing as npt
import pytest
from casadi import Function, has_nlpsol, jacobian

from bioptim import (
    BoundsList,
    DefectType,
    DynamicsOptions,
    DynamicsOptionsList,
    OdeSolver,
    OptimalControlProgram,
    OrderingStrategy,
    PhaseDynamics,
    Solver,
    TorqueBiorbdModel,
    VariableScalingList,
)
from bioptim.interfaces.fatrop_interface import FatropInterface
from bioptim.interfaces.ipopt_interface import IpoptInterface
from tests.utils import TestUtils


def _prepare_scaled_pendulum(ode_solver, ordering_strategy=OrderingStrategy.TIME_MAJOR, n_shooting=3):
    from bioptim.examples.toy_examples.feature_examples import example_variable_scaling as ocp_module

    return ocp_module.prepare_ocp(
        biorbd_model_path=TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod",
        final_time=1,
        n_shooting=n_shooting,
        ode_solver=ode_solver,
        ordering_strategy=ordering_strategy,
    )


def _prepare_scaled_multiphase_pendulum():
    model_path = TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod"
    models = (TorqueBiorbdModel(model_path), TorqueBiorbdModel(model_path))
    dynamics = DynamicsOptionsList()
    x_bounds = BoundsList()
    u_bounds = BoundsList()
    x_scaling = VariableScalingList()
    u_scaling = VariableScalingList()

    for phase, model in enumerate(models):
        dynamics.add(
            DynamicsOptions(ode_solver=OdeSolver.RK4(), phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE)
        )
        x_bounds.add("q", bounds=model.bounds_from_ranges("q"), phase=phase)
        x_bounds.add("qdot", bounds=model.bounds_from_ranges("qdot"), phase=phase)
        u_bounds.add("tau", min_bound=[-1000] * model.nb_tau, max_bound=[1000] * model.nb_tau, phase=phase)
        x_scaling.add("q", scaling=[1, 3], phase=phase)
        x_scaling.add("qdot", scaling=[85, 85], phase=phase)
        u_scaling.add("tau", scaling=[900, 1], phase=phase)

    return OptimalControlProgram(
        models,
        (2, 2),
        (1, 1),
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_scaling=x_scaling,
        u_scaling=u_scaling,
        ordering_strategy=OrderingStrategy.TIME_MAJOR,
    )


def test_fatrop_rejects_variable_major_ordering():
    ocp = _prepare_scaled_pendulum(OdeSolver.RK4(), ordering_strategy=OrderingStrategy.VARIABLE_MAJOR)

    with pytest.raises(ValueError, match="FATROP requires OrderingStrategy.TIME_MAJOR"):
        FatropInterface(ocp)


@pytest.mark.parametrize(
    "ode_solver",
    [
        OdeSolver.RK4(),
        OdeSolver.COLLOCATION(polynomial_degree=3),
    ],
)
def test_fatrop_normalizes_scaled_state_continuity(ode_solver):
    ocp = _prepare_scaled_pendulum(ode_solver)
    nlp = ocp.nlp[0]
    variables = ocp.variables_vector

    fatrop_constraints, _ = FatropInterface(ocp).dispatch_bounds()
    ipopt_constraints, _ = IpoptInterface(ocp).dispatch_bounds()

    fatrop_jacobian = Function("fatrop_jacobian", [variables], [jacobian(fatrop_constraints, variables)])
    ipopt_jacobian = Function("ipopt_jacobian", [variables], [jacobian(ipopt_constraints, variables)])
    fatrop_jacobian_at_init = np.array(fatrop_jacobian(ocp.init_vector))
    ipopt_jacobian_at_init = np.array(ipopt_jacobian(ocp.init_vector))

    n_states = nlp.states.shape
    next_state_slice = ocp.vector_layout.index_map[(0, "states", 1)][0]
    next_state_columns = slice(next_state_slice.start, next_state_slice.start + n_states)
    state_scaling = np.concatenate([nlp.x_scaling[key].scaling[:, 0] for key in nlp.states.keys()])

    npt.assert_allclose(
        ipopt_jacobian_at_init[:n_states, next_state_columns],
        np.diag(state_scaling),
        atol=1e-12,
    )
    npt.assert_allclose(
        fatrop_jacobian_at_init[:n_states, next_state_columns],
        np.eye(n_states),
        atol=1e-12,
    )

    if ode_solver.is_direct_collocation:
        rows_per_node = n_states * (ode_solver.polynomial_degree + 1)
        constraint_scaling = np.tile(state_scaling, ode_solver.polynomial_degree + 1)
        test_point = np.array(ocp.init_vector)
        test_point[1:] += np.random.default_rng(42).normal(scale=0.01, size=test_point[1:].shape)

        fatrop_values = np.array(Function("fatrop_g", [variables], [fatrop_constraints])(test_point))[
            :rows_per_node, 0
        ]
        ipopt_values = np.array(Function("ipopt_g", [variables], [ipopt_constraints])(test_point))[:rows_per_node, 0]
        npt.assert_allclose(fatrop_values * constraint_scaling, ipopt_values, atol=1e-10)


@pytest.mark.skipif(not has_nlpsol("fatrop"), reason="CasADi was built without FATROP")
def test_fatrop_normalizes_scaled_phase_transition():
    ocp = _prepare_scaled_multiphase_pendulum()
    constraints, _ = FatropInterface(ocp).dispatch_bounds()
    variables = ocp.variables_vector
    jacobian_function = Function("multiphase_jacobian", [variables], [jacobian(constraints, variables)])
    jacobian_at_init = np.array(jacobian_function(ocp.init_vector))

    n_states = ocp.nlp[0].states.shape
    transition_rows = slice(ocp.nlp[0].ns * n_states, (ocp.nlp[0].ns + 1) * n_states)
    post_state_slice = ocp.vector_layout.index_map[(1, "states", 0)][0]
    post_state_columns = slice(post_state_slice.start, post_state_slice.start + n_states)
    npt.assert_allclose(jacobian_at_init[transition_rows, post_state_columns], np.eye(n_states), atol=1e-12)

    solver = Solver.FATROP()
    solver.set_print_level(0)
    solution = ocp.solve(solver)
    assert solution.status == 0


@pytest.mark.skipif(not has_nlpsol("fatrop"), reason="CasADi was built without FATROP")
@pytest.mark.parametrize(
    "ode_solver",
    [
        OdeSolver.RK4(),
        OdeSolver.COLLOCATION(polynomial_degree=3),
        OdeSolver.COLLOCATION(
            polynomial_degree=3,
            defects_type=DefectType.TAU_EQUALS_INVERSE_DYNAMICS,
        ),
    ],
)
def test_fatrop_solves_scaled_pendulum(ode_solver):
    ocp = _prepare_scaled_pendulum(ode_solver, n_shooting=30)
    solver = Solver.FATROP()
    solver.set_print_level(0)

    solution = ocp.solve(solver)

    assert solution.status == 0
    assert np.max(np.abs(np.array(solution.constraints))) < 1e-6
