import casadi
import numpy as np
import pytest

import bioptim
from bioptim import (
    BlockShooting,
    BoundsList,
    Constraint,
    ConstraintFcn,
    ControlType,
    DynamicsOptions,
    DynamicsOptionsList,
    InitialGuessList,
    InterpolationType,
    Node,
    Objective,
    ObjectiveFcn,
    OdeSolver,
    OptimalControlProgram,
    OrderingStrategy,
    ParameterList,
    PhaseDynamics,
    Solution,
    SolutionMerge,
    Solver,
    TorqueBiorbdModel,
    VariableScalingList,
    VariableScaling,
)
from bioptim.optimization.vector_layout import VectorLayout

from ..utils import TestUtils


def _prepare_block_ocp(
    n_shooting=6,
    block_shooting=BlockShooting(n_blocks=2),
    *,
    ode_solver=None,
    control_type=ControlType.CONSTANT,
    use_sx=False,
    ordering_strategy=OrderingStrategy.VARIABLE_MAJOR,
    skip_continuity=True,
    x_bounds=None,
    x_init=None,
    constraints=None,
    objective_functions=None,
    x_scaling=None,
    phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
):
    model = TorqueBiorbdModel(TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod")
    dynamics = DynamicsOptions(
        ode_solver=ode_solver if ode_solver is not None else OdeSolver.RK4(),
        skip_continuity=skip_continuity,
        phase_dynamics=phase_dynamics,
    )
    return OptimalControlProgram(
        model,
        n_shooting=n_shooting,
        phase_time=1.0,
        dynamics=dynamics,
        control_type=control_type,
        use_sx=use_sx,
        ordering_strategy=ordering_strategy,
        block_shooting=block_shooting,
        x_bounds=x_bounds,
        x_init=x_init,
        constraints=constraints,
        objective_functions=objective_functions,
        x_scaling=x_scaling,
    )


def _prepare_solvable_block_ocp(n_blocks, ordering_strategy=OrderingStrategy.VARIABLE_MAJOR):
    model = TorqueBiorbdModel(TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod")

    x_bounds = BoundsList()
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][1, -1] = 0.5
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0

    x_init = InitialGuessList()
    x_init.add("q", initial_guess=[[0.0, 0.0], [0.0, 0.5]], interpolation=InterpolationType.LINEAR)
    x_init["qdot"] = [0.0, 0.0]

    u_bounds = BoundsList()
    u_bounds["tau"] = [-100.0, 0.0], [100.0, 0.0]

    return OptimalControlProgram(
        model,
        n_shooting=6,
        phase_time=1.0,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4()),
        x_bounds=x_bounds,
        x_init=x_init,
        u_bounds=u_bounds,
        objective_functions=Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau"),
        block_shooting=None if n_blocks is None else BlockShooting(n_blocks=n_blocks),
        ordering_strategy=ordering_strategy,
        use_sx=False,
    )


@pytest.mark.parametrize(
    ("n_shooting", "n_blocks", "expected"),
    [
        (1, 1, (0, 1)),
        (10, 1, (0, 10)),
        (10, 2, (0, 5, 10)),
        (10, 3, (0, 4, 7, 10)),
        (10, 4, (0, 3, 6, 8, 10)),
        (10, 6, (0, 2, 4, 6, 8, 9, 10)),
        (10, 10, tuple(range(11))),
    ],
)
def test_compute_block_boundaries_from_number_of_blocks(n_shooting, n_blocks, expected):
    assert bioptim.compute_block_boundaries(n_shooting, n_blocks=n_blocks) == expected


@pytest.mark.parametrize(
    ("n_shooting", "block_size", "expected"),
    [
        (1, 1, (0, 1)),
        (10, 1, tuple(range(11))),
        (10, 3, (0, 3, 6, 9, 10)),
        (10, 4, (0, 4, 8, 10)),
        (10, 5, (0, 5, 10)),
        (10, 10, (0, 10)),
        (10, 20, (0, 10)),
    ],
)
def test_compute_block_boundaries_from_block_size(n_shooting, block_size, expected):
    assert bioptim.compute_block_boundaries(n_shooting, block_size=block_size) == expected


def test_number_of_blocks_partition_invariants_on_a_broad_input_grid():
    for n_shooting in range(1, 31):
        for n_blocks in range(1, n_shooting + 1):
            boundaries = bioptim.compute_block_boundaries(n_shooting, n_blocks=n_blocks)
            block_lengths = tuple(stop - start for start, stop in zip(boundaries, boundaries[1:]))

            assert boundaries[0] == 0
            assert boundaries[-1] == n_shooting
            assert len(boundaries) == n_blocks + 1
            assert all(length >= 1 for length in block_lengths)
            assert max(block_lengths) - min(block_lengths) <= 1
            assert tuple(sorted(block_lengths, reverse=True)) == block_lengths


def test_block_size_partition_invariants_on_a_broad_input_grid():
    for n_shooting in range(1, 31):
        for block_size in range(1, 35):
            boundaries = bioptim.compute_block_boundaries(n_shooting, block_size=block_size)
            block_lengths = tuple(stop - start for start, stop in zip(boundaries, boundaries[1:]))

            assert boundaries[0] == 0
            assert boundaries[-1] == n_shooting
            assert all(length >= 1 for length in block_lengths)
            assert all(length == block_size for length in block_lengths[:-1])
            assert block_lengths[-1] <= block_size


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "Exactly one of n_blocks and block_size must be provided"),
        (
            {"n_blocks": 2, "block_size": 5},
            "Exactly one of n_blocks and block_size must be provided",
        ),
        ({"n_blocks": 0}, "n_blocks must be between 1 and n_shooting"),
        ({"n_blocks": -1}, "n_blocks must be between 1 and n_shooting"),
        ({"n_blocks": 11}, "n_blocks must be between 1 and n_shooting"),
        ({"block_size": 0}, "block_size must be greater than or equal to 1"),
        ({"block_size": -1}, "block_size must be greater than or equal to 1"),
    ],
)
def test_compute_block_boundaries_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bioptim.compute_block_boundaries(10, **kwargs)


@pytest.mark.parametrize(
    ("n_shooting", "kwargs", "argument"),
    [
        (True, {"n_blocks": 1}, "n_shooting"),
        (10.0, {"n_blocks": 1}, "n_shooting"),
        ("10", {"n_blocks": 1}, "n_shooting"),
        (10, {"n_blocks": True}, "n_blocks"),
        (10, {"n_blocks": 2.0}, "n_blocks"),
        (10, {"n_blocks": "2"}, "n_blocks"),
        (10, {"block_size": True}, "block_size"),
        (10, {"block_size": 2.0}, "block_size"),
        (10, {"block_size": "2"}, "block_size"),
    ],
)
def test_compute_block_boundaries_rejects_non_integer_values(n_shooting, kwargs, argument):
    with pytest.raises(TypeError, match=rf"{argument} must be an integer"):
        bioptim.compute_block_boundaries(n_shooting, **kwargs)


@pytest.mark.parametrize("n_shooting", [0, -1])
def test_compute_block_boundaries_rejects_non_positive_n_shooting(n_shooting):
    with pytest.raises(ValueError, match="n_shooting must be greater than or equal to 1"):
        bioptim.compute_block_boundaries(n_shooting, n_blocks=1)


def test_block_shooting_public_api_and_convenience_constructors():
    assert bioptim.BlockShooting(n_blocks=1) == bioptim.BlockShooting.single()
    assert bioptim.BlockShooting(n_blocks=5) == bioptim.BlockShooting.from_number_of_blocks(5)
    assert bioptim.BlockShooting(block_size=10) == bioptim.BlockShooting.from_block_size(10)

    assert bioptim.BlockShooting.single().compute_boundaries(10) == (0, 10)
    assert bioptim.BlockShooting.from_number_of_blocks(3).compute_boundaries(10) == (0, 4, 7, 10)
    assert bioptim.BlockShooting.from_block_size(4).compute_boundaries(10) == (0, 4, 8, 10)


def test_getting_started_block_shooting_example_builds():
    from bioptim.examples.getting_started import block_shooting as example

    ocp = example.prepare_ocp(
        biorbd_model_path=TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod",
        n_shooting=6,
        n_blocks=2,
    )

    assert ocp.nlp[0].block_boundaries == (0, 3, 6)
    assert ocp.nlp[0].decision_state_nodes == (0, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"n_blocks": 2, "block_size": 5},
        {"n_blocks": True},
        {"n_blocks": 0},
        {"block_size": True},
        {"block_size": 0},
    ],
)
def test_block_shooting_rejects_invalid_configuration_at_construction(kwargs):
    with pytest.raises((TypeError, ValueError)):
        bioptim.BlockShooting(**kwargs)


def test_block_shooting_configuration_is_immutable():
    shooting = bioptim.BlockShooting(n_blocks=3)

    with pytest.raises(AttributeError):
        shooting.n_blocks = 4


@pytest.mark.parametrize("ordering", [OrderingStrategy.VARIABLE_MAJOR, OrderingStrategy.TIME_MAJOR])
def test_default_dms_keeps_a_decision_state_at_every_node(ordering):
    from bioptim.examples.getting_started import basic_ocp

    n_shooting = 4
    ocp = basic_ocp.prepare_ocp(
        biorbd_model_path=TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod",
        final_time=1.0,
        n_shooting=n_shooting,
        use_sx=False,
        ordering_strategy=ordering,
    )

    state_keys = tuple(key for key in ocp.vector_layout.index_map if len(key) == 3 and key[1] == "states")

    assert state_keys == tuple((0, "states", node) for node in range(n_shooting + 1))
    assert len(ocp.nlp[0].X_scaled) == n_shooting + 1
    assert all(state.is_symbolic() for state in ocp.nlp[0].X_scaled)


def test_vector_layout_preserves_all_default_dms_state_keys():
    from bioptim.examples.getting_started import basic_ocp

    ocp = basic_ocp.prepare_ocp(
        biorbd_model_path=TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod",
        final_time=1.0,
        n_shooting=3,
        use_sx=False,
    )
    layout = VectorLayout(ocp)
    original_state_keys = tuple(key for key in layout.index_map if len(key) == 3 and key[1] == "states")

    assert original_state_keys == (
        (0, "states", 0),
        (0, "states", 1),
        (0, "states", 2),
        (0, "states", 3),
    )


@pytest.mark.parametrize(
    ("configuration", "expected_boundaries"),
    [
        (BlockShooting.single(), (0, 6)),
        (BlockShooting(n_blocks=2), (0, 3, 6)),
        (BlockShooting(n_blocks=3), (0, 2, 4, 6)),
        (BlockShooting(block_size=4), (0, 4, 6)),
        (BlockShooting(n_blocks=6), tuple(range(7))),
    ],
)
def test_ocp_exposes_block_partition_and_decision_state_nodes(configuration, expected_boundaries):
    ocp = _prepare_block_ocp(block_shooting=configuration)
    nlp = ocp.nlp[0]

    assert nlp.block_shooting == configuration
    assert nlp.block_boundaries == expected_boundaries
    assert nlp.decision_state_nodes == expected_boundaries[:-1]


@pytest.mark.parametrize("ordering", [OrderingStrategy.VARIABLE_MAJOR, OrderingStrategy.TIME_MAJOR])
def test_block_shooting_vector_layout_contains_only_independent_states(ordering):
    ocp = _prepare_block_ocp(
        n_shooting=6,
        block_shooting=BlockShooting(n_blocks=3),
        ordering_strategy=ordering,
    )

    state_keys = tuple(key for key in ocp.vector_layout.index_map if len(key) == 3 and key[1] == "states")

    assert state_keys == ((0, "states", 0), (0, "states", 2), (0, "states", 4))
    assert ocp.variables_vector.numel() == 1 + 3 * 4 + 6 * 2


@pytest.mark.parametrize("n_blocks", [1, 2, 3, 6])
def test_block_shooting_separates_independent_states_from_the_full_symbolic_trajectory(n_blocks):
    ocp = _prepare_block_ocp(n_shooting=6, block_shooting=BlockShooting(n_blocks=n_blocks))
    nlp = ocp.nlp[0]

    assert len(nlp.X_decision_scaled) == n_blocks
    assert len(nlp.X_decision) == n_blocks
    assert len(nlp.X_scaled) == nlp.ns + 1
    assert len(nlp.X) == nlp.ns + 1
    assert len(nlp.block_end_states_scaled) == n_blocks
    assert len(nlp.block_end_states) == n_blocks
    assert nlp.X_scaled[-1] is nlp.block_end_states_scaled[-1]

    for node in nlp.decision_state_nodes:
        assert nlp.X_scaled[node] is nlp.X_decision_scaled[node]
        assert nlp.X_scaled[node].is_symbolic()

    for node in set(range(nlp.ns + 1)) - set(nlp.decision_state_nodes):
        assert not nlp.X_scaled[node].is_symbolic()


def test_default_dms_keeps_decision_and_full_state_trajectories_aliased():
    from bioptim.examples.getting_started import basic_ocp

    ocp = basic_ocp.prepare_ocp(
        biorbd_model_path=TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod",
        final_time=1.0,
        n_shooting=4,
        use_sx=False,
    )
    nlp = ocp.nlp[0]

    assert nlp.block_shooting is None
    assert nlp.block_boundaries is None
    assert nlp.decision_state_nodes == tuple(range(nlp.ns + 1))
    assert nlp.X_decision_scaled is nlp.X_scaled
    assert nlp.X_decision is nlp.X


def test_single_block_configuration_is_duplicated_across_phases():
    model_path = TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod"
    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.RK4(), skip_continuity=True))
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.RK4(), skip_continuity=True))

    ocp = OptimalControlProgram(
        [TorqueBiorbdModel(model_path), TorqueBiorbdModel(model_path)],
        n_shooting=[4, 6],
        phase_time=[1.0, 1.0],
        dynamics=dynamics,
        use_sx=False,
        block_shooting=BlockShooting(n_blocks=2),
    )

    assert [nlp.block_boundaries for nlp in ocp.nlp] == [(0, 2, 4), (0, 3, 6)]


def test_block_configuration_can_differ_across_phases():
    model_path = TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod"
    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.RK4(), skip_continuity=True))
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.RK4(), skip_continuity=True))

    ocp = OptimalControlProgram(
        [TorqueBiorbdModel(model_path), TorqueBiorbdModel(model_path)],
        n_shooting=[4, 6],
        phase_time=[1.0, 1.0],
        dynamics=dynamics,
        use_sx=False,
        block_shooting=[BlockShooting.single(), BlockShooting(block_size=2)],
    )

    assert [nlp.block_boundaries for nlp in ocp.nlp] == [(0, 4), (0, 2, 4, 6)]


@pytest.mark.parametrize(
    ("configuration", "message"),
    [
        ([BlockShooting.single()], "block_shooting must provide one configuration per phase"),
        ([BlockShooting.single(), "invalid"], "block_shooting entries must be BlockShooting instances"),
        ("invalid", "block_shooting must be a BlockShooting or a sequence of BlockShooting"),
    ],
)
def test_multiphase_block_configuration_rejects_invalid_inputs(configuration, message):
    model_path = TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod"
    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.RK4(), skip_continuity=True))
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.RK4(), skip_continuity=True))

    with pytest.raises((TypeError, ValueError), match=message):
        OptimalControlProgram(
            [TorqueBiorbdModel(model_path), TorqueBiorbdModel(model_path)],
            n_shooting=[4, 6],
            phase_time=[1.0, 1.0],
            dynamics=dynamics,
            use_sx=False,
            block_shooting=configuration,
        )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK1(), OdeSolver.RK2(), OdeSolver.RK4(), OdeSolver.RK8()])
def test_block_shooting_accepts_explicit_runge_kutta_integrators(ode_solver, phase_dynamics):
    ocp = _prepare_block_ocp(ode_solver=ode_solver, phase_dynamics=phase_dynamics)

    assert ocp.nlp[0].block_boundaries == (0, 3, 6)


@pytest.mark.parametrize("ode_solver", [OdeSolver.COLLOCATION(), OdeSolver.IRK(), OdeSolver.TRAPEZOIDAL()])
def test_block_shooting_rejects_unsupported_integrators(ode_solver):
    with pytest.raises(NotImplementedError, match="explicit Runge-Kutta integrators with MX graphs only"):
        _prepare_block_ocp(ode_solver=ode_solver)


def test_block_shooting_rejects_sx_graphs():
    with pytest.raises(NotImplementedError, match="explicit Runge-Kutta integrators with MX graphs only"):
        _prepare_block_ocp(use_sx=True)


def test_block_shooting_rejects_nonempty_algebraic_states():
    from .test_controltype_none import NonControlledMethod

    class ModelWithAlgebraicState(NonControlledMethod):
        @property
        def extra_configuration_functions(self):
            return [lambda ocp, nlp: self.declare_variables(ocp, nlp)]

        def declare_variables(self, ocp, nlp):
            super().declare_variables(ocp, nlp)
            bioptim.ConfigureVariables.configure_new_variable(
                "algebraic",
                ["algebraic"],
                ocp,
                nlp,
                as_states=False,
                as_controls=False,
                as_algebraic_states=True,
            )

        def dynamics(
            self,
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
        ):
            return bioptim.DynamicsEvaluation(dxdt=casadi.vertcat(states[1], -states[0], states[2]), defects=None)

    with pytest.raises(NotImplementedError, match="non-empty algebraic states"):
        OptimalControlProgram(
            ModelWithAlgebraicState(),
            n_shooting=4,
            phase_time=1.0,
            dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4()),
            block_shooting=BlockShooting(n_blocks=2),
            use_sx=False,
        )


@pytest.mark.parametrize("n_blocks", [1, 2, 3, 6])
def test_block_shooting_declares_only_block_continuity_constraints(n_blocks):
    ocp = _prepare_block_ocp(
        n_shooting=6,
        block_shooting=BlockShooting(n_blocks=n_blocks),
        skip_continuity=False,
    )
    nlp = ocp.nlp[0]
    block_continuities = [penalty for penalty in nlp.g_internal if penalty]

    assert len(block_continuities) == n_blocks - 1
    assert all(penalty.type == ConstraintFcn.BLOCK_STATE_CONTINUITY for penalty in block_continuities)
    assert [penalty.node_idx for penalty in block_continuities] == [
        [boundary - 1] for boundary in nlp.block_boundaries[1:-1]
    ]


@pytest.mark.parametrize("n_blocks", [1, 5, 10, 25, 100])
def test_n100_block_graph_has_theoretical_nlp_dimensions(n_blocks):
    ocp = _prepare_block_ocp(
        n_shooting=100,
        block_shooting=BlockShooting(n_blocks=n_blocks),
        skip_continuity=False,
    )
    nlp = ocp.nlp[0]
    block_continuities = [penalty for penalty in nlp.g_internal if penalty]

    assert len(nlp.X_decision_scaled) == n_blocks
    assert ocp.variables_vector.numel() == 1 + n_blocks * nlp.states.shape + 100 * nlp.controls.shape
    assert len(block_continuities) == n_blocks - 1
    continuity_size = sum(
        penalty.function[node].size_out("val")[0]
        for penalty in block_continuities
        for node in penalty.node_idx
    )
    assert continuity_size == (n_blocks - 1) * nlp.states.shape


@pytest.mark.parametrize("n_blocks", [2, 3, 6])
def test_each_block_continuity_has_exactly_one_state_vector(n_blocks):
    ocp = _prepare_block_ocp(
        n_shooting=6,
        block_shooting=BlockShooting(n_blocks=n_blocks),
        skip_continuity=False,
    )
    nlp = ocp.nlp[0]

    for boundary, penalty in zip(nlp.block_boundaries[1:-1], nlp.g_internal):
        function = penalty.function[boundary - 1]
        assert function is not None
        assert function.size_out("val") == (nlp.states.shape, 1)


def test_block_shooting_rejects_continuity_as_an_objective_for_now():
    model = TorqueBiorbdModel(TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod")
    dynamics = DynamicsOptions(
        ode_solver=OdeSolver.RK4(),
        skip_continuity=False,
        state_continuity_weight=100,
    )

    with pytest.raises(NotImplementedError, match="continuity as an objective"):
        OptimalControlProgram(
            model,
            n_shooting=6,
            phase_time=1.0,
            dynamics=dynamics,
            use_sx=False,
            block_shooting=BlockShooting(n_blocks=2),
        )


def test_finite_bounds_on_eliminated_state_nodes_become_internal_constraints():
    minimum = np.array([[-10.0, -20.0, -30.0], [-11.0, -21.0, -31.0]])
    maximum = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])
    x_bounds = BoundsList()
    x_bounds.add("q", min_bound=minimum, max_bound=maximum)

    ocp = _prepare_block_ocp(x_bounds=x_bounds)
    bound_constraints = [
        penalty for penalty in ocp.nlp[0].g_internal if penalty and penalty.type == ConstraintFcn.BOUND_STATE
    ]

    assert [penalty.node_idx for penalty in bound_constraints] == [[1], [2], [4], [5], [6]]
    for penalty in bound_constraints[:-1]:
        np.testing.assert_allclose(penalty.bounds.min, minimum[:, 1:2])
        np.testing.assert_allclose(penalty.bounds.max, maximum[:, 1:2])
    np.testing.assert_allclose(bound_constraints[-1].bounds.min, minimum[:, 2:3])
    np.testing.assert_allclose(bound_constraints[-1].bounds.max, maximum[:, 2:3])


def test_infinite_rows_do_not_create_constraints_on_eliminated_states():
    x_bounds = BoundsList()
    x_bounds.add(
        "q",
        min_bound=np.array([[-np.inf], [-1.0]]),
        max_bound=np.array([[np.inf], [1.0]]),
        interpolation=InterpolationType.CONSTANT,
    )

    ocp = _prepare_block_ocp(x_bounds=x_bounds)
    bound_constraints = [
        penalty for penalty in ocp.nlp[0].g_internal if penalty and penalty.type == ConstraintFcn.BOUND_STATE
    ]

    assert len(bound_constraints) == 5
    assert all(penalty.rows.tolist() == [1] for penalty in bound_constraints)


def test_state_bounds_vector_only_contains_independent_block_starts():
    minimum = np.vstack((np.arange(7), np.arange(10, 17)))
    maximum = minimum + 100
    x_bounds = BoundsList()
    x_bounds.add(
        "q",
        min_bound=minimum,
        max_bound=maximum,
        interpolation=InterpolationType.EACH_FRAME,
    )

    ocp = _prepare_block_ocp(x_bounds=x_bounds)
    lower, upper = ocp.bounds_vectors
    unstacked_lower = ocp.vector_layout.unstack(lower)
    unstacked_upper = ocp.vector_layout.unstack(upper)

    assert (0, "states", 0) in unstacked_lower
    assert (0, "states", 3) in unstacked_lower
    assert (0, "states", 6) not in unstacked_lower
    np.testing.assert_allclose(unstacked_lower[(0, "states", 0)][:2, 0], minimum[:, 0])
    np.testing.assert_allclose(unstacked_lower[(0, "states", 3)][:2, 0], minimum[:, 3])
    np.testing.assert_allclose(unstacked_upper[(0, "states", 0)][:2, 0], maximum[:, 0])
    np.testing.assert_allclose(unstacked_upper[(0, "states", 3)][:2, 0], maximum[:, 3])


def test_state_initial_guess_vector_only_samples_independent_block_starts():
    values = np.vstack((np.arange(7), np.arange(10, 17)))
    x_init = InitialGuessList()
    x_init.add("q", initial_guess=values, interpolation=InterpolationType.EACH_FRAME)

    ocp = _prepare_block_ocp(x_init=x_init)
    unstacked = ocp.vector_layout.unstack(ocp.init_vector)

    assert (0, "states", 0) in unstacked
    assert (0, "states", 3) in unstacked
    assert (0, "states", 6) not in unstacked
    np.testing.assert_allclose(unstacked[(0, "states", 0)][:2, 0], values[:, 0])
    np.testing.assert_allclose(unstacked[(0, "states", 3)][:2, 0], values[:, 3])


def test_solution_reconstruction_respects_nontrivial_state_scaling():
    values = np.vstack((np.arange(2, 9), np.arange(12, 19)))
    x_init = InitialGuessList()
    x_init.add("q", initial_guess=values, interpolation=InterpolationType.EACH_FRAME)

    x_scaling = VariableScalingList()
    x_scaling.add("q", scaling=[2.0, 3.0])
    x_scaling.add("qdot", scaling=[4.0, 5.0])

    ocp = _prepare_block_ocp(x_init=x_init, x_scaling=x_scaling)
    solution = Solution.from_vector(ocp, ocp.init_vector)
    scaled = solution.decision_states(scaled=True, to_merge=SolutionMerge.NODES)
    unscaled = solution.decision_states(scaled=False, to_merge=SolutionMerge.NODES)

    np.testing.assert_allclose(unscaled["q"], scaled["q"] * np.array([[2.0], [3.0]]))
    np.testing.assert_allclose(unscaled["qdot"], scaled["qdot"] * np.array([[4.0], [5.0]]))
    np.testing.assert_allclose(unscaled["q"][:, 0], values[:, 0])
    np.testing.assert_allclose(unscaled["q"][:, 3], values[:, 3])


@pytest.mark.parametrize(
    ("control_type", "expected_control_nodes", "last_control_is_used"),
    [
        (ControlType.CONSTANT, 6, True),
        (ControlType.CONSTANT_WITH_LAST_NODE, 7, False),
        (ControlType.LINEAR_CONTINUOUS, 7, True),
    ],
)
def test_block_rollout_uses_the_expected_controls_on_the_last_interval(
    control_type, expected_control_nodes, last_control_is_used
):
    ocp = _prepare_block_ocp(control_type=control_type)
    nlp = ocp.nlp[0]
    terminal_symbol_names = {symbol.name() for symbol in casadi.symvar(nlp.X_scaled[-1])}

    assert len(nlp.U_scaled) == expected_control_nodes
    last_control_name = f"U_scaled_0_{expected_control_nodes - 1}"
    assert (last_control_name in terminal_symbol_names) is last_control_is_used


@pytest.mark.parametrize("block_shooting", [None, BlockShooting(n_blocks=2)])
def test_explicit_runge_kutta_supports_control_type_none(block_shooting):
    from .test_controltype_none import NonControlledMethod

    class AutonomousModel(NonControlledMethod):
        def dynamics(
            self,
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
        ):
            return bioptim.DynamicsEvaluation(
                dxdt=casadi.vertcat(100.0 + states[1], states[0] / 100.0, states[0] * states[1] + states[2]),
                defects=None,
            )

    ocp = OptimalControlProgram(
        AutonomousModel(),
        n_shooting=4,
        phase_time=0.01,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4()),
        control_type=ControlType.NONE,
        block_shooting=block_shooting,
        use_sx=False,
    )

    number_of_state_nodes_in_vector = 5 if block_shooting is None else 2
    assert ocp.nlp[0].controls.shape == 0
    assert ocp.variables_vector.numel() == 1 + number_of_state_nodes_in_vector * 3
    if block_shooting is not None:
        assert ocp.nlp[0].state_rollout_function(ocp.init_vector).shape == (3, 5)


def test_block_rollout_keeps_optimized_time_symbolic():
    x_init = InitialGuessList()
    x_init["q"] = [0.0, 0.2]
    objective = Objective(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.5, max_bound=1.5)
    ocp = _prepare_block_ocp(x_init=x_init, objective_functions=objective)

    short_duration_vector = ocp.init_vector.copy()
    long_duration_vector = ocp.init_vector.copy()
    short_duration_vector[0, 0] = 0.5 / ocp.nlp[0].ns
    long_duration_vector[0, 0] = 1.5 / ocp.nlp[0].ns

    short_rollout = ocp.nlp[0].state_rollout_function(short_duration_vector).full()
    long_rollout = ocp.nlp[0].state_rollout_function(long_duration_vector).full()

    assert not np.allclose(short_rollout[:, -1], long_rollout[:, -1])


def test_block_rollout_binds_numerical_timeseries_at_each_interval():
    from .test_controltype_none import NonControlledMethod

    class ForcedAutonomousModel(NonControlledMethod):
        def dynamics(
            self,
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
        ):
            return bioptim.DynamicsEvaluation(
                dxdt=casadi.vertcat(numerical_timeseries[0], 0.0, 0.0),
                defects=None,
            )

    forcing = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape((1, 1, 5))
    ocp = OptimalControlProgram(
        ForcedAutonomousModel(),
        n_shooting=4,
        phase_time=1.0,
        dynamics=DynamicsOptions(
            ode_solver=OdeSolver.RK4(),
            skip_continuity=True,
            numerical_data_timeseries={"forcing": forcing},
        ),
        control_type=ControlType.NONE,
        block_shooting=BlockShooting.single(),
        use_sx=False,
    )

    rollout = ocp.nlp[0].state_rollout_function(ocp.init_vector).full()

    np.testing.assert_allclose(rollout[0, :], [0.0, 0.25, 0.75, 1.5, 2.5])
    np.testing.assert_allclose(rollout[1:, :], 0.0)


def test_block_shooting_primal_warm_start_samples_only_block_starts():
    ocp = _prepare_block_ocp()
    original_solution = Solution.from_vector(ocp, ocp.init_vector)
    expected_states = original_solution.decision_states(scaled=True, to_merge=SolutionMerge.NODES)

    ocp.set_warm_start(original_solution)
    warm_start = ocp.vector_layout.unstack(ocp.init_vector)

    assert ocp._is_warm_starting
    assert (0, "states", 6) not in warm_start
    for node in ocp.nlp[0].decision_state_nodes:
        expected = np.vstack((expected_states["q"][:, node : node + 1], expected_states["qdot"][:, node : node + 1]))
        np.testing.assert_allclose(warm_start[(0, "states", node)], expected)


def test_warm_start_from_dms_skips_incompatible_dual_variables(monkeypatch):
    dms_ocp = _prepare_solvable_block_ocp(n_blocks=None)
    dms_solution = Solution.from_vector(dms_ocp, dms_ocp.init_vector)
    dms_states = dms_solution.decision_states(scaled=True, to_merge=SolutionMerge.NODES)

    block_ocp = _prepare_solvable_block_ocp(n_blocks=2)
    block_ocp.set_ocp_solver(Solver.IPOPT())
    dual_warm_start_calls = []
    monkeypatch.setattr(
        block_ocp.ocp_solver,
        "set_lagrange_multiplier",
        lambda solution: dual_warm_start_calls.append(solution),
    )

    block_ocp.set_warm_start(dms_solution)
    warm_start = block_ocp.vector_layout.unstack(block_ocp.init_vector)

    assert not dual_warm_start_calls
    for node in block_ocp.nlp[0].decision_state_nodes:
        expected = np.vstack((dms_states["q"][:, node : node + 1], dms_states["qdot"][:, node : node + 1]))
        np.testing.assert_allclose(warm_start[(0, "states", node)], expected)


@pytest.mark.parametrize("node", [1, 2, 4, 5, 6])
def test_user_state_constraint_can_target_an_eliminated_node(node):
    constraint = Constraint(
        ConstraintFcn.BOUND_STATE,
        key="q",
        node=node,
        min_bound=[-1.0, -2.0],
        max_bound=[1.0, 2.0],
    )

    ocp = _prepare_block_ocp(constraints=constraint)
    user_constraint = next(penalty for penalty in ocp.nlp[0].g if penalty)

    assert user_constraint.node_idx == [node]
    assert user_constraint.function[node].size_out("val") == (2, 1)


@pytest.mark.parametrize("node", [1, 2, 4, 5, 6])
def test_user_state_objective_can_target_an_eliminated_node(node):
    objective = Objective(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", node=node)

    ocp = _prepare_block_ocp(objective_functions=objective)
    user_objective = next(penalty for penalty in ocp.nlp[0].J if penalty)

    assert user_objective.node_idx == [node]
    assert user_objective.function[node].size_out("val") == (2, 1)


@pytest.mark.parametrize("n_blocks", [1, 2, 6])
def test_block_shooting_solves_a_bounded_pendulum_problem(n_blocks):
    ocp = _prepare_solvable_block_ocp(n_blocks)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(100)

    solution = ocp.solve(solver)
    states = solution.decision_states(to_merge=SolutionMerge.NODES)

    assert solution.status == 0
    np.testing.assert_allclose(states["q"][:, 0], [0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(states["q"][:, -1], [0.0, 0.5], atol=1e-5)
    np.testing.assert_allclose(states["qdot"][:, [0, -1]], 0.0, atol=1e-5)


def test_dss_and_block_shooting_costs_match_the_historical_dms():
    costs = {}
    terminal_states = {}
    for n_blocks in (None, 1, 2, 6):
        ocp = _prepare_solvable_block_ocp(n_blocks=n_blocks)
        solver = Solver.IPOPT(show_online_optim=False)
        solver.set_maximum_iterations(100)
        solution = ocp.solve(solver)

        assert solution.status == 0
        costs[n_blocks] = float(solution.cost)
        states = solution.decision_states(to_merge=SolutionMerge.NODES)
        terminal_states[n_blocks] = np.vstack((states["q"][:, -1:], states["qdot"][:, -1:]))

    for n_blocks in (1, 2, 6):
        np.testing.assert_allclose(costs[n_blocks], costs[None], rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(terminal_states[n_blocks], terminal_states[None], atol=1e-5)


def test_block_solution_stepwise_and_reintegration_post_processing():
    ocp = _prepare_solvable_block_ocp(n_blocks=2)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(100)
    solution = ocp.solve(solver)

    stepwise_states = solution.stepwise_states(to_merge=SolutionMerge.NODES)
    reintegrated_states = solution.integrate(
        shooting_type=bioptim.Shooting.MULTIPLE,
        to_merge=SolutionMerge.NODES,
    )

    assert stepwise_states["q"].shape[0] == 2
    assert stepwise_states["qdot"].shape[0] == 2
    assert reintegrated_states["q"].shape[0] == 2
    assert reintegrated_states["qdot"].shape[0] == 2
    assert np.isfinite(stepwise_states["q"]).all()
    assert np.isfinite(reintegrated_states["q"]).all()


def test_fatrop_generic_nlp_dispatch_accepts_block_shooting():
    ocp = _prepare_solvable_block_ocp(n_blocks=2, ordering_strategy=OrderingStrategy.TIME_MAJOR)
    ocp.set_ocp_solver(Solver.FATROP())

    constraints, constraint_bounds = ocp.ocp_solver.dispatch_bounds()
    objective = ocp.ocp_solver.dispatch_obj_func()

    assert constraints.shape[0] == constraint_bounds.shape[0]
    assert constraints.shape[0] > 0
    assert objective.shape[0] > 0
    assert objective.shape[1] == 1


def test_continuous_phase_transition_uses_integrated_terminal_state():
    model_path = TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod"
    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.RK4()))
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.RK4()))

    ocp = OptimalControlProgram(
        [TorqueBiorbdModel(model_path), TorqueBiorbdModel(model_path)],
        n_shooting=[4, 6],
        phase_time=[0.5, 0.5],
        dynamics=dynamics,
        block_shooting=[BlockShooting.single(), BlockShooting(n_blocks=2)],
        use_sx=False,
    )
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(20)

    solution = ocp.solve(solver)
    states = solution.decision_states(to_merge=SolutionMerge.NODES)

    assert solution.status == 0
    assert [phase_states["q"].shape[1] for phase_states in states] == [5, 7]
    np.testing.assert_allclose(states[0]["q"][:, -1], states[1]["q"][:, 0], atol=1e-6)
    np.testing.assert_allclose(states[0]["qdot"][:, -1], states[1]["qdot"][:, 0], atol=1e-6)


def test_block_rollout_depends_on_an_optimized_dynamics_parameter():
    def set_gravity_z(model, parameter):
        gravity = casadi.MX.zeros(3, 1)
        gravity[2] = parameter.mx
        model.set_gravity(gravity)

    parameters = ParameterList(use_sx=False)
    parameters.add(
        "gravity_z",
        set_gravity_z,
        size=1,
        scaling=VariableScaling("gravity_z", np.array([1.0])),
    )
    model = TorqueBiorbdModel(
        TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod",
        parameters=parameters,
    )
    parameter_bounds = BoundsList()
    parameter_bounds.add(
        "gravity_z",
        min_bound=[-20.0],
        max_bound=[-1.0],
        interpolation=InterpolationType.CONSTANT,
    )
    parameter_init = InitialGuessList()
    parameter_init["gravity_z"] = [-9.81]
    x_init = InitialGuessList()
    x_init["q"] = [0.0, 0.2]

    ocp = OptimalControlProgram(
        model,
        n_shooting=6,
        phase_time=1.0,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4(), skip_continuity=True),
        x_init=x_init,
        parameters=parameters,
        parameter_bounds=parameter_bounds,
        parameter_init=parameter_init,
        block_shooting=BlockShooting(n_blocks=2),
        use_sx=False,
    )
    weak_gravity_vector = ocp.init_vector.copy()
    strong_gravity_vector = ocp.init_vector.copy()
    parameter_slice = ocp.vector_layout.index_map[("global", "parameters")][0]
    weak_gravity_vector[parameter_slice] = -2.0
    strong_gravity_vector[parameter_slice] = -15.0

    weak_gravity_rollout = ocp.nlp[0].state_rollout_function(weak_gravity_vector).full()
    strong_gravity_rollout = ocp.nlp[0].state_rollout_function(strong_gravity_vector).full()

    assert not np.allclose(weak_gravity_rollout[:, -1], strong_gravity_rollout[:, -1])


def test_state_penalties_cover_lagrange_terminal_marker_and_consecutive_nodes():
    objectives = bioptim.ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q")
    objectives.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", node=Node.END)
    objectives.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", node=2, derivative=True)
    constraints = Constraint(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=2,
        first_marker="marker_1",
        second_marker="marker_2",
    )

    ocp = _prepare_block_ocp(objective_functions=objectives, constraints=constraints)
    penalties = [penalty for penalty in ocp.nlp[0].J if penalty]
    marker_constraint = next(penalty for penalty in ocp.nlp[0].g if penalty)

    assert penalties[0].node_idx == list(range(6))
    assert penalties[1].node_idx == [6]
    assert penalties[2].node_idx == [2]
    assert marker_constraint.node_idx == [2]
    assert marker_constraint.function[2].size_out("val") == (3, 1)


@pytest.mark.parametrize("n_blocks", [1, 2, 3, 6])
def test_state_rollout_function_reconstructs_every_shooting_node(n_blocks):
    ocp = _prepare_block_ocp(n_shooting=6, block_shooting=BlockShooting(n_blocks=n_blocks))
    nlp = ocp.nlp[0]

    rollout = nlp.state_rollout_function(ocp.init_vector)

    assert rollout.shape == (nlp.states.shape, nlp.ns + 1)


@pytest.mark.parametrize("ordering", [OrderingStrategy.VARIABLE_MAJOR, OrderingStrategy.TIME_MAJOR])
def test_solution_decision_states_returns_the_full_block_rollout(ordering):
    ocp = _prepare_block_ocp(
        n_shooting=6,
        block_shooting=BlockShooting(n_blocks=2),
        ordering_strategy=ordering,
    )
    expected_scaled_states = ocp.nlp[0].state_rollout_function(ocp.init_vector).full()

    solution = Solution.from_vector(ocp, ocp.init_vector)
    reconstructed_states = solution.decision_states(scaled=True, to_merge=SolutionMerge.NODES)

    assert reconstructed_states["q"].shape == (2, 7)
    assert reconstructed_states["qdot"].shape == (2, 7)
    np.testing.assert_allclose(reconstructed_states["q"], expected_scaled_states[:2, :])
    np.testing.assert_allclose(reconstructed_states["qdot"], expected_scaled_states[2:, :])
