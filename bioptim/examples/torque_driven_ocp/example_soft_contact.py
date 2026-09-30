"""
A very simple optimal control program playing with a soft contact sphere rolling going from one point to another.

The soft contact sphere are hard to make converge and sensitive to parameters.
"""

import numpy as np
from casadi import Function, MX, rootfinder
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    OdeSolverBase,
    Node,
    Solver,
    SoftContactDynamics,
    PhaseDynamics,
    SolutionMerge,
    ContactType,
)


def prepare_single_shooting(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    ode_solver: OdeSolverBase,
    n_threads: int = 1,
    use_sx: bool = False,
) -> OptimalControlProgram:
    """
    Prepare the ss

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Dynamics
    dynamics = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN,
        soft_contacts_dynamics=SoftContactDynamics.ODE,
        contact_type=[ContactType.SOFT_EXPLICIT],
        ode_solver=ode_solver,
    )

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        use_sx=use_sx,
        n_threads=n_threads,
    )


def initial_states_from_static_equilibrium(model):
    """
    Compute the static initial state of the rolling sphere.

    The former initialization integrated a transient drop for the duration of
    the OCP. Its result therefore varied with the horizon and could conflict
    with the hard initial marker constraint. Solving the vertical acceleration
    directly gives a reproducible equilibrium instead.
    """

    bio_model = BiorbdModel(model)
    vertical_dof = [i for i, name in enumerate(bio_model.name_dof) if name.endswith("TransZ")]
    if len(vertical_dof) != 1:
        raise RuntimeError("The soft-contact example requires exactly one vertical translational degree of freedom.")

    qz = MX.sym("qz")
    q = MX.zeros(bio_model.nb_q, 1)
    q[vertical_dof[0]] = qz
    qdot = MX.zeros(bio_model.nb_qdot, 1)
    tau = MX.zeros(bio_model.nb_tau, 1)
    vertical_acceleration = Function(
        "soft_contact_static_vertical_acceleration",
        [qz],
        [bio_model.forward_dynamics()(q, qdot, tau, [], bio_model.parameters)[vertical_dof[0]]],
    ).expand()
    equilibrium_solver = rootfinder("soft_contact_static_equilibrium", "newton", vertical_acceleration)

    # A unilateral contact has exactly zero force and derivative out of contact.
    # Start Newton from the plane so that it is initialized in penetration.
    qz_equilibrium = float(equilibrium_solver(0.0))

    if not np.isfinite(qz_equilibrium) or abs(float(vertical_acceleration(qz_equilibrium))) > 1e-8:
        raise RuntimeError("Could not find a static equilibrium for the soft-contact example.")

    x = InitialGuessList()
    q_initial = np.zeros(bio_model.nb_q)
    q_initial[vertical_dof[0]] = qz_equilibrium
    x["q"] = q_initial
    x["qdot"] = np.zeros(bio_model.nb_qdot)

    return x


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    ode_solver: OdeSolverBase,
    slack: float = 1e-4,
    n_threads: int = 8,
    use_sx: bool = False,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """
    bio_model = BiorbdModel(biorbd_model_path)

    # Problem parameters

    tau_min, tau_max = -100, 100

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_SOFT_CONTACT_FORCES, weight=0.0001)
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
        node=Node.START,
        first_marker="marker_point",
        second_marker="start",
        weight=10,
        axes=2,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="marker_point",
        second_marker="end",
        weight=10,
    )

    # Dynamics
    dynamics = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN,
        soft_contacts_dynamics=SoftContactDynamics.ODE,
        contact_type=[ContactType.SOFT_EXPLICIT],
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="marker_point", second_marker="start"
    )
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="marker_point", second_marker="end")

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")

    init = initial_states_from_static_equilibrium(biorbd_model_path)
    x_bounds["q"].min[:, 0] = (init["q"].init - slack)[:, 0]
    x_bounds["q"].max[:, 0] = (init["q"].init + slack)[:, 0]

    x_bounds["qdot"].min[:, 0] = -slack
    x_bounds["qdot"].max[:, 0] = slack

    # Initial guess
    x_init = InitialGuessList()
    x_init["q"] = init["q"]
    x_init["qdot"] = init["qdot"]

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objective_functions,
        constraints=constraints,
        use_sx=use_sx,
        n_threads=n_threads,
    )


def main():
    """
    Defines a multiphase ocp and animate the results
    """
    biorbd_model_path = "../torque_driven_ocp/models/soft_contact_sphere.bioMod"
    ode_solver = OdeSolver.RK8()

    # Prepare OCP to reach the second marker
    ocp = prepare_ocp(biorbd_model_path, 37, 0.3, ode_solver, slack=1e-4)
    # ocp.add_plot_penalty(CostType.ALL)
    # ocp.print(to_graph=True)

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    sol = ocp.solve(solver)

    sol.print_cost()
    sol.graphs()

    # --- Show results --- #
    viewer = "pyorerun"
    if viewer == "pyorerun":
        from pyorerun import BiorbdModel, PhaseRerun

        # Model
        model = BiorbdModel(biorbd_model_path)
        model.options.transparent_mesh = False
        model.options.show_gravity = True
        model.options.show_floor = True

        # Visualization
        time = sol.decision_time(to_merge=SolutionMerge.NODES)
        q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"]
        viz = PhaseRerun(time)
        viz.add_animated_model(model, q)
        viz.rerun_by_frame("Optimal solution")
    else:
        sol.animate()


if __name__ == "__main__":
    main()
