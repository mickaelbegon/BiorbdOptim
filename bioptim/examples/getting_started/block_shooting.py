"""Pendulum example using a partially condensed block-shooting transcription."""

from bioptim import (
    BlockShooting,
    BoundsList,
    DynamicsOptions,
    InitialGuessList,
    InterpolationType,
    Objective,
    ObjectiveFcn,
    OdeSolver,
    OptimalControlProgram,
    Solver,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float = 1.0,
    n_shooting: int = 20,
    n_blocks: int = 4,
) -> OptimalControlProgram:
    """Prepare a torque-driven pendulum with configurable block condensation."""
    model = TorqueBiorbdModel(biorbd_model_path)

    x_bounds = BoundsList()
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][1, -1] = 3.14
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0

    x_init = InitialGuessList()
    x_init.add(
        "q",
        initial_guess=[[0.0, 0.0], [0.0, 3.14]],
        interpolation=InterpolationType.LINEAR,
    )
    x_init["qdot"] = [0.0, 0.0]

    u_bounds = BoundsList()
    u_bounds["tau"] = [-100.0, 0.0], [100.0, 0.0]

    return OptimalControlProgram(
        model,
        n_shooting=n_shooting,
        phase_time=final_time,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4()),
        x_bounds=x_bounds,
        x_init=x_init,
        u_bounds=u_bounds,
        objective_functions=Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau"),
        block_shooting=BlockShooting(n_blocks=n_blocks),
        use_sx=False,
    )


def main():
    ocp = prepare_ocp(
        biorbd_model_path=ExampleUtils.folder + "/models/pendulum.bioMod",
        n_shooting=20,
        n_blocks=4,
    )
    solution = ocp.solve(Solver.IPOPT(show_online_optim=False))
    solution.print_cost()
    solution.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
