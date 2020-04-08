import biorbd

from biorbd_optim import OptimalControlProgram
from biorbd_optim.problem_type import ProblemType
from biorbd_optim.mapping import Mapping
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.constraints import Constraint
from biorbd_optim.path_conditions import Bounds, QAndQDotBounds, InitialConditions


def prepare_ocp(biorbd_model_path="eocar.bioMod", show_online_optim=False):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    number_shooting_points = 30
    final_time = 2
    torque_min, torque_max, torque_init = -100, 100, 0
    dof_mapping = Mapping([0, 1, 2, 1, 2], [0, 1, 2], [3, 4])

    # Add objective functions
    objective_functions = ((ObjectiveFunction.minimize_torque, 100),)

    # Dynamics
    variable_type = ProblemType.torque_driven

    # Constraints
    constraints = (
        (Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.START, (0, 1)),
        (Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.END, (0, 2)),
    )

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model, dof_mapping)
    for i in range(1, 6):
        X_bounds.first_node_min[i] = 0
        X_bounds.last_node_min[i] = 0
        X_bounds.first_node_max[i] = 0
        X_bounds.last_node_max[i] = 0
    X_bounds.last_node_min[2] = 1.57
    X_bounds.last_node_max[2] = 1.57

    # Initial guess
    X_init = InitialConditions([0] * dof_mapping.nb_reduced * 2)

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * dof_mapping.nb_reduced, [torque_max] * dof_mapping.nb_reduced,
    )
    U_init = InitialConditions([torque_init] * dof_mapping.nb_reduced)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        variable_type,
        number_shooting_points,
        final_time,
        objective_functions,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        constraints,
        dof_mapping=dof_mapping,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(show_online_optim=True)

    # --- Solve the program --- #
    sol = ocp.solve()