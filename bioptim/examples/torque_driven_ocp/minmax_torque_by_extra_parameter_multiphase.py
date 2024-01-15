"""
This example is inspired from the clear pike circle gymnastics skill. It is composed of two pendulums
representing the trunk and legs segments (only the hip flexion is actuated). The objective is to minimize the
extreme torque (minmax) of the hip flexion while performing the clear pike circle motion. The extreme torques are included to the
problem as parameters, all torque intervals are constrained to be smaller than this parameter, these parameters are
minimized.
"""

import numpy as np
import biorbd_casadi as biorbd
from casadi import MX
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    Node,
    ObjectiveFcn,
    BiMappingList,
    ParameterList,
    InterpolationType,
    BiorbdModel,
    PenaltyController,
    ParameterObjectiveList,
)


def custom_constraint_max_tau(controller: PenaltyController) -> MX:
    return controller.parameters["max_tau"].cx - controller.controls["tau"].cx


def custom_constraint_min_tau(controller: PenaltyController) -> MX:
    return controller.parameters["min_tau"].cx - controller.controls["tau"].cx


def my_parameter_function(bio_model: biorbd.Model, value: MX):
    return


def prepare_ocp(
    bio_model_path: str = "models/double_pendulum.bioMod",
) -> OptimalControlProgram:
    bio_model = (BiorbdModel(bio_model_path), BiorbdModel(bio_model_path))

    # Problem parameters
    n_shooting = (30, 30)
    final_time = (2, 3)
    tau_min, tau_max, tau_init = -40, 40, 0

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)

    # Mapping
    tau_mappings = BiMappingList()
    tau_mappings.add("tau", to_second=[None, 0], to_first=[1])
    tau_mappings.add("tau", to_second=[None, 0], to_first=[1])

    # Define the parameter to optimize
    parameters = ParameterList()
    parameter_init = InitialGuessList()
    parameter_bounds = BoundsList()

    parameters.add(
        "max_tau",
        my_parameter_function,
        size=1,
    )
    parameters.add(
        "min_tau",
        my_parameter_function,
        size=1,
    )

    parameter_init["max_tau"] = 1
    parameter_init["min_tau"] = -1

    parameter_bounds.add("max_tau", min_bound=0, max_bound=tau_max, interpolation=InterpolationType.CONSTANT)
    parameter_bounds.add("min_tau", min_bound=tau_min, max_bound=0, interpolation=InterpolationType.CONSTANT)

    # Add phase independent objective functions
    parameter_objectives = ParameterObjectiveList()
    parameter_objectives.add(ObjectiveFcn.Parameter.MINIMIZE_PARAMETER, key="max_tau", weight=100, quadratic=False)
    parameter_objectives.add(ObjectiveFcn.Parameter.MINIMIZE_PARAMETER, key="min_tau", weight=-100, quadratic=False)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=0, min_bound=0.5, max_bound=3)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=1, min_bound=0.5, max_bound=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=1)

    # Constraints
    constraints = ConstraintList()
    constraints.add(custom_constraint_max_tau, phase=0, node=Node.ALL_SHOOTING, min_bound=0, max_bound=tau_max)
    constraints.add(custom_constraint_min_tau, phase=0, node=Node.ALL_SHOOTING, min_bound=tau_min, max_bound=0)
    #
    constraints.add(custom_constraint_max_tau, phase=1, node=Node.ALL_SHOOTING, min_bound=0, max_bound=tau_max)
    constraints.add(custom_constraint_min_tau, phase=1, node=Node.ALL_SHOOTING, min_bound=tau_min, max_bound=0)

    constraints.add(
        ConstraintFcn.TRACK_STATE, key="q", phase=0, node=Node.START, target=[np.pi + 0.01, 0]
    )  # Initial state
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", phase=0, node=Node.START, target=[0, 0])
    constraints.add(ConstraintFcn.TRACK_STATE, key="q", index=1, phase=1, node=Node.START, target=np.pi)
    constraints.add(ConstraintFcn.TRACK_STATE, key="q", phase=1, node=Node.END, target=[3 * np.pi, 0])  # Final state

    for i in range(len(bio_model)):
        constraints.add(
            ConstraintFcn.BOUND_STATE,
            key="q",
            phase=i,
            node=Node.ALL,
            min_bound=[np.pi, -np.pi / 3],
            max_bound=[3 * np.pi, np.pi],
        ),

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        objective_functions=objective_functions,
        parameter_objectives=parameter_objectives,
        parameter_bounds=parameter_bounds,
        parameter_init=parameter_init,
        constraints=constraints,
        variable_mappings=tau_mappings,
        parameters=parameters,
    )


def main():
    # --- Prepare the ocp --- #
    ocp = prepare_ocp()

    # --- Solve the ocp --- #
    sol = ocp.solve()

    # --- Show results --- #
    sol.animate()
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
