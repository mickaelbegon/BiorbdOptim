from casadi import MX, vertcat, Function

from .dynamics import Dynamics
from .mapping import BidirectionalMapping, Mapping
from .plot import CustomPlot
from .enums import PlotType


class ProblemType:
    """
    Includes methods suitable for several situations
    """

    @staticmethod
    def torque_driven(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques but without muscles, must be used with dynamics without contacts.
        :param nlp: An instance of the OptimalControlProgram class.
        """
        ProblemType.__configure_torque_driven(nlp)
        ProblemType.__configure_forward_dyn_func(nlp, Dynamics.forward_dynamics_torque_driven)

    @staticmethod
    def torque_driven_with_contact(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques, without muscles, must be used with dynamics with contacts.
        :param nlp: An OptimalControlProgram class.
        """

        ProblemType.__configure_torque_driven(nlp)
        ProblemType.__configure_forward_dyn_func(nlp, Dynamics.forward_dynamics_torque_driven_with_contact)
        ProblemType.__configure_contact(nlp, Dynamics.forces_from_forward_dynamics_with_contact)

    @staticmethod
    def muscle_activations_and_torque_driven(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        ProblemType.__configure_torque_driven(nlp)
        ProblemType.__configure_muscles(nlp, False, True)

        u = MX()
        for i in range(nlp["nbMuscle"]):
            u = vertcat(u, MX.sym(f"Muscle_{nlp['muscleNames']}_activation"))
        nlp["u"] = vertcat(nlp["u"], u)
        nlp["nu"] = nlp["u"].rows()
        nlp["var_controls"]["muscles"] = nlp["nbMuscle"]

        ProblemType.__configure_forward_dyn_func(nlp, Dynamics.forward_dynamics_torque_muscle_driven)

    @staticmethod
    def muscle_excitations_and_torque_driven(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        ProblemType.__configure_torque_driven(nlp)
        ProblemType.__configure_muscles(nlp, True, True)

        u = MX()
        x = MX()
        for i in range(nlp["nbMuscle"]):
            u = vertcat(u, MX.sym(f"Muscle_{nlp['muscleNames']}_excitation"))
            x = vertcat(x, MX.sym(f"Muscle_{nlp['muscleNames']}_activation"))
        nlp["u"] = vertcat(nlp["u"], u)
        nlp["x"] = vertcat(nlp["x"], x)
        nlp["var_states"]["muscles"] = nlp["nbMuscle"]
        nlp["var_controls"]["muscles"] = nlp["nbMuscle"]

        ProblemType.__configure_forward_dyn_func(nlp, Dynamics.forward_dynamics_muscle_excitations_and_torque_driven)

    @staticmethod
    def muscles_and_torque_driven_with_contact(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        ProblemType.__configure_torque_driven(nlp)
        ProblemType.__configure_muscles(nlp, False, True)

        u = MX()
        for i in range(nlp["nbMuscle"]):
            u = vertcat(u, MX.sym(f"Muscle_{nlp['muscleNames']}_activation"))
        nlp["u"] = vertcat(nlp["u"], u)
        nlp["var_controls"]["muscles"] = nlp["nbMuscle"]

        ProblemType.__configure_forward_dyn_func(nlp, Dynamics.forward_dynamics_torque_muscle_driven_with_contact)
        ProblemType.__configure_contact(nlp, Dynamics.forces_from_forward_dynamics_torque_muscle_driven_with_contact)

    @staticmethod
    def muscle_excitations_and_torque_driven_with_contact(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        ProblemType.__configure_torque_driven(nlp)
        ProblemType.__configure_muscles(nlp, True, True)

        u = MX()
        x = MX()
        for i in range(nlp["nbMuscle"]):
            u = vertcat(u, MX.sym(f"Muscle_{nlp['muscleNames']}_excitation"))
            x = vertcat(x, MX.sym(f"Muscle_{nlp['muscleNames']}_activation"))
        nlp["u"] = vertcat(nlp["u"], u)
        nlp["x"] = vertcat(nlp["x"], x)
        nlp["var_states"]["muscles"] = nlp["nbMuscle"]
        nlp["var_controls"]["muscles"] = nlp["nbMuscle"]

        ProblemType.__configure_forward_dyn_func(
            nlp, Dynamics.forward_dynamics_muscle_excitations_and_torque_driven_with_contact
        )
        ProblemType.__configure_contact(
            nlp, Dynamics.forces_from_forward_dynamics_muscle_excitations_and_torque_driven_with_contact
        )

    @staticmethod
    def __configure_torque_driven(nlp):
        """
        Configures common settings for torque driven problems with and without contacts.
        :param nlp: An OptimalControlProgram class.
        """
        if nlp["q_mapping"] is None:
            nlp["q_mapping"] = BidirectionalMapping(
                Mapping(range(nlp["model"].nbQ())), Mapping(range(nlp["model"].nbQ()))
            )
        if nlp["q_dot_mapping"] is None:
            nlp["q_dot_mapping"] = BidirectionalMapping(
                Mapping(range(nlp["model"].nbQdot())), Mapping(range(nlp["model"].nbQdot()))
            )
        if nlp["tau_mapping"] is None:
            nlp["tau_mapping"] = BidirectionalMapping(
                Mapping(range(nlp["model"].nbGeneralizedTorque())), Mapping(range(nlp["model"].nbGeneralizedTorque()))
            )

        dof_names = nlp["model"].nameDof()
        q = MX()
        q_dot = MX()
        for i in nlp["q_mapping"].reduce.map_idx:
            q = vertcat(q, MX.sym("Q_" + dof_names[i].to_string(), 1, 1))
        for i in nlp["q_dot_mapping"].reduce.map_idx:
            q_dot = vertcat(q_dot, MX.sym("Qdot_" + dof_names[i].to_string(), 1, 1))
        nlp["x"] = vertcat(q, q_dot)

        u = MX()
        for i in nlp["tau_mapping"].reduce.map_idx:
            u = vertcat(u, MX.sym("Tau_" + dof_names[i].to_string(), 1, 1))
        nlp["u"] = u

        nlp["nbQ"] = nlp["q_mapping"].reduce.len
        nlp["nbQdot"] = nlp["q_dot_mapping"].reduce.len
        nlp["nbTau"] = nlp["tau_mapping"].reduce.len

        nlp["var_states"] = {"q": nlp["q_mapping"].reduce.len, "q_dot": nlp["q_dot_mapping"].reduce.len}
        nlp["var_controls"] = {"tau": nlp["tau_mapping"].reduce.len}

        legend_q = ["q_" + nlp["model"].nameDof()[idx].to_string() for idx in nlp["q_mapping"].reduce.map_idx]
        legend_qdot = ["qdot_" + nlp["model"].nameDof()[idx].to_string() for idx in nlp["q_dot_mapping"].reduce.map_idx]
        legend_tau = ["tau_" + nlp["model"].nameDof()[idx].to_string() for idx in nlp["tau_mapping"].reduce.map_idx]
        nlp["plot"] = {
            "q": CustomPlot(lambda x, u: x[: nlp["nbQ"]], plot_type=PlotType.INTEGRATED, legend=legend_q),
            "q_dot": CustomPlot(
                lambda x, u: x[nlp["nbQ"] : nlp["nbQ"] + nlp["nbQdot"]],
                plot_type=PlotType.INTEGRATED,
                legend=legend_qdot,
            ),
        }
        nlp["plot"]["tau"] = CustomPlot(lambda x, u: u[: nlp["nbTau"]], plot_type=PlotType.STEP, legend=legend_tau)

    @staticmethod
    def __configure_contact(nlp, dyn_func):
        symbolic_states = MX.sym("x", nlp["nx"], 1)
        symbolic_controls = MX.sym("u", nlp["nu"], 1)
        nlp["contact_forces_func"] = Function(
            "contact_forces_func",
            [symbolic_states, symbolic_controls],
            [dyn_func(symbolic_states, symbolic_controls, nlp)],
            ["x", "u"],
            ["contact_forces"],
        ).expand()

        nlp["nbContact"] = nlp["model"].nbContacts()
        contact_names = [n.to_string() for n in nlp["model"].contactNames()]
        phase_mappings = nlp["plot_mappings"]["contact_forces"] if "contact_forces" in nlp["plot_mappings"] else None
        nlp["plot"]["contact_forces"] = CustomPlot(
            nlp["contact_forces_func"], axes_idx=phase_mappings, legend=contact_names
        )

    @staticmethod
    def __configure_muscles(nlp, muscles_are_states=False, muscles_are_controls=False):
        nlp["nbMuscle"] = nlp["model"].nbMuscles()
        nlp["muscleNames"] = [names.to_string() for names in nlp["model"].muscleNames()]

        combine = None
        if muscles_are_states:
            nx_q = nlp["nbQ"] + nlp["nbQdot"]
            nlp["plot"]["muscles_states"] = CustomPlot(
                lambda x, u: x[nx_q : nx_q + nlp["nbMuscle"]], plot_type=PlotType.INTEGRATED, legend=nlp["muscleNames"]
            )
            combine = "muscles_states"
        if muscles_are_controls:
            nlp["plot"]["muscles_control"] = CustomPlot(
                lambda x, u: u[nlp["nbTau"] : nlp["nbTau"] + nlp["nbMuscle"]],
                plot_type=PlotType.STEP,
                legend=nlp["muscleNames"],
                combine_to=combine,
            )

    @staticmethod
    def __configure_forward_dyn_func(nlp, dyn_func):
        nlp["nu"] = nlp["u"].rows()
        nlp["nx"] = nlp["x"].rows()

        symbolic_states = MX.sym("x", nlp["nx"], 1)
        symbolic_controls = MX.sym("u", nlp["nu"], 1)
        nlp["dynamics_func"] = Function(
            "ForwardDyn",
            [symbolic_states, symbolic_controls],
            [dyn_func(symbolic_states, symbolic_controls, nlp)],
            ["x", "u"],
            ["xdot"],
        ).expand()  # .map(nlp["ns"], "thread", 2)
