"""CasADi-backed penalty input providers used while building an OCP."""

from casadi import vertcat

from .penalty_inputs import PenaltyInputProvider
from .penalty_subnodes import Slicy
from ..misc.enums import ControlType, Node


class OcpPenaltyInputProviderFactory:
    """Create providers that read scaled symbolic data from an OCP.

    The resolver owns node and subnode selection. This factory only translates
    the selected locations into the OCP's CasADi storage.
    """

    @classmethod
    def build(cls, penalty, ocp) -> PenaltyInputProvider:
        """Build the storage callbacks used while compiling a penalty."""

        return PenaltyInputProvider(
            time=lambda phase_idx, node_idx: ocp.node_time(phase_idx=phase_idx, node_idx=node_idx),
            states=lambda phase_idx, node_idx, subnodes: cls._states(
                ocp, ocp.nlp[phase_idx].states, phase_idx, node_idx, subnodes
            ),
            controls=lambda phase_idx, node_idx, subnodes: cls._controls(
                penalty, ocp, phase_idx, node_idx, subnodes
            ),
            parameters=lambda phase_idx, node_idx, subnodes: ocp.parameters.scaled.cx_start,
            algebraic_states=lambda phase_idx, node_idx, subnodes: cls._states(
                ocp, ocp.nlp[phase_idx].algebraic_states, phase_idx, node_idx, subnodes
            ),
            numerical_timeseries=lambda phase_idx, node_idx, subnodes: cls._numerical_timeseries(
                ocp, phase_idx, node_idx, subnodes
            ),
        )

    @staticmethod
    def _states(ocp, states, phase_idx: int, node_idx: int, subnodes: Slicy):
        states.node_index = node_idx

        values = ocp.cx()
        if states.scaled.cx_start.shape == (0, 0):
            return values

        if subnodes.start == Node.START:
            values = vertcat(values, states.scaled.cx_start)
            if subnodes.stop == 1:
                pass
            elif subnodes.stop == Node.PENULTIMATE:
                if node_idx < ocp.nlp[phase_idx].ns + 1:
                    values = vertcat(values, vertcat(*states.scaled.cx_intermediates_list))
            elif subnodes.stop == Node.END:
                if node_idx < ocp.nlp[phase_idx].ns + 1:
                    values = vertcat(
                        vertcat(values, vertcat(*states.scaled.cx_intermediates_list)), states.scaled.cx_end
                    )
            else:
                raise ValueError("The sn_idx.stop should be 1 or None if sn_idx.start == 0")

        elif subnodes.start == 1:
            if subnodes.stop == 2:
                values = vertcat(values, vertcat(states.scaled.cx_mid))
            else:
                raise ValueError("The sn_idx.stop should be 2 if sn_idx.start == 1")

        elif subnodes.start == 2:
            if subnodes.stop == 3:
                values = vertcat(values, vertcat(states.scaled.cx_end))
            else:
                raise ValueError("The sn_idx.stop should be 3 if sn_idx.start == 2")

        elif subnodes.start == Node.END:
            values = vertcat(values, vertcat(states.scaled.cx_end))
            if subnodes.stop is not None:
                raise ValueError("The sn_idx.stop should be None if sn_idx.start == -1")

        else:
            raise ValueError(f"The sn_idx.start {subnodes.start} not recognized.")

        return values

    @staticmethod
    def _controls(penalty, ocp, phase_idx: int, node_idx: int, subnodes: Slicy):
        nlp = ocp.nlp[phase_idx]
        controls = nlp.controls
        controls.node_index = node_idx

        values = ocp.cx()

        def append_end():
            if nlp.control_type in (ControlType.LINEAR_CONTINUOUS,):
                return vertcat(values, controls.scaled.cx_end)
            elif nlp.control_type in (ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE):
                if node_idx < nlp.n_controls_nodes - 1:
                    return vertcat(values, controls.scaled.cx_end)

                if node_idx == nlp.n_controls_nodes - 1:
                    # At the penultimate node, cx_end is available except for an
                    # integration, derivative, or multinode penalty.
                    if nlp.control_type in (ControlType.CONSTANT_WITH_LAST_NODE,):
                        return vertcat(values, controls.scaled.cx_end)
                    if (
                        penalty.integrate
                        or penalty.derivative
                        or penalty.explicit_derivative
                        or penalty.is_multinode_penalty
                    ):
                        return values
                    return vertcat(values, controls.scaled.cx_end)

                return values
            else:
                raise NotImplementedError(f"Control type {nlp.control_type} not implemented yet")

        if subnodes.start == Node.START:
            values = vertcat(values, controls.scaled.cx_start)
            if subnodes.stop == 1:
                pass
            elif subnodes.stop == Node.PENULTIMATE or subnodes.stop == Node.END:
                values = append_end()
            else:
                raise ValueError(f"The sn_idx.stop {subnodes.stop} was not recognized.")

        elif subnodes.start == 1:
            if subnodes.stop == 2:
                values = vertcat(values, controls.scaled.cx_mid)
            else:
                raise ValueError(f"The sn_idx [{subnodes.start}, {subnodes.stop}] was not recognized.")

        elif subnodes.start == 2:
            # This is not the actual endpoint but a midpoint that must use cx_end.
            if subnodes.stop == 3:
                values = vertcat(values, controls.scaled.cx_end)
            else:
                raise ValueError(f"The sn_idx [{subnodes.start}, {subnodes.stop}] was not recognized.")

        elif subnodes.start == Node.END:
            if subnodes.stop is not None:
                raise ValueError(f"The sn_idx [{subnodes.start}, {subnodes.stop}] was not recognized.")
            values = append_end()

        else:
            raise ValueError(f"The sn_idx.start {subnodes.start} not recognized.")

        return values

    @staticmethod
    def _numerical_timeseries(ocp, phase_idx: int, node_idx: int, subnodes: Slicy):
        numerical_timeseries = ocp.nlp[phase_idx].numerical_timeseries

        if numerical_timeseries.cx_start.shape == (0, 0):
            return ocp.cx()
        if subnodes.start == Node.START:
            return numerical_timeseries.cx_start
        if subnodes.start == Node.END:
            return numerical_timeseries.cx_end
        raise ValueError(f"The sn_idx [{subnodes.start}, {subnodes.stop}] was not recognized.")
