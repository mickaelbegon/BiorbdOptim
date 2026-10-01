from typing import Protocol

from ..misc.enums import Node, PhaseDynamics


class Slicy:
    """A slice descriptor for penalty subnodes.

    A request ending at ``Node.END`` uses the symbolic end state while a
    penalty is constructed. During numerical evaluation the caller instead
    requests the start of the following node. When subnodes are decision
    states, a start-to-end request includes all intermediate states.
    """

    def __init__(self, start: int | Node, stop: int | None | Node):
        self.start = Node.START if start == 0 else start
        self.stop = stop

    def index(self) -> slice:
        """Return the corresponding Python slice."""

        start = 0 if self.start == Node.START else self.start
        stop = None if self.stop in (Node.END, Node.PENULTIMATE) else self.stop
        return slice(start, stop)


class MultinodePenaltyProtocol(Protocol):
    """Structural information needed to plan multinode penalty inputs."""

    is_multinode_penalty: bool
    is_transition: bool
    nodes_phase: list[int]
    multinode_idx: list[int]
    subnodes_are_decision_states: list[bool]
    phase_dynamics: list[PhaseDynamics]
    ns: list[int]


def multinode_starting_indices(penalty: MultinodePenaltyProtocol) -> list[int]:
    """Choose the symbolic input slot used by each multinode controller."""

    out = []
    share_phase_nodes = {}
    for phase_idx, node_idx, phase_dynamics, ns in zip(
        penalty.nodes_phase, penalty.multinode_idx, penalty.phase_dynamics, penalty.ns
    ):
        if phase_idx not in share_phase_nodes:
            share_phase_nodes[phase_idx] = {"nodes_used": [], "available_cx": [0, 1, 2]}

        if not share_phase_nodes[phase_idx]["available_cx"]:
            raise ValueError(
                "Valid values for setting the cx is 0, 1 or 2. If you reach this error message, you probably tried "
                "to add more penalties than available in a multinode constraint. You can try to split the "
                "constraints into more penalties or use phase_dynamics=PhaseDynamics.ONE_PER_NODE"
            )

        if node_idx in share_phase_nodes[phase_idx]["nodes_used"]:
            raise ValueError("It is not possible to constraints the same node twice")
        share_phase_nodes[phase_idx]["nodes_used"].append(node_idx)

        is_last_node = node_idx == ns
        if phase_dynamics == PhaseDynamics.ONE_PER_NODE:
            out.append(2 if is_last_node else 0)
            continue

        next_idx = share_phase_nodes[phase_idx]["available_cx"].pop(-1 if is_last_node else 0)
        out.append(-1 if is_last_node else next_idx)

    return out


def multinode_subnode_plan(
    penalty: MultinodePenaltyProtocol, is_constructing_penalty: bool
) -> tuple[list[int], list[int], list[Slicy]]:
    """Return the phase, node, and subnode requests for a multinode penalty."""

    if not penalty.is_multinode_penalty:
        raise RuntimeError("This function should only be called for multinode penalties")

    phases = penalty.nodes_phase
    nodes = penalty.multinode_idx
    subnodes = []
    for index, starting in enumerate(multinode_starting_indices(penalty)):
        if starting < 0:
            subnodes.append(Slicy(Node.END, None) if is_constructing_penalty else Slicy(Node.START, 1))
        elif starting == 2:
            subnodes.append(Slicy(2, 3) if is_constructing_penalty else Slicy(Node.START, 1))
        elif penalty.subnodes_are_decision_states[index] and not penalty.is_transition:
            subnodes.append(
                Slicy(Node.START, 1) if nodes[index] >= penalty.ns[index] else Slicy(Node.START, Node.END)
            )
        else:
            subnodes.append(Slicy(starting, starting + 1))

    return phases, nodes, subnodes
