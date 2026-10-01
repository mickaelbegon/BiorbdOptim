from types import SimpleNamespace

import pytest

from bioptim import Node
from bioptim.limits.penalty_helpers import PenaltyHelpers, Slicy as LegacySlicy
from bioptim.limits.penalty_subnodes import Slicy, multinode_starting_indices, multinode_subnode_plan
from bioptim.misc.enums import PhaseDynamics


def multinode_penalty(**overrides):
    data = {
        "is_multinode_penalty": True,
        "is_transition": False,
        "nodes_phase": [0],
        "multinode_idx": [0],
        "subnodes_are_decision_states": [False],
        "phase_dynamics": [PhaseDynamics.SHARED_DURING_THE_PHASE],
        "ns": [3],
    }
    data.update(overrides)
    n_controllers = len(data["nodes_phase"])
    for field in ("subnodes_are_decision_states", "phase_dynamics", "ns"):
        if len(data[field]) == 1 and n_controllers > 1:
            data[field] *= n_controllers
    return SimpleNamespace(**data)


@pytest.mark.parametrize(
    "penalty,expected",
    [
        (
            multinode_penalty(nodes_phase=[0, 0, 1], multinode_idx=[0, 1, 2], ns=[3, 3, 3]),
            [0, 1, 0],
        ),
        (multinode_penalty(nodes_phase=[0, 0, 0], multinode_idx=[0, 1, 3], ns=[3, 3, 3]), [0, 1, -1]),
        (
            multinode_penalty(
                nodes_phase=[0, 0, 0],
                multinode_idx=[0, 1, 3],
                phase_dynamics=[PhaseDynamics.ONE_PER_NODE] * 3,
                ns=[3, 3, 3],
            ),
            [0, 0, 2],
        ),
    ],
)
def test_multinode_starting_indices_preserve_the_historical_allocation(penalty, expected):
    assert multinode_starting_indices(penalty) == expected
    assert PenaltyHelpers.get_multinode_penalty_subnodes_starting_index(penalty) == expected


@pytest.mark.parametrize(
    "penalty,is_constructing_penalty,expected",
    [
        (
            multinode_penalty(
                nodes_phase=[0, 0, 0],
                multinode_idx=[0, 1, 2],
                subnodes_are_decision_states=[True, False, False],
                ns=[3, 3, 3],
            ),
            True,
            [(Node.START, Node.END), (1, 2), (2, 3)],
        ),
        (
            multinode_penalty(
                nodes_phase=[0, 0, 0],
                multinode_idx=[0, 1, 2],
                subnodes_are_decision_states=[True, False, False],
                ns=[3, 3, 3],
            ),
            False,
            [(Node.START, Node.END), (1, 2), (Node.START, 1)],
        ),
        (multinode_penalty(multinode_idx=[3]), True, [(Node.END, None)]),
        (multinode_penalty(multinode_idx=[3]), False, [(Node.START, 1)]),
    ],
)
def test_multinode_subnode_plan_selects_symbolic_and_evaluation_subnodes(
    penalty, is_constructing_penalty, expected
):

    phases, nodes, subnodes = multinode_subnode_plan(penalty, is_constructing_penalty)

    assert phases == penalty.nodes_phase
    assert nodes == penalty.multinode_idx
    assert [(subnode.start, subnode.stop) for subnode in subnodes] == expected


def test_slicy_keeps_the_public_slice_contract():
    assert LegacySlicy is Slicy
    assert Slicy(0, Node.PENULTIMATE).start is Node.START
    assert Slicy(0, Node.PENULTIMATE).index() == slice(0, None)
