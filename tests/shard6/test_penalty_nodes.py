import pytest

from bioptim import Node
from bioptim.limits.penalty_nodes import PenaltyNodeResolver


@pytest.mark.parametrize(
    "nodes,n_shooting,expected",
    [
        (Node.START, 6, (0,)),
        (Node.MID, 6, (3,)),
        (Node.INTERMEDIATES, 6, (1, 2, 3, 4)),
        (Node.PENULTIMATE, 6, (5,)),
        (Node.END, 6, (6,)),
        (Node.ALL_SHOOTING, 3, (0, 1, 2)),
        (Node.ALL, 3, (0, 1, 2, 3)),
        ((Node.START, 2, Node.END), 6, (0, 2, 6)),
    ],
)
def test_resolve_penalty_nodes(nodes, n_shooting, expected):
    plan = PenaltyNodeResolver.resolve(nodes, n_shooting)

    expected_nodes = (nodes,) if isinstance(nodes, Node) else nodes
    assert plan.requested_nodes == expected_nodes
    assert plan.indices == expected


def test_resolve_penalty_nodes_rejects_invalid_index():
    with pytest.raises(RuntimeError, match="between 0 and 3"):
        PenaltyNodeResolver.resolve(4, 3)


def test_resolve_penalty_nodes_rejects_mid_for_odd_shooting_count():
    with pytest.raises(ValueError, match="must be even"):
        PenaltyNodeResolver.resolve(Node.MID, 3)
