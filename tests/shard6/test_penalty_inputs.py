from types import SimpleNamespace

import numpy as np

from bioptim import ControlType, Node
from bioptim.limits.penalty_inputs import PenaltyInputProvider, PenaltyInputResolver


class _Weight:
    def evaluate_at(self, node_index, n_rows):
        assert node_index == 0
        assert n_rows == 2
        return 4.0


def _penalty():
    return SimpleNamespace(
        is_multinode_penalty=False,
        is_transition=False,
        phase=3,
        node_idx=[2],
        ns=[5],
        subnodes_are_decision_states=[False],
        control_types=[ControlType.CONSTANT],
        integrate=True,
        rows=np.array([0, 1]),
        weight=_Weight(),
        target=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )


def test_resolver_preserves_penalty_helper_selection_and_argument_order():
    calls = []

    def data_getter(name):
        def get_data(phase_idx, node_idx, subnodes):
            calls.append((name, phase_idx, node_idx, subnodes.start, subnodes.stop))
            return np.array([[phase_idx * 10 + node_idx]])

        return get_data

    inputs = PenaltyInputResolver.resolve(
        _penalty(),
        0,
        PenaltyInputProvider(
            time=lambda phase_idx, node_idx: phase_idx * 10 + node_idx,
            states=data_getter("x"),
            controls=data_getter("u"),
            parameters=lambda phase_idx, node_idx, _: np.array([[99.0]]),
            algebraic_states=data_getter("a"),
            numerical_timeseries=data_getter("d"),
        ),
        is_constructing_penalty=True,
        include_weight_and_target=True,
    )

    np.testing.assert_equal(inputs.x, np.array([[32], [32]]))
    np.testing.assert_equal(inputs.u, np.array([[32], [32]]))
    np.testing.assert_equal(inputs.a, np.array([[32], [32]]))
    np.testing.assert_equal(inputs.d, np.array([[32]]))
    np.testing.assert_equal(inputs.p, np.array([[99.0]]))
    assert inputs.t0 == 32

    function_arguments = inputs.function_arguments("phases_dt")
    assert function_arguments[0] == 32
    assert function_arguments[1] == "phases_dt"
    assert function_arguments[2] is inputs.x
    assert function_arguments[6] is inputs.d

    weighted_arguments = inputs.weighted_function_arguments("phases_dt")
    assert weighted_arguments[-2] == 4.0
    np.testing.assert_equal(weighted_arguments[-1], np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert calls == [
        ("x", 3, 2, Node.START, 1),
        ("x", 3, 2, Node.END, None),
        ("u", 3, 2, Node.START, 1),
        ("u", 3, 2, Node.END, None),
        ("a", 3, 2, Node.START, 1),
        ("a", 3, 2, Node.END, None),
        ("d", 3, 2, Node.START, 1),
    ]
