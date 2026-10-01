from types import SimpleNamespace

from casadi import DM

from bioptim import (
    MultinodeObjective,
    MultinodeObjectiveFcn,
    Node,
    ObjectiveWeight,
    PhaseTransition,
    PhaseTransitionFcn,
)
from bioptim.misc.enums import ControlType, PhaseDynamics


def controller_stub():
    nlp = SimpleNamespace(
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        ns=3,
        control_type=ControlType.CONSTANT,
    )
    return SimpleNamespace(get_nlp=nlp, cx_index_to_get=None)


def custom_function(calls):
    def custom(controllers, token):
        calls.append((controllers, token))
        return DM([17])

    return custom


def test_multinode_custom_kernel_preserves_controller_order_and_parameters():
    calls = []
    custom = custom_function(calls)
    penalty = MultinodeObjective(
        MultinodeObjectiveFcn,
        nodes=(Node.END, Node.START),
        nodes_phase=(0, 1),
        multinode_penalty=custom,
        weight=ObjectiveWeight(),
    )
    penalty.multinode_idx = [3, 0]
    controllers = [controller_stub(), controller_stub()]

    result = penalty.type.kernel.evaluate(penalty, controllers, token="sentinel")

    assert penalty.type is MultinodeObjectiveFcn.CUSTOM
    assert penalty.custom_function is custom
    assert calls == [(controllers, "sentinel")]
    assert float(result) == 17
    assert [controller.cx_index_to_get for controller in controllers] == [-1, 0]


def test_transition_custom_kernel_preserves_controller_order_and_parameters():
    calls = []
    custom = custom_function(calls)
    transition = PhaseTransition(0, custom, token="sentinel")
    transition.multinode_idx = [3, 0]
    controllers = [controller_stub(), controller_stub()]

    result = transition.type.kernel.evaluate(transition, controllers, token=transition.extra_arguments["token"])

    assert transition.type is PhaseTransitionFcn.CUSTOM
    assert transition.custom_function is custom
    assert transition.nodes == (Node.END, Node.START)
    assert transition.nodes_phase == (0, 1)
    assert calls == [(controllers, "sentinel")]
    assert float(result) == 17
    assert [controller.cx_index_to_get for controller in controllers] == [-1, 0]
