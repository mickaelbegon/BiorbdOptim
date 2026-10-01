from casadi import MX

from bioptim.limits.penalty_controller import PenaltyController


def test_copy_preserves_numerical_timeseries_and_node_index():
    controller = PenaltyController(
        ocp=object(),
        nlp=object(),
        t=[2],
        x=[MX.sym("x", 1, 1)],
        u=[MX.sym("u", 1, 1)],
        x_scaled=[MX.sym("x_scaled", 1, 1)],
        u_scaled=[MX.sym("u_scaled", 1, 1)],
        p=MX.sym("p", 1, 1),
        a=[MX.sym("a", 1, 1)],
        a_scaled=[MX.sym("a_scaled", 1, 1)],
        d=[MX.sym("d", 1, 1)],
        node_index=2,
    )

    copied = controller.copy()

    assert copied.node_index == 2
    assert copied.d == controller.d
    assert copied.t == controller.t
    assert str(copied.p) == str(controller.p)
