import numpy as np
from casadi import DM, MX, vertcat

from bioptim.interfaces.post_shake_penalty_registry import PostShakePenaltyRegistry


def test_post_shake_registry_localizes_exact_value_jacobian_hessian_and_global_g_rows():
    """The registry exposes an exact local view while retaining canonical g rows."""

    v = MX.sym("v", 4, 1)
    registry = PostShakePenaltyRegistry()
    # Deliberately record stage 2 first: canonical g concatenates stage -1
    # before stage 2, so this checks that rows are not merely insertion order.
    registry.record(
        v[1] ** 2 + 2 * v[3],
        kind="constraint",
        scope="nlp_g",
        penalty_name="late",
        phase=0,
        stage=2,
        occurrence=0,
        multi_thread=False,
    )
    registry.record(
        v[0] * v[2],
        kind="constraint",
        scope="ocp_g",
        penalty_name="early",
        phase=None,
        stage=-1,
        occurrence=0,
        multi_thread=False,
    )

    late, early = registry.materialize(v, lambda expression: expression)

    assert late.decision_indices == (1, 3)
    assert late.metadata.g_row_start == 1
    assert late.metadata.g_row_stop == 2
    np.testing.assert_allclose(late.value(DM([3, 5])), 19.0)
    np.testing.assert_allclose(late.jacobian(DM([3, 5])), [[6.0, 2.0]])
    np.testing.assert_allclose(late.lagrangian_hessian(DM([3, 5]), DM([1])), [[2.0, 0.0], [0.0, 0.0]])

    assert early.decision_indices == (0, 2)
    assert early.metadata.g_row_start == 0
    assert early.metadata.g_row_stop == 1
    np.testing.assert_allclose(early.value(DM([2, 7])), 14.0)
    np.testing.assert_allclose(early.jacobian(DM([2, 7])), [[7.0, 2.0]])
    np.testing.assert_allclose(early.lagrangian_hessian(DM([2, 7]), DM([1])), [[0.0, 1.0], [1.0, 0.0]])
    assert early.jacobian_sparsity.nnz() == 2
    assert early.hessian_sparsity.nnz() == 2


def test_post_shake_registry_keeps_constants_as_zero_input_kernels():
    v = MX.sym("v", 2, 1)
    registry = PostShakePenaltyRegistry()
    registry.record(
        MX(3),
        kind="objective",
        scope="ocp_J",
        penalty_name="constant",
        phase=None,
        stage=-1,
        occurrence=0,
        multi_thread=False,
    )
    (term,) = registry.materialize(v, lambda expression: expression)
    assert term.decision_indices == ()
    assert term.metadata.g_row_start is None
    np.testing.assert_allclose(term.value(DM.zeros(0, 1)), 3.0)
    assert term.jacobian_sparsity.nnz() == 0
    assert term.hessian_sparsity.nnz() == 0


def test_thread_map_fragments_resolve_to_slices_of_their_aggregate_parent_rows():
    v = MX.sym("v", 2, 1)
    registry = PostShakePenaltyRegistry()
    parent = registry.record(
        vertcat(v[0] ** 2, v[1] ** 2),
        kind="constraint",
        scope="nlp_g",
        penalty_name="mapped",
        phase=0,
        stage=0,
        occurrence=0,
        multi_thread=True,
    )
    registry.record_thread_map_fragment(
        v[0] ** 2,
        parent_term_index=parent,
        parent_row_offset=0,
        kind="thread_map_fragment",
        scope="nlp_g",
        penalty_name="mapped",
        phase=0,
        stage=0,
        occurrence=0,
        multi_thread=True,
    )
    registry.record_thread_map_fragment(
        v[1] ** 2,
        parent_term_index=parent,
        parent_row_offset=1,
        kind="thread_map_fragment",
        scope="nlp_g",
        penalty_name="mapped",
        phase=0,
        stage=1,
        occurrence=1,
        multi_thread=True,
    )
    parent_term, first, second = registry.materialize(v, lambda expression: expression)
    assert (parent_term.metadata.g_row_start, parent_term.metadata.g_row_stop) == (0, 2)
    assert (first.metadata.g_row_start, first.metadata.g_row_stop) == (0, 1)
    assert (second.metadata.g_row_start, second.metadata.g_row_stop) == (1, 2)
    np.testing.assert_allclose(first.value(DM([3])), [[9]])
    np.testing.assert_allclose(second.value(DM([5])), [[25]])
