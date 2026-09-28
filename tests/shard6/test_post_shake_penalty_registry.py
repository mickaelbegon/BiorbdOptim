import numpy as np
import casadi as ca
from casadi import DM, MX, vertcat

from bioptim.interfaces.post_shake_penalty_registry import PostShakePenaltyRegistry
from bioptim.interfaces.compiled_thread_map_external import (
    compile_external_output_packets,
    compile_external_with_exact_derivatives,
)


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


def test_pre_shake_thread_map_fragment_uses_its_original_stage_graph_and_canonical_parent_rows():
    """A pre-map stage graph need not be a slice of the aggregate MX graph."""

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
    # Deliberately equivalent but structurally independent stage graphs.
    registry.record_pre_shake_thread_map_fragment(
        (v[0] + 0) ** 2,
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
    registry.record_pre_shake_thread_map_fragment(
        (v[1] + 0) ** 2,
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
    assert first.metadata.pre_shake_thread_map_fragment is True
    np.testing.assert_allclose(first.value(DM([3])), [[9]])
    np.testing.assert_allclose(second.value(DM([5])), [[25]])


def test_pre_shake_thread_map_packet_keeps_one_exact_kernel_and_stage_scatter_plan():
    """A packet avoids materializing one MX Hessian graph per mapped stage."""

    v = MX.sym("v", 2, 1)
    registry = PostShakePenaltyRegistry()
    parent = registry.record(
        vertcat(v[0] ** 2, v[1] ** 2),
        kind="constraint",
        scope="nlp_g",
        penalty_name="STATE_CONTINUITY",
        phase=0,
        stage=0,
        occurrence=0,
        multi_thread=True,
    )
    for occurrence, column in enumerate((0, 1)):
        registry.record_pre_shake_thread_map_fragment(
            v[column] ** 2,
            parent_term_index=parent,
            parent_row_offset=occurrence,
            kind="thread_map_fragment",
            scope="nlp_g",
            penalty_name="STATE_CONTINUITY",
            phase=0,
            stage=occurrence,
            occurrence=occurrence,
            multi_thread=True,
        )

    packet = registry.build_pre_shake_thread_map_packet(v, lambda expression: expression, "STATE_CONTINUITY")
    assert packet.decision_indices == ((0,), (1,))
    assert [(item.g_row_start, item.g_row_stop) for item in packet.metadata] == [(0, 1), (1, 2)]
    np.testing.assert_allclose(packet.value(DM([3])), [[9]])
    np.testing.assert_allclose(packet.lagrangian_hessian(DM([3]), DM([1])), [[2]])


def test_compiled_thread_map_external_preserves_exact_jacobian_and_hessian(tmp_path):
    """The C primal alone is insufficient: this verifies its exact AD helpers."""

    x = ca.MX.sym("x", 2)
    y = ca.MX.sym("y")
    native = ca.Function("thread_map_stage", [x, y], [ca.vertcat(x[0] ** 2 + y, ca.sin(x[1]) * y)])
    compiled = compile_external_with_exact_derivatives(native, tmp_path)
    decision = ca.vertcat(x, y)
    multipliers = ca.MX.sym("lambda", 2)
    native_jacobian = ca.Function("native_jacobian", [x, y], [ca.jacobian(native(x, y), decision)])
    compiled_jacobian = ca.Function("compiled_jacobian", [x, y], [ca.jacobian(compiled(x, y), decision)])
    native_hessian = ca.Function(
        "native_hessian", [x, y, multipliers], [ca.hessian(ca.dot(multipliers, native(x, y)), decision)[0]]
    )
    compiled_hessian = ca.Function(
        "compiled_hessian", [x, y, multipliers], [ca.hessian(ca.dot(multipliers, compiled(x, y)), decision)[0]]
    )
    args = (ca.DM([2.0, 3.0]), ca.DM([4.0]))
    np.testing.assert_allclose(compiled(*args), native(*args), atol=0, rtol=0)
    np.testing.assert_allclose(compiled_jacobian(*args), native_jacobian(*args), atol=0, rtol=0)
    np.testing.assert_allclose(compiled_hessian(*args, ca.DM([1.0, 2.0])), native_hessian(*args, ca.DM([1.0, 2.0])), atol=0, rtol=0)

    # A second independent call is the normal FHO-successor case: the
    # content-addressed artifact and its manifest must be reused, rather than
    # recompiling an identical local stage kernel.
    libraries_before = sorted(tmp_path.glob("*.so"))
    manifests_before = sorted(tmp_path.glob("*.json"))
    cached = compile_external_with_exact_derivatives(native, tmp_path)
    assert sorted(tmp_path.glob("*.so")) == libraries_before
    assert sorted(tmp_path.glob("*.json")) == manifests_before
    np.testing.assert_allclose(cached(*args), native(*args), atol=0, rtol=0)


def test_packetized_compiled_thread_map_external_preserves_exact_second_order_ad(tmp_path):
    """Output packetization bounds C units without changing the MX derivatives."""

    x = ca.MX.sym("x", 3)
    native = ca.Function(
        "wide_thread_map_stage", [x], [ca.vertcat(x[0] ** 2, ca.sin(x[1]), x[0] * x[2], x[2] ** 3)]
    )
    packetized = compile_external_output_packets(native, tmp_path, max_output_rows=2)
    multiplier = ca.MX.sym("lambda", 4)
    native_hessian = ca.Function("wide_native_h", [x, multiplier], [ca.hessian(ca.dot(multiplier, native(x)), x)[0]])
    packet_hessian = ca.Function(
        "wide_packet_h", [x, multiplier], [ca.hessian(ca.dot(multiplier, packetized(x)), x)[0]]
    )
    values = ca.DM([2.0, 0.3, -1.5])
    lambdas = ca.DM([1.0, -2.0, 3.0, 4.0])
    np.testing.assert_allclose(packetized(values), native(values), atol=0, rtol=0)
    np.testing.assert_allclose(packet_hessian(values, lambdas), native_hessian(values, lambdas), atol=0, rtol=0)
    assert len(list(tmp_path.glob("*.so"))) == 2
