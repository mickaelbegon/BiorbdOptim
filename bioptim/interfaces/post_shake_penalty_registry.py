"""Introspection of penalty terms after Bioptim's symbolic ``shake`` step.

The canonical NLP is deliberately assembled as one CasADi graph.  This module
does *not* change that graph or the solver callbacks.  It provides a stable,
opt-in description of the individual terms that made it into that graph.  A
consumer can use the description to group equal local terms, generate one C
kernel per group, and map that kernel over stages while retaining exact
derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from casadi import Function, MX, SX, hessian, jacobian, substitute, sum1, vertcat


CX = MX | SX


@dataclass(frozen=True)
class PostShakePenaltyMetadata:
    """Location and provenance of a single canonical penalty contribution."""

    kind: str
    scope: str
    penalty_name: str
    phase: int | None
    stage: int
    occurrence: int
    multi_thread: bool
    g_row_start: int | None = None
    g_row_stop: int | None = None


@dataclass
class PostShakePenaltyTerm:
    """A local exact view of one post-shake canonical penalty term.

    ``value``, ``jacobian`` and ``lagrangian_hessian`` use only
    ``decision_indices`` as inputs.  They therefore contain no inactive NLP
    variables and can safely be used to build a per-type compiled kernel.
    """

    metadata: PostShakePenaltyMetadata
    decision_indices: tuple[int, ...]
    jacobian_sparsity: Any
    hessian_sparsity: Any
    value: Function
    jacobian: Function
    lagrangian_hessian: Function


@dataclass
class _PendingTerm:
    metadata: PostShakePenaltyMetadata
    expression: CX


@dataclass
class PostShakePenaltyRegistry:
    """Collect raw terms, then materialize exact post-shake local functions.

    A registry is intentionally ephemeral: create it while dispatching the
    exact problem, call :meth:`materialize`, and discard it when the NLP graph
    changes.  This avoids stale indices during receding-horizon solves.
    """

    _pending: list[_PendingTerm] = field(default_factory=list)
    terms: list[PostShakePenaltyTerm] = field(default_factory=list)

    def record(self, expression: CX, **metadata: Any) -> None:
        """Record one raw contribution in canonical dispatch order."""

        if expression.shape[1] != 1:
            expression = vertcat(expression)
        self._pending.append(_PendingTerm(PostShakePenaltyMetadata(**metadata), expression))

    def materialize(self, v: CX, shake: Callable[[CX], CX]) -> list[PostShakePenaltyTerm]:
        """Apply ``shake`` and create exact local value/Jacobian/Hessian views.

        Parameters
        ----------
        v
            The final NLP decision vector before ``shake``.
        shake
            The same transformation used by the solver path.  Passing it in
            keeps this module independent from :mod:`interface_utils`.
        """

        self.terms = []
        # The final canonical g is concatenated by increasing stage key.  The
        # registry may also contain objectives, so compute these bases before
        # materializing any individual term.
        rows_per_stage: dict[int, int] = {}
        for pending in self._pending:
            if pending.metadata.kind == "constraint":
                rows_per_stage[pending.metadata.stage] = rows_per_stage.get(pending.metadata.stage, 0) + pending.expression.shape[0]
        stage_base: dict[int, int] = {}
        next_row = 0
        for stage in sorted(rows_per_stage):
            stage_base[stage] = next_row
            next_row += rows_per_stage[stage]
        constraint_offsets: dict[int, int] = {}
        for pending in self._pending:
            metadata = pending.metadata
            expression = shake(pending.expression)
            jac = jacobian(expression, v)
            active_columns = tuple(sorted(set(int(column) for column in jac.sparsity().get_col())))
            local_x = v.__class__.sym("post_shake_x", len(active_columns), 1)
            local_expression = self._localize(expression, v, active_columns, local_x)
            local_jacobian = jacobian(local_expression, local_x)
            multipliers = v.__class__.sym("post_shake_lambda", expression.shape[0], 1)
            local_hessian, _ = hessian(sum1(multipliers * local_expression), local_x)

            term_metadata = metadata
            if metadata.kind == "constraint":
                row_start = stage_base[metadata.stage] + constraint_offsets.get(metadata.stage, 0)
                row_stop = row_start + expression.shape[0]
                constraint_offsets[metadata.stage] = row_stop - stage_base[metadata.stage]
                term_metadata = PostShakePenaltyMetadata(
                    **{**metadata.__dict__, "g_row_start": row_start, "g_row_stop": row_stop}
                )

            identifier = f"post_shake_{len(self.terms)}"
            self.terms.append(
                PostShakePenaltyTerm(
                    metadata=term_metadata,
                    decision_indices=active_columns,
                    jacobian_sparsity=local_jacobian.sparsity(),
                    hessian_sparsity=local_hessian.sparsity(),
                    value=Function(f"{identifier}_value", [local_x], [local_expression]),
                    jacobian=Function(f"{identifier}_jacobian", [local_x], [local_jacobian]),
                    lagrangian_hessian=Function(
                        f"{identifier}_hessian", [local_x, multipliers], [local_hessian]
                    ),
                )
            )
        return self.terms

    @staticmethod
    def _localize(expression: CX, v: CX, indices: tuple[int, ...], local_x: CX) -> CX:
        if not indices:
            return expression
        # ``substitute`` is exact here: active indices come from the symbolic
        # Jacobian itself, hence no other component of ``v`` remains in the
        # localized expression. CasADi requires the *substituted variable* to
        # be purely symbolic, so replace the full decision vector rather than
        # a sliced expression of it.
        local_position = {index: position for position, index in enumerate(indices)}
        replacement = vertcat(
            *[local_x[local_position[index]] if index in local_position else 0 for index in range(v.numel())]
        )
        return substitute(expression, v, replacement)
