"""Planning of penalty nodes within a phase."""

from dataclasses import dataclass

from ..misc.enums import Node


@dataclass(frozen=True)
class PenaltyNodePlan:
    """Canonical node specification and the corresponding shooting indices."""

    requested_nodes: tuple[Node | int, ...]
    indices: tuple[int, ...]


class PenaltyNodeResolver:
    """Resolve public :class:`Node` values independently from penalty storage."""

    @staticmethod
    def resolve(node: Node | int | tuple[Node | int, ...] | list[Node | int], n_shooting: int) -> PenaltyNodePlan:
        """Resolve a node specification for a phase with ``n_shooting`` intervals."""

        requested_nodes = node if isinstance(node, (list, tuple)) else (node,)
        indices = []
        for requested_node in requested_nodes:
            if isinstance(requested_node, int):
                if requested_node < 0 or requested_node > n_shooting:
                    raise RuntimeError(f"Invalid node, {requested_node} must be between 0 and {n_shooting}")
                indices.append(requested_node)
            elif requested_node == Node.START:
                indices.append(0)
            elif requested_node == Node.MID:
                if n_shooting % 2 == 1:
                    raise ValueError("Number of shooting points must be even to use MID")
                indices.append(n_shooting // 2)
            elif requested_node == Node.INTERMEDIATES:
                indices.extend(range(1, n_shooting - 1))
            elif requested_node == Node.PENULTIMATE:
                if n_shooting < 2:
                    raise ValueError("Number of shooting points must be greater than 1")
                indices.append(n_shooting - 1)
            elif requested_node == Node.END:
                indices.append(n_shooting)
            elif requested_node == Node.ALL_SHOOTING:
                indices.extend(range(n_shooting))
            elif requested_node == Node.ALL:
                indices.extend(range(n_shooting + 1))
            else:
                raise RuntimeError(f"{requested_node} is not a valid node")

        return PenaltyNodePlan(requested_nodes=tuple(requested_nodes), indices=tuple(indices))
