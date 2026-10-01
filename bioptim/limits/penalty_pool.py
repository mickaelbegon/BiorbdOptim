"""Utilities for allocating penalty slots in objective and constraint pools."""

from typing import Any


class PenaltyPool:
    """Keep the sparse-list allocation contract shared by all penalty families."""

    @staticmethod
    def reserve_slot(pool: list[Any], list_index: int) -> int:
        """Reserve ``list_index`` in ``pool`` and return its effective index.

        A negative index selects the first empty slot or appends one.  A
        non-negative index extends the pool as needed and clears that slot for
        replacement.  This preserves the historical pool semantics.
        """

        if list_index < 0:
            for index, penalty in enumerate(pool):
                if not penalty:
                    return index
            pool.append([])
            return len(pool) - 1

        while list_index >= len(pool):
            pool.append([])
        pool[list_index] = []
        return list_index
