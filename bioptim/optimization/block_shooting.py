from __future__ import annotations

from dataclasses import dataclass


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be greater than or equal to 1")


def compute_block_boundaries(
    n_shooting: int,
    *,
    n_blocks: int | None = None,
    block_size: int | None = None,
) -> tuple[int, ...]:
    """Compute the shooting-node boundaries of a deterministic block partition."""

    _validate_positive_integer(n_shooting, "n_shooting")
    if (n_blocks is None) == (block_size is None):
        raise ValueError("Exactly one of n_blocks and block_size must be provided")

    if n_blocks is not None:
        if not isinstance(n_blocks, int) or isinstance(n_blocks, bool):
            raise TypeError("n_blocks must be an integer")
        if not 1 <= n_blocks <= n_shooting:
            raise ValueError("n_blocks must be between 1 and n_shooting")

        minimum_block_size, number_of_larger_blocks = divmod(n_shooting, n_blocks)
        block_lengths = (minimum_block_size + 1,) * number_of_larger_blocks + (minimum_block_size,) * (
            n_blocks - number_of_larger_blocks
        )
    else:
        _validate_positive_integer(block_size, "block_size")
        number_of_full_blocks, remainder = divmod(n_shooting, block_size)
        block_lengths = (block_size,) * number_of_full_blocks
        if remainder:
            block_lengths += (remainder,)

    boundaries = [0]
    for length in block_lengths:
        boundaries.append(boundaries[-1] + length)
    return tuple(boundaries)


@dataclass(frozen=True)
class BlockShooting:
    """Configuration of the block partition for one OCP phase."""

    n_blocks: int | None = None
    block_size: int | None = None

    def __post_init__(self) -> None:
        if (self.n_blocks is None) == (self.block_size is None):
            raise ValueError("Exactly one of n_blocks and block_size must be provided")
        if self.n_blocks is not None:
            _validate_positive_integer(self.n_blocks, "n_blocks")
        else:
            _validate_positive_integer(self.block_size, "block_size")

    @classmethod
    def single(cls) -> BlockShooting:
        return cls(n_blocks=1)

    @classmethod
    def from_number_of_blocks(cls, n_blocks: int) -> BlockShooting:
        return cls(n_blocks=n_blocks)

    @classmethod
    def from_block_size(cls, block_size: int) -> BlockShooting:
        return cls(block_size=block_size)

    def compute_boundaries(self, n_shooting: int) -> tuple[int, ...]:
        return compute_block_boundaries(
            n_shooting,
            n_blocks=self.n_blocks,
            block_size=self.block_size,
        )


def normalize_block_shooting(
    block_shooting: BlockShooting | list[BlockShooting] | tuple[BlockShooting, ...] | None,
    n_shooting: int | list[int] | tuple[int, ...],
    n_phases: int,
) -> tuple[BlockShooting | None, ...]:
    """Normalize a public block-shooting option into one validated entry per phase."""

    if block_shooting is None:
        return (None,) * n_phases
    if isinstance(block_shooting, BlockShooting):
        configurations = (block_shooting,) * n_phases
    elif isinstance(block_shooting, (list, tuple)):
        if len(block_shooting) != n_phases:
            raise ValueError("block_shooting must provide one configuration per phase")
        if not all(isinstance(configuration, BlockShooting) for configuration in block_shooting):
            raise TypeError("block_shooting entries must be BlockShooting instances")
        configurations = tuple(block_shooting)
    else:
        raise TypeError("block_shooting must be a BlockShooting or a sequence of BlockShooting")

    shooting_points = (n_shooting,) * n_phases if isinstance(n_shooting, int) else tuple(n_shooting)
    for configuration, phase_n_shooting in zip(configurations, shooting_points):
        configuration.compute_boundaries(phase_n_shooting)
    return configurations
