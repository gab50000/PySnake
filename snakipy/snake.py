from collections import deque
from dataclasses import dataclass
from enum import Enum
import random
import logging
from typing import List, Tuple, Optional

import numpy as np

from snakipy.neuro import NeuralNet


logger = logging.getLogger(__name__)


Direction = Enum(
    "Direction", "NORTH NORTHEAST EAST SOUTHEAST SOUTH SOUTHWEST WEST NORTHWEST"
)


def direction_to_vector(direction):
    return {Direction.NORTH: (0, -1), Direction.NORTHEAST: (1, -1)}


@dataclass
class Snake:
    coordinates: List[Tuple[int, int]]
    board_width: int
    board_height: int
    direction: Direction
    length: int = 2
    periodic: bool = False
    game_over: bool = False

    _idx = 0

    def __post_init__(self):
        self._idx = type(self)._idx
        type(self)._idx += 1

    def __eq__(self, other):
        return isinstance(other, Snake) and self._idx == other._idx

    @classmethod
    def new_snake(cls, x, y, board_width, board_height, direction, **kwargs):
        coordinates = [(x, y)]
        board_width, board_height = board_width, board_height
        direction = direction
        length = 2
        return cls(coordinates, board_width, board_height, direction, length, **kwargs)

    def __repr__(self):
        x, y = self.head
        return f"Snake({x}, {y})"

    @property
    def head(self):
        return self.coordinates[-1]

    @property
    def tail(self):
        return self.coordinates[0]

    @classmethod
    def random_init(cls, board_width, board_height, **kwargs):
        start_direction = random.choice(
            [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        )
        x, y = random.randint(1, board_width - 1), random.randint(1, board_height - 1)
        return cls(x, y, board_width, board_height, start_direction, **kwargs)

    def update(self, direction):
        """

        Args:
            direction: new direction

        Returns: snake

        Examples:
            >>> snk = Snake.new_snake(3, 3, 10, 10, Direction.EAST)
            >>> snk.update(Direction.EAST)
            Snake(4, 3)
            >>> snk = Snake.new_snake(3, 9, 10, 10, Direction.SOUTH, periodic=True)
            >>> snk.update(Direction.SOUTH)
            Snake(3, 0)
        """
        if direction:
            new_direction = direction
        else:
            new_direction = self.direction

        head_x, head_y = self.coordinates[-1]
        logger.debug("Old position: (%s, %s)", head_x, head_y)

        # Do not allow 180° turnaround
        if (new_direction, self.direction) in [
            (Direction.NORTH, Direction.SOUTH),
            (Direction.SOUTH, Direction.NORTH),
            (Direction.EAST, Direction.WEST),
            (Direction.WEST, Direction.EAST),
        ]:
            logger.debug(
                "180° turn from %s to %s not allowed",
                self.direction.name.title(),
                new_direction.name.title(),
            )
            new_direction = self.direction

        if new_direction == Direction.NORTH:
            dx, dy = 0, -1
        elif new_direction == Direction.EAST:
            dx, dy = 1, 0
        elif new_direction == Direction.SOUTH:
            dx, dy = 0, 1
        else:
            dx, dy = -1, 0

        new_x, new_y = head_x + dx, head_y + dy

        if self.periodic:
            new_x = new_x % self.board_width
            new_y = new_y % self.board_height

        logger.debug("New position: (%s, %s)", new_x, new_y)

        new_coordinates = self.coordinates + [(new_x, new_y)]
        new_coordinates = new_coordinates[-self.length :]

        init_params = {
            **vars(self),
            "coordinates": new_coordinates,
            "direction": new_direction,
        }
        init_params = {k: v for k, v in init_params.items() if not k.startswith("_")}
        return type(self)(**init_params)


@dataclass(eq=False)
class NeuroSnake(Snake):
    input_size: int = 16
    hidden_size: int = 5
    dna: Optional[np.ndarray] = None
    net: Optional[NeuralNet] = None

    def __post_init__(self):
        self.net = NeuralNet(self.input_size, self.hidden_size, 3, dna=self.dna)
        if self.dna is None:
            self.dna = self.net.dna

    def decide_direction(self, view):
        dirs = (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST)
        if self.direction is None:
            self.direction = random.choice(dirs)
            return self.direction

        net_output = self.net.forward(view)
        dir_idx = dirs.index(self.direction)
        idx = np.argmax(net_output) - 1
        new_dir = dirs[(dir_idx + idx) % 4]
        logger.debug("Old direction: %s", self.direction)
        logger.debug("New direction: %s", new_dir)
        return new_dir
