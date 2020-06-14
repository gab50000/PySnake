from collections import deque
from enum import Enum
import random
import logging

import numpy as np

from .neuro import NeuralNet


logger = logging.getLogger(__name__)


Direction = Enum(
    "Direction", "NORTH NORTHEAST EAST SOUTHEAST SOUTH SOUTHWEST WEST NORTHWEST"
)


def direction_to_vector(direction):
    return {Direction.NORTH: (0, -1), Direction.NORTHEAST: (1, -1)}


class Snake:
    def __init__(self, x, y, max_x, max_y, direction):
        self.coordinates = deque([(x, y)])
        self.max_x, self.max_y = max_x, max_y
        self.direction = direction
        self.length = 1

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
    def random_init(cls, width, height, **kwargs):
        start_direction = random.choice(
            [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        )
        x, y = random.randint(1, width - 1), random.randint(1, height - 1)
        return cls(x, y, width, height, start_direction, **kwargs)

    def update(self, direction):
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
            new_x, new_y = head_x, head_y - 1
        elif new_direction == Direction.EAST:
            new_x, new_y = head_x + 1, head_y
        elif new_direction == Direction.SOUTH:
            new_x, new_y = head_x, head_y + 1
        else:
            new_x, new_y = head_x - 1, head_y

        logger.debug("New position: (%s, %s)", new_x, new_y)
        self.direction = new_direction

        self.coordinates.append((new_x, new_y))
        if len(self.coordinates) > self.length:
            self.coordinates.popleft()


class NeuroSnake(Snake):
    def __init__(
        self, x, y, max_x, max_y, input_size, hidden_size, direction=None, dna=None
    ):
        super().__init__(x, y, max_x, max_y, direction=direction)
        self.net = NeuralNet(input_size, hidden_size, 3, dna=dna)
        self.net_output = None

    def decide_direction(self, view):
        dirs = (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST)
        if self.direction is None:
            self.direction = random.choice(dirs)
            return self.direction

        self.net_output = self.net.forward(view)
        dir_idx = dirs.index(self.direction)
        idx = np.argmax(self.net_output) - 1
        new_dir = dirs[(dir_idx + idx) % 4]
        logger.debug("Old direction: %s", self.direction)
        logger.debug("New direction: %s", new_dir)
        return new_dir
