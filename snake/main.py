from collections import deque
import curses
from enum import Enum
import logging
import random
import sys

import numpy as np

logger = logging.getLogger(__name__)

curses_colors = (
    curses.COLOR_WHITE,
    curses.COLOR_CYAN,
    curses.COLOR_BLUE,
    curses.COLOR_GREEN,
    curses.COLOR_YELLOW,
    curses.COLOR_MAGENTA,
    curses.COLOR_RED,
    curses.COLOR_RED,
    curses.COLOR_RED,
    curses.COLOR_RED,
    curses.COLOR_RED,
)

Direction = Enum("Direction", "NORTH EAST SOUTH WEST")


class Game:
    """
    Contains and manages the game state
    """

    def __init__(
        self,
        width,
        height,
        *,
        snakes=None,
        player_snake=None,
        max_number_of_fruits=1,
        max_number_of_snakes=1,
        log=None,
    ):
        self.fruits = []

        if snakes is None and player_snake is None:
            raise ValueError("There are no snakes!")
        if snakes is None:
            snakes = []
        self.snakes = snakes
        self.player_snake = player_snake
        if player_snake:
            self.snakes.append(player_snake)
        self.width, self.height = width, height
        self.log = log
        self.max_number_of_fruits = max_number_of_fruits
        self.max_number_of_snakes = max_number_of_snakes
        self.rewards = [0 for s in snakes]

    def __iter__(self):
        game_over = False

        while True:
            direction = yield
            if self.player_snake:
                self.player_snake.update(direction)
            for snake in self.snakes:
                if snake is not self.player_snake:
                    snake.update(None)
            self.check_collisions()
            if not self.snakes:
                game_over = True
            self.update_fruits()

            if game_over:
                break

    def update_fruits(self):
        """Add fruits to the game until max_number_of_fruits is reached."""
        while len(self.fruits) < self.max_number_of_fruits:
            new_x, new_y = (
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1),
            )
            self.fruits.append((new_x, new_y))

    def check_collisions(self):
        fruits_to_be_deleted = []
        snakes_to_be_deleted = []

        for s_idx, s in enumerate(self.snakes):
            x_s, y_s = s.coordinates[-1]
            # Check fruit collision
            for fruit in self.fruits:
                if (x_s, y_s) == fruit:
                    s.length += 2
                    fruits_to_be_deleted.append(fruit)
                    self.rewards[s_idx] += 1
                    logger.debug("Snake %s got a fruit", s_idx)
            # Check snake collisions
            for s2_idx, s2 in enumerate(self.snakes):
                if s_idx != s2_idx:
                    for x2s, y2s in s2.coordinates:
                        if (x_s, y_s) == (x2s, y2s):
                            snakes_to_be_deleted.append(s)
                else:
                    for x2s, y2s in list(s2.coordinates)[:-1]:
                        if (x_s, y_s) == (x2s, y2s):
                            snakes_to_be_deleted.append(s)

        for tbd in fruits_to_be_deleted:
            self.fruits.remove(tbd)
        for snk in snakes_to_be_deleted:
            self.snakes.remove(snk)

    @property
    def state_array(self):
        """Return array of current state.
        The game board is encoded as follows:
        Snake body: 1
        Snake head: 2
        Fruit : 3
        Snake head on fruit: 4"""

        state = np.zeros((self.width, self.height))
        for snake in self.snakes:
            for i in range(len(snake.coordinates) - 1):
                x, y = snake.coordinates[i]
                state[x, y] = 1
            head_coord = snake.coordinates[-1]
            if head_coord in self.fruits:
                state[snake.coordinates[-1]] = 4
                self.fruits.remove(head_coord)
            else:
                state[snake.coordinates[-1]] = 2

        for x, y in self.fruits:
            state[x, y] = 3
        return state

    def get_surrounding_view(self, snake):
        idx = self.snakes.index(snake)
        arr = self.state_array
        x, y = self.snakes[idx].coordinates[-1]
        view = np.roll(arr, (arr.shape[0] // 2 - x, arr.shape[1] // 2 - y), axis=(0, 1))
        return view[
            view.shape[0] // 2 - 2 : view.shape[0] // 2 + 3,
            view.shape[1] // 2 - 2 : view.shape[1] // 2 + 3,
        ].T

    def reduced_coordinates(self, snake):
        """
        Returns an array of length three.
        If the first entry is one, there is a fruit left to the snake.
        If the second entry is one, there is a fruit ahead of the snake.
        If the third entry is one, there is a fruit right of the snake.

        Parameters
        ----------
        snake : Snake
        """
        head_x, head_y = snake.coordinates[-1]
        direction = snake.direction
        result = np.zeros(3)

        for x, y in self.fruits:
            if y == head_y:
                if x > head_x:
                    if direction == Direction.EAST:
                        result[1] = 1
                    elif direction == Direction.NORTH:
                        result[2] = 1
                    elif direction == Direction.SOUTH:
                        result[0] = 1
                elif x < head_x:
                    if direction == Direction.WEST:
                        result[1] = 1
                    elif direction == Direction.NORTH:
                        result[0] = 1
                    elif direction == Direction.SOUTH:
                        result[2] = 1
            elif x == head_x:
                if y > head_y:
                    if direction == Direction.EAST:
                        result[0] = 1
                    elif direction == Direction.NORTH:
                        result[1] = 1
                    elif direction == Direction.WEST:
                        result[2] = 1
                elif y < head_y:
                    if direction == Direction.EAST:
                        result[0] = 1
                    elif direction == Direction.NORTH:
                        result[1] = 1
                    elif direction == Direction.WEST:
                        result[2] = 1
        return result


class Snake:
    def __init__(self, x, y, max_x, max_y, direction):
        self.coordinates = deque([(x, y)])
        self.max_x, self.max_y = max_x, max_y
        self.direction = direction
        self.length = 1

    def __repr__(self):
        x, y = self.coordinates[-1]
        return f"Snake({x}, {y})"

    @classmethod
    def random_init(cls, width, height):
        start_direction = random.choice(list(Direction))
        x, y = random.randint(1, width - 1), random.randint(1, height - 1)
        return cls(x, y, width, height, start_direction)

    def update(self, direction):
        if direction:
            new_direction = direction
        else:
            new_direction = self.direction

        head_x, head_y = self.coordinates[-1]

        # Do not allow 180° turnaround
        if (new_direction, self.direction) in [
            (Direction.NORTH, Direction.SOUTH),
            (Direction.SOUTH, Direction.NORTH),
            (Direction.EAST, Direction.WEST),
            (Direction.WEST, Direction.EAST),
        ]:
            new_direction = self.direction

        if new_direction == Direction.NORTH:
            new_x, new_y = head_x, head_y - 1
        elif new_direction == Direction.EAST:
            new_x, new_y = head_x + 1, head_y
        elif new_direction == Direction.SOUTH:
            new_x, new_y = head_x, head_y + 1
        else:
            new_x, new_y = head_x - 1, head_y

        self.direction = new_direction

        new_x %= self.max_x
        new_y %= self.max_y

        self.coordinates.append((new_x, new_y))
        if len(self.coordinates) > self.length:
            self.coordinates.popleft()

