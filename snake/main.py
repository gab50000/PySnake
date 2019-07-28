import curses
from enum import Enum
import logging
import random
import sys

import numpy as np

from snake import Direction

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
        view_size=3,
        border=False
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
        self.view_size = view_size
        self.border = border
        self.max_number_of_fruits = max_number_of_fruits
        self.max_number_of_snakes = max_number_of_snakes
        self.rewards = [0 for s in snakes]

    def __iter__(self):
        game_over = False
        old_direction = None

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

            if self.border:
                if any((x_s < 0, x_s > self.width, y_s < 0, y_s > self.height)):
                    snakes_to_be_deleted.append(s)
                    continue
            else:
                x_s %= self.width
                y_s %= self.height
                s.coordinates[-1] = x_s, y_s

            # Check fruit collision
            for fruit in self.fruits:
                if (x_s, y_s) == fruit:
                    s.length += 2
                    fruits_to_be_deleted.append(fruit)
                    self.rewards[s_idx] += 10
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
        """
        Return array of current state.
        The game board is encoded as follows:
        Snake body: 1
        Fruit : 2
        """

        state = np.zeros((self.width, self.height), int)
        for snake in self.snakes:
            for x, y in snake.coordinates:
                state[x, y] = 1

        for x, y in self.fruits:
            state[x, y] = 2
        return state

    def get_surrounding_view(self, snake, onehot=False):
        vs = self.view_size
        idx = self.snakes.index(snake)
        arr = self.state_array
        x, y = self.snakes[idx].coordinates[-1]
        view = np.roll(arr, (arr.shape[0] // 2 - x, arr.shape[1] // 2 - y), axis=(0, 1))
        view = view[
            view.shape[0] // 2 - vs + 1 : view.shape[0] // 2 + vs,
            view.shape[1] // 2 - vs + 1 : view.shape[1] // 2 + vs,
        ].T

        if onehot:
            vec = np.zeros((*view.shape, 2), int)
            nonzero = view > 0
            vec[nonzero, view[nonzero] - 1] = 1
            return vec

        return view

    def coordinate_occupied(self, coord):
        if coord in self.fruits:
            return 1
        if any(coord in snake.coordinates for snake in self.snakes):
            return 2

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
        result = np.zeros((4, 2))

        # look north
        for y in reversed(range(head_y)):
            occ = self.coordinate_occupied((head_x, y))
            if occ:
                result[0, occ - 1] = 1
                break

        # look east
        for x in range(head_x + 1, self.width):
            occ = self.coordinate_occupied((x, head_y))
            if occ:
                result[1, occ - 1] = 1
                break

        # look south
        for y in range(head_y + 1, self.height):
            occ = self.coordinate_occupied((head_x, y))
            if occ:
                result[2, occ - 1] = 1
                break

        # look west
        for x in reversed(range(head_x)):
            occ = self.coordinate_occupied((x, head_y))
            if occ:
                result[3, occ - 1] = 1
                break

        direction_idx = direction.value
        result = np.roll(result, Direction.EAST.value - direction_idx, axis=0)
        return result[:3]

