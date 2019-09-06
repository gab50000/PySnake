import curses
from enum import Enum
import logging
import random
import sys

import numpy as np

from .snake import Direction

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


FRUIT_REWARD = 10
DEATH_REWARD = -50
DISTANCE_REWARD = 2


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
        border=False,
        seed=None
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
        self.closest_distance = [None for s in snakes]
        self.rng = np.random.RandomState(seed)
        self.update_fruits()

    def __iter__(self):
        game_over = False

        while True:
            direction = yield
            if self.player_snake:
                self.punish_circles(self.player_snake, direction)
                self.player_snake.update(direction)
            for snake in self.snakes:
                if snake is not self.player_snake:
                    snake.update(None)
            self.check_collisions()
            if not self.snakes:
                game_over = True
            self.update_fruits()
            self.update_distances()

            if game_over:
                break

    def punish_circles(self, snake, new_direction):
        dir_list = list(Direction)
        dir_idx = dir_list.index(snake.direction)
        snake_idx = self.snakes.index(snake)
        i1 = (dir_idx + 1) % 4
        i2 = (dir_idx - 1) % 4
        if dir_list[i1] == new_direction or dir_list[i2] == new_direction:
            self.rewards[snake_idx] -= 1

    def update_fruits(self):
        """Add fruits to the game until max_number_of_fruits is reached."""
        while len(self.fruits) < self.max_number_of_fruits:
            new_x, new_y = (
                self.rng.randint(0, self.width - 1),
                self.rng.randint(0, self.height - 1),
            )
            self.fruits.append((new_x, new_y))

    def update_distances(self):
        new_distances = self.determine_fruit_distances()
        for idx, (old_dist, new_dist) in enumerate(
            zip(self.closest_distance, new_distances)
        ):
            if old_dist is None:
                self.closest_distance[idx] = new_dist
                continue

            if new_dist < old_dist:
                self.rewards[idx] += DISTANCE_REWARD
            elif new_dist > old_dist:
                self.rewards[idx] -= DISTANCE_REWARD + 1
            self.closest_distance[idx] = new_dist

    def determine_fruit_distances(self):
        return [
            min([self.fruit_distance(snake, fruit) for fruit in self.fruits])
            for snake in self.snakes
        ]

    @staticmethod
    def fruit_distance(snake, fruit):
        x, y = snake.coordinates[-1]
        xf, yf = fruit
        return abs(x - xf) + abs(y - yf)

    def check_collisions(self):
        fruits_to_be_deleted = []
        snakes_to_be_deleted = []

        for s_idx, s in enumerate(self.snakes):
            x_s, y_s = s.coordinates[-1]

            if self.border:
                if any((x_s < 0, x_s >= self.width, y_s < 0, y_s >= self.height)):
                    snakes_to_be_deleted.append(s)
                    self.rewards[s_idx] += DEATH_REWARD
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
                    self.rewards[s_idx] += FRUIT_REWARD
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
                            self.rewards[s_idx] += DEATH_REWARD

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

        state = np.zeros((self.width, self.height, 2), float)
        for snake in self.snakes:
            for x, y in snake.coordinates:
                state[x, y, 0] = 1

        for x, y in self.fruits:
            state[x, y, 1] = 1
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

    def is_wall_or_snake(self, coord):
        if self.border:
            if coord[0] in (-1, self.width) or coord[1] in (-1, self.height):
                return True
        for snake in self.snakes:
            if coord in snake.coordinates:
                return True
        return False

    def fruit_ahead(self, coord, direction):
        head_x, head_y = coord

        # look north
        if direction == Direction.NORTH:
            for y in reversed(range(head_y)):
                if (head_x, y) in self.fruits:
                    return True

        # look east
        if direction == Direction.EAST:
            for x in range(head_x + 1, self.width):
                if (x, head_y) in self.fruits:
                    return True

        # look south
        if direction == Direction.SOUTH:
            for y in range(head_y + 1, self.height):
                if (head_x, y) in self.fruits:
                    return True

        # look west
        if direction == Direction.WEST:
            for x in reversed(range(head_x)):
                if (x, head_y) in self.fruits:
                    return True
        return False

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
        if self.is_wall_or_snake((head_x, head_y - 1)):
            result[0, 1] = 1
        if self.fruit_ahead((head_x, head_y), Direction.NORTH):
            result[0, 0] = 1

        # look east
        if self.is_wall_or_snake((head_x + 1, head_y)):
            result[1, 1] = 1
        if self.fruit_ahead((head_x, head_y), Direction.EAST):
            result[1, 0] = 1

        # look south
        if self.is_wall_or_snake((head_x, head_y + 1)):
            result[2, 1] = 1
        if self.fruit_ahead((head_x, head_y), Direction.SOUTH):
            result[2, 0] = 1

        # look west
        if self.is_wall_or_snake((head_x - 1, head_y)):
            result[3, 1] = 1
        if self.fruit_ahead((head_x, head_y), Direction.WEST):
            result[3, 0] = 1

        direction_idx = direction.value
        result = np.roll(result, Direction.EAST.value - direction_idx, axis=0)
        return result[:3]
