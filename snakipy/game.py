import curses
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from snakipy.snake import Direction, NeuroSnake, Snake

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
DISTANCE_REWARD = 0.4


@dataclass
class Game:
    """
    Contains and manages the game state
    """

    width: int
    height: int
    snakes: List[Snake] = field(default_factory=list)
    player_snake: Optional[Snake] = None
    max_number_of_fruits: int = 1
    max_number_of_snakes: int = 1
    border: bool = False
    seed: Optional[int] = None

    def __post_init__(self):
        self.fruits = []

        if self.snakes is None and self.player_snake is None:
            raise ValueError("There are no snakes!")
        if self.player_snake:
            self.snakes.append(self.player_snake)
        self.rewards = [0 for s in self.snakes]
        self.closest_distance = [None for _ in self.snakes]
        self.rng = np.random.RandomState(self.seed)
        self.update_fruits()

    def __iter__(self):
        game_over = False

        while True:
            direction = yield
            logger.debug("New direction: %s", direction)
            for snake in self.snakes:
                if not isinstance(snake, NeuroSnake):
                    continue

                coords = self.reduced_coordinates(snake).flatten()
                self.punish_circles(snake, direction)
                direction = snake.decide_direction(coords)
                snake.update(direction)

                self.check_collisions()

            if not self.snakes:
                game_over = True
            self.update_fruits()
            self.update_distances()

            if game_over:
                break

    def punish_circles(self, snake, new_direction):
        dir_list = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
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
            self.add_fruit(new_x, new_y)

    def add_fruit(self, x, y):
        self.fruits.append((x, y))

    def clear_fruits(self):
        self.fruits = []

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
                self.rewards[idx] -= DISTANCE_REWARD
            self.closest_distance[idx] = new_dist

    def determine_fruit_distances(self):
        if not self.fruits:
            return [0 for _ in self.snakes]

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

        # look north-east
        if direction == Direction.NORTHEAST:
            for x, y in zip(range(head_x + 1, self.width), reversed(range(head_y))):
                if (x, y) in self.fruits:
                    return True

        # look east
        if direction == Direction.EAST:
            for x in range(head_x + 1, self.width):
                if (x, head_y) in self.fruits:
                    return True

        # look south-east
        if direction == Direction.SOUTHEAST:
            for x, y in zip(
                range(head_x + 1, self.width), range(head_y + 1, self.height)
            ):
                if (x, y) in self.fruits:
                    return True

        # look south
        if direction == Direction.SOUTH:
            for y in range(head_y + 1, self.height):
                if (head_x, y) in self.fruits:
                    return True

        # look south-west
        if direction == Direction.SOUTHWEST:
            for x, y in zip(reversed(range(head_x)), range(head_y + 1, self.height)):
                if (x, y) in self.fruits:
                    return True

        # look west
        if direction == Direction.WEST:
            for x in reversed(range(head_x)):
                if (x, head_y) in self.fruits:
                    return True

        # look north-west
        if direction == Direction.NORTHWEST:
            for x, y in zip(reversed(range(head_x)), reversed(range(head_y))):
                if (x, y) in self.fruits:
                    return True

        return False

    def reduced_coordinates(self, snake):
        """
        Returns an array of length eight.
        If the fruit is:
            * in front of the snake: arr[0, 0] == 1
            * in front right of the snake: arr[1, 0] == 1
            * right of the snake: arr[2, 0] == 1
            ...

        Parameters
        ----------
        snake : Snake
        """
        head_x, head_y = snake.coordinates[-1]
        direction = snake.direction
        result = np.zeros((8, 2))

        # look north
        if self.is_wall_or_snake((head_x, head_y - 1)):
            logger.debug("Wall at north")
            result[0, 1] = 1
        if self.fruit_ahead((head_x, head_y), Direction.NORTH):
            logger.debug("Fruit at north")
            result[0, 0] = 1

        # look north-east
        if self.is_wall_or_snake((head_x + 1, head_y - 1)):
            logger.debug("Wall at north-east")
            result[1, 1] = 1
        if self.fruit_ahead((head_x, head_y), Direction.NORTHEAST):
            logger.debug("Fruit at north-east")
            result[1, 0] = 1

        # look east
        if self.is_wall_or_snake((head_x + 1, head_y)):
            logger.debug("Wall at east")
            result[2, 1] = 1
        if self.fruit_ahead((head_x, head_y), Direction.EAST):
            logger.debug("Fruit at east")
            result[2, 0] = 1

        # look south-east
        if self.is_wall_or_snake((head_x + 1, head_y + 1)):
            logger.debug("Wall at south-east")
            result[3, 1] = 1
        if self.fruit_ahead((head_x, head_y), Direction.SOUTHEAST):
            logger.debug("Fruit at south-east")
            result[3, 0] = 1

        # look south
        if self.is_wall_or_snake((head_x, head_y + 1)):
            logger.debug("Wall at south")
            result[4, 1] = 1
        if self.fruit_ahead((head_x, head_y), Direction.SOUTH):
            logger.debug("Wall at south")
            result[4, 0] = 1

        # look south-west
        if self.is_wall_or_snake((head_x - 1, head_y + 1)):
            logger.debug("Wall at south-west")
            result[5, 1] = 1
        if self.fruit_ahead((head_x, head_y), Direction.SOUTHWEST):
            logger.debug("Fruit at south-west")
            result[5, 0] = 1

        # look west
        if self.is_wall_or_snake((head_x - 1, head_y)):
            logger.debug("Wall at west")
            result[6, 1] = 1
        if self.fruit_ahead((head_x, head_y), Direction.WEST):
            logger.debug("Fruit at west")
            result[6, 0] = 1

        # look north-west
        if self.is_wall_or_snake((head_x - 1, head_y - 1)):
            logger.debug("Wall at north-west")
            result[7, 1] = 1
        if self.fruit_ahead((head_x, head_y), Direction.NORTHWEST):
            logger.debug("Fruit at north-west")
            result[7, 0] = 1

        direction_idx = direction.value
        result = np.roll(result, Direction.NORTH.value - direction_idx, axis=0)
        return result
