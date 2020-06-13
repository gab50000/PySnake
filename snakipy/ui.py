"""Classes for rendering the Snake game"""
import curses
import logging
from itertools import count, islice

import numpy as np
import pygame

from .game import Game
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


def interpret_snake_sensor(arr):
    ii, jj = np.where(arr)

    result = []

    for i, j in zip(ii, jj):
        if j == 0:
            result.append(f"Fruit at {list(Direction)[i].name.title()}")
        else:
            result.append(f"Wall at {list(Direction)[i].name.title()}")
    return "\n".join(result)


class UI:
    """
    Base class for all user interfaces
    """

    def __init__(self, game: Game, **kwargs):
        self.game = game

    def draw(self, canvas):
        raise NotImplementedError

    def draw_fruits(self, canvas):
        for x, y in self.game.fruits:
            self.draw_fruit(canvas, x, y)

    def draw_fruit(self, canvas, x, y):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def check_input(self, canvas):
        raise NotImplementedError


class Curses(UI):
    def __init__(
        self,
        game,
        *,
        debug=False,
        robot=False,
        generate_data=False,
        sleep=70,
        n_steps=None,
    ):
        super().__init__(game)
        self.debug = debug
        self.robot = robot
        self.generate_data = generate_data
        self.sleep = sleep
        self.n_steps = n_steps

    def draw_fruit(self, canvas, x, y):
        canvas.addstr(y, x, "O", curses.color_pair(6))

    def draw_snake(self, screen, snake):
        for x, y in snake.coordinates:
            screen.addstr(y, x, "X", curses.color_pair(3))

    def draw(self, canvas):
        self.draw_fruits(canvas)
        for snake in self.game.snakes:
            self.draw_snake(canvas, snake)

    def check_input(self, canvas):
        inp = canvas.getch()
        if inp == curses.KEY_UP:
            direction = Direction.NORTH
        elif inp == curses.KEY_DOWN:
            direction = Direction.SOUTH
        elif inp == curses.KEY_LEFT:
            direction = Direction.WEST
        elif inp == curses.KEY_RIGHT:
            direction = Direction.EAST
        else:
            direction = None
        return direction

    def run(self):
        curses.wrapper(self._loop)

    def debug_msg(self, screen, msg):
        screen.addstr(0, 0, msg)

    def _loop(self, screen):
        y, x = screen.getmaxyx()
        assert (
            self.game.width <= x and self.game.height <= y
        ), f"Wrong game dimensions {self.game.width}, {self.game.height} != {x}, {y}!"
        y -= 1
        game = self.game
        player_snake = self.game.player_snake
        curses.curs_set(0)
        screen.nodelay(True)

        for i in range(1, 11):
            curses.init_pair(i, curses_colors[i], curses.COLOR_BLACK)
        game_it = iter(game)
        direction = None

        for step in islice(count(), self.n_steps):
            logger.debug(step)
            screen.clear()
            coords = self.game.reduced_coordinates(player_snake)
            fruit_dir = interpret_snake_sensor(coords)
            if fruit_dir:
                logger.debug(fruit_dir)
            coords = coords.flatten()
            # coords = self.game.state_array.flatten()
            if self.debug:
                # arr = self.game.reduced_coordinates(player_snake)
                self.debug_msg(
                    screen,
                    str(
                        [
                            coords,
                            player_snake.net_output,
                            game.rewards,
                            step,
                            # player_snake.direction,
                        ]
                    ),
                )
            self.draw(screen)
            screen.refresh()
            curses.napms(self.sleep)
            game_it.send(direction)
            player_input = self.check_input(screen)
            if player_input is None and self.robot:
                direction = player_snake.decide_direction(coords)
            else:
                direction = player_input

            if self.generate_data:
                pass


def _get_screen_size(screen):
    y, x = screen.getmaxyx()
    return x, y


def get_screen_size():
    print(curses.wrapper(_get_screen_size))
