"""Classes for rendering the Snake game"""
import curses
import logging
from dataclasses import dataclass
from itertools import count, islice
from typing import Optional, Tuple

import numpy as np
import pygame

from snakipy.game import Game, BoardState
from snakipy.snake import Direction, NeuroSnake

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


@dataclass
class UI:
    """
    Base class for all user interfaces
    """

    game: Game
    n_steps: Optional[int] = None
    debug: bool = False
    robot: bool = False
    fps: int = 20

    def __post_init__(self):
        self.sleep = 1 / self.fps

    def draw(self, canvas):
        for snake in self.game.snakes:
            for x, y in snake.coordinates:
                self.draw_snake_element(canvas, x, y)

        for x, y in self.game.fruits:
            self.draw_fruit(canvas, x, y)

    def draw_snake_element(self, canvas, x, y):
        raise NotImplementedError

    def draw_fruit(self, canvas, x, y):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def check_input(self, canvas):
        raise NotImplementedError

    def get_canvas_size(self, canvas=None):
        raise NotImplementedError

    def clear(self, canvas):
        raise NotImplementedError

    def debug_msg(self, canvas, message):
        pass

    def _loop(self, canvas):
        x, y = self.get_canvas_size(canvas)
        assert (
            self.game.width <= x and self.game.height <= y
        ), f"Wrong game dimensions {self.game.width}, {self.game.height} != {x}, {y}!"
        y -= 1
        game = self.game
        player_snake = self.game.player_snake
        game_it = iter(game)
        direction = None

        for step in islice(count(), self.n_steps):
            logger.debug(step)
            self.clear(canvas)
            self.draw(canvas)
            self.refresh(canvas)
            self.nap()
            game_it.send(direction)
            if player_snake:
                player_input = self.check_input(canvas)
                direction = player_input
            else:
                direction = None

    def nap(self):
        raise NotImplementedError

    def refresh(self, canvas):
        raise NotImplementedError


@dataclass
class CursesUI(UI):
    size: Tuple[int, int] = None

    def draw_fruit(self, canvas, x, y):
        canvas.addstr(y, x, "O", curses.color_pair(6))

    def draw_snake_element(self, canvas, x, y):
        canvas.addstr(y, x, "X", curses.color_pair(3))

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

    def _loop(self, canvas):
        curses.curs_set(0)
        canvas.nodelay(True)
        for i in range(1, 11):
            curses.init_pair(i, curses_colors[i], curses.COLOR_BLACK)
        super()._loop(canvas)

    def debug_msg(self, screen, msg):
        screen.addstr(0, 0, msg)

    def clear(self, canvas):
        canvas.clear()

    def refresh(self, canvas):
        canvas.refresh()

    def nap(self):
        curses.napms(int(1000 * self.sleep))

    @staticmethod
    def _get_screen_size(screen):
        y, x = screen.getmaxyx()
        return x, y

    def get_canvas_size(self, canvas=None):
        y, x = canvas.getmaxyx()
        return x, y


@dataclass
class PygameUI(UI):
    size: Tuple[int, int] = (20, 20)
    canvas_size: Tuple[int, int] = (800, 600)

    def __post_init__(self):
        self._pixel_width = self.canvas_size[0] // self.size[0]
        self._pixel_height = self.canvas_size[1] // self.size[1]

    def nap(self):
        self._fps.tick(self.fps)

    def clear(self, canvas):
        canvas.fill((0, 0, 0))

    def refresh(self, canvas):
        pygame.display.update()

    def draw_snake_element(self, canvas, x, y):
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            (
                x * self._pixel_width,
                y * self._pixel_height,
                self._pixel_width,
                self._pixel_height,
            ),
        )

    def get_canvas_size(self, canvas=None):
        return self.canvas_size

    def draw_fruit(self, canvas, x, y):
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (x * self._pixel_width, y * self._pixel_height),
            self._pixel_width,
        )

    def run(self):
        pygame.init()
        self._fps = pygame.time.Clock()
        window = pygame.display.set_mode(self.canvas_size)
        self._loop(window)

    def check_input(self, canvas):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    return Direction.WEST
                elif event.key == pygame.K_RIGHT:
                    return Direction.EAST
                elif event.key == pygame.K_DOWN:
                    return Direction.SOUTH
                elif event.key == pygame.K_UP:
                    return Direction.NORTH
