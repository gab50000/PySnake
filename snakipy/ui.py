"""Classes for rendering the Snake game"""
import curses
import logging
from itertools import count

import fire
import numpy as np
from abc_algorithm.main import Swarm

from .main import Game
from .optimize import ParameterSearch
from .snake import Direction, NeuroSnake


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

    def draw_fruits(self, canvas):
        raise NotImplementedError

    def draw(self, canvas):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def check_input(self, canvas):
        raise NotImplementedError


class Curses(UI):
    def __init__(
        self, game, *, debug=False, robot=False, generate_data=False, sleep=70
    ):
        super().__init__(game)
        self.debug = debug
        self.robot = robot
        self.generate_data = generate_data
        self.sleep = sleep

    def draw_fruits(self, canvas):
        for x, y in self.game.fruits:
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

        for step in count():
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


def main(
    debug=False,
    robot=False,
    dna_file=None,
    width=None,
    n_fruits=30,
    hidden_size=10,
    sleep=70,
    border=False,
):
    """Play the game"""

    logging.basicConfig(level=logging.DEBUG, filename="snake.log", filemode="a")
    x, y = curses.wrapper(_get_screen_size)
    if width:
        x = width
        y = width
    if dna_file:
        dna = np.load(dna_file)
    else:
        dna = None
    input_size = 16

    game = Game(
        x,
        y,
        player_snake=NeuroSnake(
            x // 2,
            y // 2,
            max_x=x,
            max_y=y,
            input_size=input_size,
            hidden_size=hidden_size,
            dna=dna,
            direction=Direction.SOUTH,
        ),
        max_number_of_fruits=n_fruits,
        border=border,
    )
    ui = Curses(game, debug=debug, robot=robot, sleep=sleep)
    try:
        ui.run()
    except StopIteration:
        print("Game Over")
        print("Score:", *game.rewards)


def training(
    n_optimize=100,
    hidden_size=5,
    max_steps=100,
    search_radius=1,
    log_level="info",
    n_employed=20,
    n_onlooker=20,
    n_fruits=10,
    n_average=10,
    border=False,
    dna_file=None,
    width=20,
    height=None,
    seed=None,
):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        filename="snaketrain.log",
        filemode="w",
    )
    x = width
    y = height if height else x
    input_size = 16
    out_size = 3
    # Reduce y-size by one to avoid curses scroll problems
    game_options = {
        "width": x,
        "height": y,
        "max_number_of_fruits": n_fruits,
        "border": border,
        "seed": seed,
    }
    snake_options = {
        "x": x // 2,
        "y": y // 2,
        "max_x": x,
        "max_y": y,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "direction": Direction.SOUTH,
    }

    if dna_file:
        try:
            dna = np.load(dna_file)
        except FileNotFoundError:
            logger.error("File not found")
            dna = None
    else:
        dna = np.random.normal(
            size=(input_size + 1) * hidden_size + (hidden_size + 1) * out_size,
            loc=0,
            scale=1.0,
        )

    ui = ParameterSearch(
        game_options, snake_options, max_steps=max_steps, n_average=n_average, dna=dna
    )
    swarm = Swarm(
        ui.benchmark,
        (input_size + 1) * hidden_size + (hidden_size + 1) * out_size,
        n_employed=n_employed,
        n_onlooker=n_onlooker,
        limit=10,
        max_cycles=n_optimize,
        lower_bound=-1,
        upper_bound=1,
        search_radius=search_radius,
    )
    for result in swarm.run():
        logger.info("Saving to %s", dna_file)
        np.save(dna_file, result)


def entrypoint():
    fire.Fire({"main": main, "training": training})
