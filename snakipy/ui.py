"""Classes for rendering the Snake game"""
import curses
import logging
import multiprocessing

import fire
import numpy as np
from tqdm import trange
from abc_algorithm.main import Swarm

from .main import Game
from .snake import Direction, Snake, NeuroSnake


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


class UI:
    def __init__(self, game: Game, **kwargs):
        self.game = game

    def draw_fruits(self, screen):
        for x, y in self.game.fruits:
            screen.addstr(y, x, "O", curses.color_pair(6))

    def draw_snake(self, screen, snake):
        for x, y in snake.coordinates:
            screen.addstr(y, x, "X", curses.color_pair(3))

    def draw(self, screen):
        self.draw_fruits(screen)
        for snake in self.game.snakes:
            self.draw_snake(screen, snake)

    def run(self):
        pass


class Curses(UI):
    def __init__(
        self, game, *, debug=False, robot=False, generate_data=False, sleep=70
    ):
        super().__init__(game)
        self.debug = debug
        self.robot = robot
        self.generate_data = generate_data
        self.sleep = sleep

    def check_input(self, screen):
        inp = screen.getch()
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

        while True:
            screen.clear()
            # coords = self.game.reduced_coordinates(player_snake).flatten()
            coords = self.game.state_array.flatten()
            if self.debug:
                # arr = self.game.reduced_coordinates(player_snake)
                self.debug_msg(
                    screen,
                    str(
                        [
                            coords,
                            player_snake.net_output,
                            game.rewards,
                            player_snake.direction,
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


class LogPositions(UI):
    def run(self):
        for _ in self.game:
            for i, snake in enumerate(self.game.snakes):
                print(f"{i}) {snake} (reward: {self.game.rewards}")


class LogStates(UI):
    def run(self):
        for _ in self.game:
            print(self.game.state_array)


class ParameterSearch:
    def __init__(
        self, game_options, snake_options, max_steps=10_000, n_average=10, dna=None
    ):
        self.game_options = game_options
        self.snake_options = snake_options
        self.max_steps = max_steps
        self.n_average = n_average
        self.dna = dna

    def benchmark(self, dna):
        score = 0
        for _ in range(self.n_average):
            game = Game(
                **self.game_options,
                player_snake=NeuroSnake(**self.snake_options, dna=dna),
            )
            score += self.run(game)
        return -score / self.n_average

    def run(self, game):
        game_it = iter(game)
        direction = None
        player_snake = game.player_snake

        for step in range(self.max_steps):
            try:
                game_it.send(direction)
            except StopIteration:
                break
            direction = player_snake.decide_direction(game.state_array.flatten())
        logger.debug("Stopped after %s steps", step)
        return game.rewards[0]


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
    logging.basicConfig(level=logging.DEBUG)
    x, y = curses.wrapper(_get_screen_size)
    if width:
        x = width
        y = width
    if dna_file:
        dna = np.load(dna_file)
    else:
        dna = None
    input_size = 6

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
    hidden_size=10,
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
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    x = width
    y = height if height else x
    input_size = x * y * 2
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
            size=(input_size + 1) * hidden_size + (hidden_size + 1) * 3,
            loc=0,
            scale=1.0,
        )

    ui = ParameterSearch(
        game_options, snake_options, max_steps=max_steps, n_average=n_average, dna=dna
    )
    swarm = Swarm(
        ui.benchmark,
        (input_size + 1) * hidden_size + (hidden_size + 1) * 3,
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
    fire.Fire()
