"""Classes for rendering the Snake game"""
import curses
import logging
import multiprocessing

import fire
import numpy as np
from scipy.optimize import minimize
from tqdm import trange

from main import Game
from snake import Direction, Snake, NeuroSnake


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
    def __init__(self, game, *, debug=False, robot=False):
        super().__init__(game)
        self.debug = debug
        self.robot = robot

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
            if self.debug:
                arr = self.game.get_surrounding_view(player_snake, onehot=True)
                arr = np.argmax(arr, axis=-1)
                self.debug_msg(screen, str(arr))
            self.draw(screen)
            screen.refresh()
            curses.napms(70)
            game_it.send(direction)
            if self.robot:
                direction = player_snake.decide_direction(
                    self.game.get_surrounding_view(player_snake, onehot=True).flatten()
                )
            else:
                direction = self.check_input(screen)


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
    def __init__(self, game_options, snake_options, search_radius, max_steps=10_000):
        self.game_options = game_options
        self.snake_options = snake_options
        self.max_steps = max_steps
        self.search_radius = search_radius

    def benchmark(self, dna, n=10):
        score = 0
        for _ in range(n):
            game = Game(
                **self.game_options,
                player_snake=NeuroSnake(**self.snake_options, dna=dna),
            )
            score += self.run(game)
        return score / n

    def optimize(self, start_dna, n_optimize=1000, n=10):
        current_score = self.benchmark(start_dna, n=n)
        print("Current score:", current_score)
        dna = start_dna
        for i in range(n_optimize):
            logger.info("Epoch %s", i)
            new_dna = dna + np.random.normal(
                loc=0, scale=self.search_radius / np.sqrt(dna.size)
            )
            new_score = self.benchmark(new_dna, n=n)
            if new_score > current_score:
                print("Best score:", new_score)
                np.save("best_dna", new_dna)
                current_score = new_score
                dna = new_dna

    def run(self, game):
        game_it = iter(game)
        direction = None
        player_snake = game.player_snake

        for step in range(self.max_steps):
            try:
                game_it.send(direction)
            except StopIteration:
                break
            direction = player_snake.decide_direction(
                game.get_surrounding_view(player_snake, onehot=True).flatten()
            )
        logger.info("Stopped after %s steps", step)
        return game.rewards[0]


def get_screen_size(screen):
    y, x = screen.getmaxyx()
    return x, y


def main(
    debug=False,
    robot=False,
    dna_file=None,
    width=None,
    height=None,
    n_fruits=30,
    hidden_size=10,
):
    logging.basicConfig(level=logging.DEBUG)
    x, y = curses.wrapper(get_screen_size)
    if width:
        x = width
    if height:
        y = height
    # Reduce y-size by one to avoid curses scroll problems
    y -= 1
    if dna_file:
        dna = np.load("best_dna.npy")
    else:
        dna = None
    game = Game(
        x,
        y,
        player_snake=NeuroSnake(
            0,
            0,
            max_x=x,
            max_y=y,
            input_size=75,
            hidden_size=10,
            dna=dna,
            direction=Direction.EAST,
        ),
        max_number_of_fruits=n_fruits,
    )
    ui = Curses(game, debug=debug, robot=robot)
    ui.run()


def training(
    n_optimize=100,
    input_size=75,
    hidden_size=10,
    max_steps=100,
    search_radius=1,
    log_level="info",
):
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    x, y = 100, 100
    # Reduce y-size by one to avoid curses scroll problems
    game_options = {"width": x, "height": y, "max_number_of_fruits": 100}
    snake_options = {
        "x": 0,
        "y": 0,
        "max_x": x,
        "max_y": y,
        "input_size": 75,
        "hidden_size": 10,
    }
    ui = ParameterSearch(
        game_options, snake_options, max_steps=max_steps, search_radius=search_radius
    )
    # minimize(
    #    ui.benchmark,
    #    x0=np.random.rand((input_size + 1) * hidden_size + (hidden_size + 1) * 4),
    #    options=dict(disp=True),
    # )
    ui.optimize(
        start_dna=np.zeros((input_size + 1) * hidden_size + (hidden_size + 1) * 3),
        n_optimize=n_optimize,
    )


if __name__ == "__main__":
    fire.Fire()
