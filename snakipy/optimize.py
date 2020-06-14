import logging
import random

import fire
import numpy as np
from abc_algorithm import Swarm

from snakipy.game import Game
from snakipy.snake import NeuroSnake, Direction
from snakipy.ui import CursesUI, PygameUI

logger = logging.getLogger(__name__)


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
            direction = player_snake.decide_direction(
                game.reduced_coordinates(player_snake).flatten()
            )
        logger.debug("Stopped after %s steps", step)
        (game_score,) = game.rewards
        logger.info("Total score: %s", game_score)
        return game_score


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

    opt = ParameterSearch(
        game_options, snake_options, max_steps=max_steps, n_average=n_average, dna=dna
    )
    swarm = Swarm(
        opt.benchmark,
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
        game = Game(
            **game_options,
            player_snake=NeuroSnake(**snake_options, dna=np.load(dna_file)),
        )
        ui = CursesUI(game, robot=True, n_steps=max_steps)
        try:
            ui.run()
        except StopIteration:
            pass


def create_snakes(size, n_x, n_y, dnas=None):

    dist_x = size[0] // (n_x + 1)
    dist_y = size[1] // (n_y + 1)

    x_pos = np.arange(dist_x // 2, size[0], dist_x)
    y_pos = np.arange(dist_y // 2, size[1], dist_y)
    X, Y = np.meshgrid(x_pos, y_pos)

    if not dnas:
        dnas = [None] * x_pos.size * y_pos.size
    snakes = [
        NeuroSnake(
            x,
            y,
            *size,
            input_size=16,
            hidden_size=5,
            direction=random.choice(
                [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
            ),
            dna=dna,
        )
        for x, y, dna in zip(X.flatten(), Y.flatten(), dnas)
    ]
    return snakes


def snake_evolution(dnafile):
    size = (80, 60)

    snakes = create_snakes(size, 8, 6)

    game = Game(*size, snakes=snakes, border=True)
    ui = PygameUI(game, size=(80, 60))
    ui.run()


def cli():
    fire.Fire()
