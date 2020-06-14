import curses
import logging

from abc_algorithm import Swarm
import fire
import numpy as np

from snakipy.game import Game
from snakipy.optimize import ParameterSearch
from snakipy.snake import NeuroSnake, Direction
from snakipy.ui import _get_screen_size, CursesUI, PygameUI


logger = logging.getLogger(__name__)


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
    ui = PygameUI(game, debug=debug, robot=robot, sleep=sleep)
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
            player_snake=NeuroSnake(**snake_options, dna=np.load(dna_file))
        )
        ui = CursesUI(game, robot=True, n_steps=max_steps)
        try:
            ui.run()
        except StopIteration:
            pass


def entrypoint():
    fire.Fire({"main": main, "training": training})
