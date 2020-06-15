import logging

import fire
import numpy as np

from snakipy.game import Game
from snakipy.optimize import training
from snakipy.snake import NeuroSnake, Direction
from snakipy.ui import CursesUI, PygameUI


logger = logging.getLogger(__name__)


def main(
    debug=False,
    robot=False,
    dna_file=None,
    width=20,
    height=None,
    n_fruits=30,
    hidden_size=10,
    fps=20,
    border=False,
    ui="curses",
):
    """Play the game"""

    UIClass = {"curses": CursesUI, "pygame": PygameUI}.get(ui.lower())

    logging.basicConfig(level=logging.DEBUG, filename="snake.log", filemode="a")
    if not height:
        height = width
    if dna_file:
        dna = np.load(dna_file)
    else:
        dna = None
    input_size = 16

    game = Game(
        width,
        height,
        player_snake=NeuroSnake.new_snake(
            width // 2,
            height // 2,
            board_width=width,
            board_height=height,
            input_size=input_size,
            hidden_size=hidden_size,
            dna=dna,
            direction=Direction.SOUTH,
        ),
        max_number_of_fruits=n_fruits,
        border=border,
    )
    ui = UIClass(game, debug=debug, robot=robot, fps=fps, size=(width, height))
    try:
        ui.run()
    except StopIteration:
        print("Game Over")
        print("Score:", *game.rewards)


def cli():
    fire.Fire({"main": main, "training": training})
