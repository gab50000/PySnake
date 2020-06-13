import logging

from snakipy.main import Game
from snakipy.snake import NeuroSnake


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
