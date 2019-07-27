"""Classes for rendering the Snake game"""
import curses
import logging
from main import Game, Snake, Direction


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
    def __init__(self, game: Game):
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


class Curses(UI):
    def run(self):
        curses.wrapper(self._loop)

    def _loop(self, screen):
        y, x = screen.getmaxyx()
        assert (
            self.game.width <= x and self.game.height <= y
        ), f"Wrong game dimensions {self.game.width}, {self.game.height} != {x}, {y}!"
        game = self.game
        curses.curs_set(0)
        screen.nodelay(True)

        for i in range(1, 11):
            curses.init_pair(i, curses_colors[i], curses.COLOR_BLACK)
        game_it = iter(game)
        direction = None

        while True:
            screen.clear()
            self.draw(screen)
            screen.refresh()
            curses.napms(70)
            game_it.send(direction)
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


def get_screen_size(screen):
    y, x = screen.getmaxyx()
    return x, y


def main():
    logging.basicConfig(level=logging.DEBUG)
    x, y = curses.wrapper(get_screen_size)
    # Reduce y-size by one to avoid curses scroll problems
    y -= 1
    game = Game(
        x,
        y,
        player_snake=Snake(0, 0, max_x=x, max_y=y, direction=Direction.EAST),
        max_number_of_fruits=300,
    )
    ui = LogPositions(game)
    ui.run()


if __name__ == "__main__":
    main()
