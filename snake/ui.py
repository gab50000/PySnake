"""Classes for rendering the Snake game"""
import curses
from main import Game, Snake


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
            direction = "N"
        elif inp == curses.KEY_DOWN:
            direction = "S"
        elif inp == curses.KEY_LEFT:
            direction = "W"
        elif inp == curses.KEY_RIGHT:
            direction = "O"
        else:
            direction = None
        return direction


class Curses(UI):
    def run(self):
        curses.wrapper(self._loop)

    def _loop(self, screen):
        game = self.game
        curses.curs_set(0)
        screen.nodelay(True)
        y, x = screen.getmaxyx()
        # Reduce y-size by one to avoid curses scroll problems
        y -= 1

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


def main():
    game = Game(30, 20, snakes=[Snake(10, 10, 30, 20, "O")])
    ui = Curses(game)
    ui.run()


if __name__ == "__main__":
    main()
