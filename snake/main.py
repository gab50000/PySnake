import curses
import random
from collections import deque


curses_colors = [curses.COLOR_WHITE, curses.COLOR_CYAN, curses.COLOR_BLUE, curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_MAGENTA, curses.COLOR_RED, curses.COLOR_RED, curses.COLOR_RED, curses.COLOR_RED, curses.COLOR_RED]


class Game:
    def __init__(self, width, height, snakes, *, log=None):
        self.fruits = []
        self.snakes = snakes
        self.width, self.height = width, height
        self.log = log

    def update_fruits(self):
        if not self.fruits:
            new_x, new_y = random.randint(1, self.width - 1), random.randint(1, self.height - 1)
            self.fruits.append((new_x, new_y))

    def draw(self, screen):
        for x, y in self.fruits:
            screen.addstr(y, x, "O", curses.color_pair(6))

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

    def check_collisions(self):
        for s in self.snakes:
            x_s, y_s = s.coordinates[-1]
            to_be_deleted = []
            for i, (x_f, y_f) in enumerate(self.fruits):
                if (x_s, y_s) == (x_f, y_f):
                    s.length += 1
                    to_be_deleted.append(i)
        for tbd in to_be_deleted:
            self.fruits.pop(tbd)


class Snake:
    def __init__(self, x, y, max_x, max_y, direction):
        self.coordinates = deque([(x, y)])
        self.max_x, self.max_y = max_x, max_y
        self.direction = direction
        self.length = 1

    @classmethod
    def random_init(self, width, height):
        start_direction = "N O S W".split()[random.randint(0, 3)]
        x, y = random.randint(1, width -1), random.randint(1, height - 1)
        return Snake(x, y, width, height, start_direction)

    def update(self, direction):
        if direction:
            self.direction = direction

        head_x, head_y = self.coordinates[-1]
        if self.direction == "N":
            new_x, new_y = head_x, head_y - 1
        elif self.direction == "O":
            new_x, new_y = head_x + 1, head_y
        elif self.direction == "S":
            new_x, new_y = head_x, head_y + 1
        else:
            new_x, new_y = head_x - 1, head_y

        new_x %= self.max_x
        new_y %= self.max_y

        self.coordinates.append((new_x, new_y))
        if len(self.coordinates) > self.length:
            self.coordinates.popleft()

    def draw(self, screen):
        for x, y in self.coordinates:
            screen.addstr(y, x, "X", curses.color_pair(3))


def main(screen):
    log = open("log", "w")
    curses.curs_set(0)
    screen.nodelay(True)
    y, x = screen.getmaxyx()

    for i in range(1, 11):
        curses.init_pair(i, curses_colors[i], curses.COLOR_BLACK) 

    snake = Snake.random_init(x, y)
    game = Game(x, y, [snake], log=log)
    game.update_fruits()
    direction = None

    while 1:
        screen.clear()
        game.draw(screen)
        snake.draw(screen)

        screen.border(0)
        direction = game.check_input(screen)
        snake.update(direction)
        game.check_collisions()
        game.update_fruits()
        screen.refresh()
        curses.napms(50)


if __name__ == "__main__":
    curses.wrapper(main)
