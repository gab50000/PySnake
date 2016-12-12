import curses
import random
from collections import deque


curses_colors = [curses.COLOR_WHITE, curses.COLOR_CYAN, curses.COLOR_BLUE, curses.COLOR_GREEN,
                 curses.COLOR_YELLOW, curses.COLOR_MAGENTA, curses.COLOR_RED, curses.COLOR_RED,
                 curses.COLOR_RED, curses.COLOR_RED, curses.COLOR_RED]


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
        fruits_to_be_deleted = []
        snakes_to_be_deleted = []

        for s_idx, s in enumerate(self.snakes):
            x_s, y_s = s.coordinates[-1]
            # Check fruit collision
            for fruit_idx, (x_f, y_f) in enumerate(self.fruits):
                if (x_s, y_s) == (x_f, y_f):
                    s.length += 2
                    fruits_to_be_deleted.append(fruit_idx)
            # Check snake collisions
            for s2_idx, s2 in enumerate(self.snakes):
                if s_idx != s2_idx:
                    for x2s, y2s in s2.coordinates:
                        if (x_s, y_s) == (x2s, y2s):
                            snakes_to_be_deleted.append(s_idx)
                            return True
                else:
                    for x2s, y2s in list(s2.coordinates)[:-1]:
                        if (x_s, y_s) == (x2s, y2s):
                            snakes_to_be_deleted.append(s_idx)
                            return True

        for tbd in fruits_to_be_deleted:
            self.fruits.pop(tbd)
        for snk in snakes_to_be_deleted:
            self.snakes.pop(snk)


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
            new_direction = direction
        else:
            new_direction = self.direction

        head_x, head_y = self.coordinates[-1]

        # Do not allow 180° turnaround
        if (new_direction, self.direction) in [("N", "S"), ("S", "N"), ("O", "W"), ("W", "O")]:
            new_direction = self.direction

        if new_direction == "N":
            new_x, new_y = head_x, head_y - 1
        elif new_direction == "O":
            new_x, new_y = head_x + 1, head_y
        elif new_direction == "S":
            new_x, new_y = head_x, head_y + 1
        else:
            new_x, new_y = head_x - 1, head_y

        self.direction = new_direction

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

        direction = game.check_input(screen)
        snake.update(direction)
        game_over = game.check_collisions()
        game.update_fruits()
        screen.refresh()
        curses.napms(70)

        if game_over:
            break


if __name__ == "__main__":
    curses.wrapper(main)
    print("Vorbei!")
