import curses
import random


curses_colors = [curses.COLOR_WHITE, curses.COLOR_CYAN, curses.COLOR_BLUE, curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_MAGENTA, curses.COLOR_RED, curses.COLOR_RED, curses.COLOR_RED, curses.COLOR_RED, curses.COLOR_RED]


class Game:
    def __init__(self, width, height):
        self.fruits = []
        self.worms = []
        self.width, self.height = width, height

    def update_fruits(self):
        if not self.fruits:
            new_x, new_y = random.randint(1, self.width - 1), random.randint(1, self.height - 1)
            self.fruits.append((new_y, new_x))

    def draw(self, screen):
        screen.addstr(*self.fruits[0], "O", curses.color_pair(6))

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


class Snake:
    def __init__(self, x, y, max_x, max_y, direction):
        self.x, self.y = x, y
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
        if self.direction == "N":
            self.y -= 1
        elif self.direction == "O":
            self.x += 1
        elif self.direction == "S":
            self.y += 1
        else:
            self.x -= 1
        self.x %= self.max_x
        self.y %= self.max_y

    def draw(self, screen):
        screen.addstr(self.y, self.x, "X", curses.color_pair(3))


def main(screen):
    #  screen = curses.initscr()
    #  screen.keypad(True)
    #  curses.curs_set(0)
    #  curses.start_color()
    y, x = screen.getmaxyx()

    #  for i in range(1, 11):
        #  curses.init_pair(i, curses_colors[i], curses.COLOR_BLACK) 


    game = Game(x, y)
    snake = Snake.random_init(x, y)
    game.update_fruits()
    direction = None

    while 1:
        screen.clear()
        snake.update(direction)
        game.draw(screen)
        snake.draw(screen)

        screen.border(0)
        direction = game.check_input(screen)
        snake.update(direction)
        screen.refresh()
        curses.napms(100)
    print("Ende")


if __name__ == "__main__":
    curses.wrapper(main)

