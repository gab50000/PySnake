from collections import deque
import curses
import random
import sys

import numpy as np


curses_colors = (curses.COLOR_WHITE, curses.COLOR_CYAN, curses.COLOR_BLUE, curses.COLOR_GREEN,
                 curses.COLOR_YELLOW, curses.COLOR_MAGENTA, curses.COLOR_RED, curses.COLOR_RED,
                 curses.COLOR_RED, curses.COLOR_RED, curses.COLOR_RED)


class Game:
    def __init__(self, width, height, snakes, *, max_number_of_fruits=1, log=None):
        self.fruits = []
        self.snakes = snakes
        self.width, self.height = width, height
        self.log = log
        self.max_number_of_fruits = max_number_of_fruits

    def update_fruits(self):
        """Add fruits to the game until max_number_of_fruits is reached."""
        while True:
            if len(self.fruits) < self.max_number_of_fruits:
                new_x, new_y = random.randint(1, self.width - 1), random.randint(1, self.height - 1)
                self.fruits.append((new_x, new_y))
            else:
                break

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

    def return_state_array(self):
        """Return array of current state.
        Snake bodies are encoded as ones, snake heads as twos,
        fruits are 3s"""
        state = np.zeros((self.width, self.height))
        for snake in self.snakes:
            for i in range(len(snake.coordinates) - 1):
                x, y = snake.coordinates[i]
                state[x, y] = 1
            state[snake.coordinates[-1]] = 2

        for x, y in self.fruits:
           state[x, y] = 3
        return state

class Snake:
    def __init__(self, x, y, max_x, max_y, direction):
        self.coordinates = deque([(x, y)])
        self.max_x, self.max_y = max_x, max_y
        self.direction = direction
        self.length = 1

    @classmethod
    def random_init(self, width, height):
        start_direction = "N O S W".split()[random.randint(0, 3)]
        x, y = random.randint(1, width - 1), random.randint(1, height - 1)
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


class NeuroSnake(Snake):
    def __init__(self, x, y, max_x, max_y, direction):
        super().__init__(x, y, max_x, max_y, direction)
        input_size = max_x * max_y
        hidden_size = input_size
        output_size = 4
        self.brain = FFN(input_size, hidden_size, output_size)

        self.fruits_eaten = 0
        self.age = 0

    def decide_direction(self, state):
        direction = ("N", "O", "S", "W")[np.argmax(self.brain.prop(state))]
        self.update(direction)
        self.age += 1


def main(screen):
    curses.curs_set(0)
    screen.nodelay(True)
    y, x = screen.getmaxyx()
    # Reduce y-size by one to avoid curses scroll problems
    y -= 1

    for i in range(1, 11):
        curses.init_pair(i, curses_colors[i], curses.COLOR_BLACK)

    snake = Snake.random_init(x, y)
    game = Game(x, y, [snake], max_number_of_fruits=10)
    game.update_fruits()

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


def nn_training(screen):
    curses.curs_set(0)
    screen.nodelay(True)
    y, x = screen.getmaxyx()
    y, x = min(y - 1, 100), min(x, 100)

    for i in range(1, 11):
        curses.init_pair(i, curses_colors[i], curses.COLOR_BLACK)

    net = FFN(y*x, 100, 4)
    directions = "N O S W".split()

    snake = Snake.random_init(x, y)
    game = Game(x, y, [snake], max_number_of_fruits=10)
    game.update_fruits()

    while 1:
        screen.clear()
        game.draw(screen)
        snake.draw(screen)
        state = game.return_state_array()

        direction = directions[np.argmax(net.prop(state.ravel()))]
        snake.update(direction)
        game_over = game.check_collisions()
        game.update_fruits()
        screen.refresh()
        curses.napms(200)

        if game_over:
            break


if __name__ == "__main__":
    curses.wrapper(nn_training)
    print("Vorbei!")
