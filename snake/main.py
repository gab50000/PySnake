from collections import deque
import curses
import random
import sys

import numpy as np

from q_learning import FFN


curses_colors = (curses.COLOR_WHITE, curses.COLOR_CYAN, curses.COLOR_BLUE, curses.COLOR_GREEN,
                 curses.COLOR_YELLOW, curses.COLOR_MAGENTA, curses.COLOR_RED, curses.COLOR_RED,
                 curses.COLOR_RED, curses.COLOR_RED, curses.COLOR_RED)


class Game:
    def __init__(self, width, height, snakes, *, max_number_of_fruits=1, max_number_of_snakes=1,
                 log=None):
        self.fruits = []
        self.snakes = snakes
        self.width, self.height = width, height
        self.log = log
        self.max_number_of_fruits = max_number_of_fruits
        self.max_number_of_snakes = max_number_of_snakes
        self.rewards = [0 for s in snakes]

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
            self.rewards[s_idx] = 0
            x_s, y_s = s.coordinates[-1]
            # Check fruit collision
            for fruit in self.fruits:
                if (x_s, y_s) == fruit:
                    s.length += 2
                    fruits_to_be_deleted.append(fruit)
                    self.rewards[s_idx] = 10
            # Check snake collisions
            for s2_idx, s2 in enumerate(self.snakes):
                if s_idx != s2_idx:
                    for x2s, y2s in s2.coordinates:
                        if (x_s, y_s) == (x2s, y2s):
                            snakes_to_be_deleted.append(s)
                            self.rewards[s_idx] = -100
                else:
                    for x2s, y2s in list(s2.coordinates)[:-1]:
                        if (x_s, y_s) == (x2s, y2s):
                            snakes_to_be_deleted.append(s)
                            self.rewards[s_idx] = -100

        for tbd in fruits_to_be_deleted:
            self.fruits.remove(tbd)
        for snk in snakes_to_be_deleted:
            self.snakes.remove(snk)

    def return_reward(self, snake_idx):
        return self.rewards[snake_idx]


    def return_state_array(self):
        """Return array of current state.
        The game board is encoded as follows:
        Snake body: 1
        Snake head: 2
        Fruit : 3
        Snake head on fruit: 4"""

        state = np.zeros((self.width, self.height))
        for snake in self.snakes:
            for i in range(len(snake.coordinates) - 1):
                x, y = snake.coordinates[i]
                state[x, y] = 1
            head_coord = snake.coordinates[-1]
            if head_coord in self.fruits:
                state[snake.coordinates[-1]] = 4
                self.fruits.remove(head_coord)
            else:
                state[snake.coordinates[-1]] = 2

        for x, y in self.fruits:
           state[x, y] = 3
        return state.flatten()

    def create_new_snakes(self):
        if len(self.snakes) < self.max_number_of_snakes:
            self.snake_pool.sort(key=lambda x:x.fitness())
            fitness_cumsum = np.cumsum([s.fitness() for s in self.snake_pool])
        while len(self.snakes) < self.max_number_of_snakes:
            s1 = self.snake_pool[np.searchsorted(fitness_cumsum,
                                                 np.random.randint(fitness_cumsum[-1]))]
            s2 = self.snake_pool[np.searchsorted(fitness_cumsum,
                                                 np.random.randint(fitness_cumsum[-1]))]
            new_snake = NeuroSnake.from_parents(self.width, self.height, s1, s2, 0.3, 0.1)
            self.snake_pool.append(new_snake)
            self.snakes.append(new_snake)

class Snake:
    def __init__(self, x, y, max_x, max_y, direction):
        self.coordinates = deque([(x, y)])
        self.max_x, self.max_y = max_x, max_y
        self.direction = direction
        self.length = 1

    @classmethod
    def random_init(cls, width, height):
        start_direction = "N O S W".split()[random.randint(0, 3)]
        x, y = random.randint(1, width - 1), random.randint(1, height - 1)
        return cls(x, y, width, height, start_direction)

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

    def return_direction_estimations(self, state):
        return self.brain.prop_and_remember(state)


def main(screen):
    curses.curs_set(0)
    screen.nodelay(True)
    y, x = screen.getmaxyx()
    # Reduce y-size by one to avoid curses scroll problems
    y -= 1

    for i in range(1, 11):
        curses.init_pair(i, curses_colors[i], curses.COLOR_BLACK)

    snake = Snake.random_init(x, y)
    game = Game(x, y, [snake], max_number_of_fruits=100)
    game.update_fruits()
    game_over = False

    while 1:
        screen.clear()
        game.draw(screen)
        snake.draw(screen)

        direction = game.check_input(screen)
        snake.update(direction)
        game.check_collisions()
        if not game.snakes:
            game_over = True
        game.update_fruits()
        screen.refresh()
        curses.napms(70)

        if game_over:
            break


def nn_training(screen):
    curses.curs_set(0)
    screen.nodelay(True)
    y, x = screen.getmaxyx()
    y, x = min(y - 1, 20), min(x, 20)

    for i in range(1, 11):
        curses.init_pair(i, curses_colors[i], curses.COLOR_BLACK)

    number_of_snakes = 1
    number_of_epochs = 100
    game_steps = 200
    gamma = 0.9  # discount factor
    temperature = 20  # Increases exploration

    net = FFN(x * y, x * y, 4)

    for epoch in range(number_of_epochs):
        snakes = []
        for i in range(number_of_snakes):
            xx, yy = np.random.randint(x), np.random.randint(y)
            direction = ("N", "O", "S", "W")[np.random.randint(4)]
            snakes.append(Snake(xx, yy, x, y, direction))

        snake = snakes[0]
        net.clear_memory()

        rewards = []
        actions = []

        game = Game(x, y, snakes, max_number_of_fruits=10, max_number_of_snakes=10)
        game.update_fruits()
        game_over = False

        for gs in range(game_steps):
            screen.clear()
            game.draw(screen)
            state = game.return_state_array()
            snake.draw(screen)
            #screen.addstr(0, 0, "Epoch {}, Step {}, Reward {}".format(epoch, gs, rewards[-1:]))
            direction_evals = net.prop_and_remember(state)
            screen.addstr(0, 0, "Probs: {}".format(direction_evals))
            direction_evals = np.exp(np.log(direction_evals) / temperature)
            direction_evals /= direction_evals.sum()
            direction = np.argmax(np.random.multinomial(1, direction_evals))
            actions.append(direction)
            snake.update(("N", "O", "S", "W")[direction])
            game.check_collisions()
            rewards.append(game.return_reward(0))
            if not game.snakes:
                game_over = True
            game.update_fruits()
            screen.refresh()
            curses.napms(1)
            if game_over:
                break
        else:
            # at least it didn't die, so give it a small reward
            rewards[-1] = 1
        R = 0
        discounted_reward = np.zeros_like(rewards, dtype=float)
        for t in reversed(range(discounted_reward.shape[0])):
            R = gamma * R + rewards[t]
            discounted_reward[t] = R
        discounted_reward -= discounted_reward.mean()
        discounted_reward /= np.std(discounted_reward)

        net.backprop_value(actions, discounted_reward)

    net.save_state()



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "neuro":
        curses.wrapper(nn_training)
    else:
        curses.wrapper(main)
    print("Vorbei!")
