from collections import deque
import curses
from enum import Enum
import random
import sys

import numpy as np

from q_learning import FFN


FRUIT_REWARD = 30
DEATH_REWARD = -50

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

Direction = Enum("Direction", "NORTH EAST SOUTH WEST")


class Game:
    """
    Contains and manages the game state
    """

    def __init__(
        self,
        width,
        height,
        *,
        snakes=None,
        player_snake=None,
        max_number_of_fruits=1,
        max_number_of_snakes=1,
        log=None
    ):
        self.fruits = []

        if snakes is None and player_snake is None:
            raise ValueError("There are no snakes!")
        if snakes is None:
            snakes = []
        self.snakes = snakes
        self.player_snake = player_snake
        if player_snake:
            self.snakes.append(player_snake)
        self.width, self.height = width, height
        self.log = log
        self.max_number_of_fruits = max_number_of_fruits
        self.max_number_of_snakes = max_number_of_snakes
        self.rewards = [0 for s in snakes]

    def __iter__(self):
        game_over = False

        while True:
            direction = yield
            if self.player_snake:
                self.player_snake.update(direction)
            self.check_collisions()
            if not self.snakes:
                game_over = True
            for snake in self.snakes:
                if snake is not self.player_snake:
                    snake.update(None)
            self.update_fruits()

            if game_over:
                break

    def update_fruits(self):
        """Add fruits to the game until max_number_of_fruits is reached."""
        while len(self.fruits) < self.max_number_of_fruits:
            new_x, new_y = (
                random.randint(1, self.width - 1),
                random.randint(1, self.height - 1),
            )
            self.fruits.append((new_x, new_y))

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
                    self.rewards[s_idx] = FRUIT_REWARD
            # Check snake collisions
            for s2_idx, s2 in enumerate(self.snakes):
                if s_idx != s2_idx:
                    for x2s, y2s in s2.coordinates:
                        if (x_s, y_s) == (x2s, y2s):
                            snakes_to_be_deleted.append(s)
                            self.rewards[s_idx] = DEATH_REWARD
                else:
                    for x2s, y2s in list(s2.coordinates)[:-1]:
                        if (x_s, y_s) == (x2s, y2s):
                            snakes_to_be_deleted.append(s)
                            self.rewards[s_idx] = DEATH_REWARD

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
        if (new_direction, self.direction) in [
            ("N", "S"),
            ("S", "N"),
            ("O", "W"),
            ("W", "O"),
        ]:
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
            # screen.addstr(0, 0, "Epoch {}, Step {}, Reward {}".format(epoch, gs, rewards[-1:]))
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
            curses.napms(10)
            if game_over:
                break
        # Give small penalty if no fruits at all have been
        # collected
        # Give reward for length of snake
        if all(x == 0 for x in rewards):
            rewards[-1] += -20
        rewards[-1] += snake.length ** 2
        R = 0
        discounted_reward = np.zeros_like(rewards, dtype=float)
        for t in reversed(range(discounted_reward.shape[0])):
            R = gamma * R + rewards[t]
            discounted_reward[t] = R
        discounted_reward -= discounted_reward.mean()
        discounted_reward /= np.std(discounted_reward)

        net.backprop_value(actions, discounted_reward, gamma=0.01)

    net.save_state()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "neuro":
        curses.wrapper(nn_training)
    else:
        curses.wrapper(main)
    print("Vorbei!")
