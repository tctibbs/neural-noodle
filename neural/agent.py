import random
from collections import deque  # data structure to store memory

import matplotlib.pyplot as plt
import torch  # pytorch
from IPython import display

from noodle.game import GameState, SnakeGame
from noodle.model import Action

from .model import Linear_QNet
from .state import get_state, get_state_size
from .trainer import QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.01  # learning rate


class Agent:
    def __init__(self, snake_game: SnakeGame) -> None:
        self.snake_game = snake_game

        self.n_games = 0
        self.total_games = 500
        self.epsilon = 0.0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(get_state_size(snake_game), 256, len(Action))
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.distance_to_fruit = 1000

    def compute_reward(self, state, game: SnakeGame) -> int:
        """Returns the reward for the current state."""
        snake_head = game.snake.head()
        fruit_position = game.fruit.position()

        if game.state == GameState.IDLE:
            return -1

        if game.snake.last_ate() == 0:
            return 1

        # return -1
        return -0.1

        distance_to_fruit = (
            (snake_head.x - fruit_position.x) ** 2
            + (snake_head.y - fruit_position.y) ** 2
        ) ** 0.5
        if self.distance_to_fruit <= distance_to_fruit:
            reward = 10
        else:
            reward = -10

        self.distance_to_fruit = distance_to_fruit
        return reward

        # Calculate Euclidean distance between snake head and fruit position
        distance = (
            (snake_head.x - fruit_position.x) ** 2
            + (snake_head.y - fruit_position.y) ** 2
        ) ** 0.5

        # Set reward based on the distance
        reward = (
            -distance
        )  # Negative distance to encourage getting closer to the fruit

        return reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory, BATCH_SIZE
            )  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        if (
            random.randint(0, 100)
            < (self.total_games * self.epsilon) - self.n_games
        ):
            # Explore
            return random.choice(list(Action)).value

        # Exploit
        current_state = torch.tensor(state, dtype=torch.float)
        prediction = self.model(current_state)
        action = torch.argmax(prediction).item()
        return action


plt.ion()


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(0.1)


def train(WIDTH, HEIGHT, CELL_SIZE, FPS):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    snake_game = SnakeGame(WIDTH, HEIGHT, CELL_SIZE, FPS)
    agent = Agent(snake_game)
    while True:
        print("Current State")
        state_current = get_state(snake_game)
        action = agent.get_action(state_current)
        game_state, score = snake_game.play_step(Action(action))
        print("New State")
        state_new = get_state(snake_game)
        reward = agent.compute_reward(state_new, snake_game)

        # print(f"Action: {action}, Reward: {reward}")
        agent.train_short_memory(
            state_current, action, reward, state_new, game_state
        )
        agent.remember(state_current, action, reward, state_new, game_state)

        if snake_game.turns_since_eat == 100:
            snake_game.reset()

        if game_state == GameState.IDLE:
            agent.n_games += 1
            # agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()

            # print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            if agent.n_games == agent.total_games:
                input("Press Enter to continue...")
                break
