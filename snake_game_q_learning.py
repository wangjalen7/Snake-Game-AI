import pygame
import random
import numpy as np
import pickle

# Initialize Pygame
pygame.init()

CELL_SIZE = 20

WHITE = (255, 255, 255)
GRAY = (40, 40, 40)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREEN = (0, 180, 0)

# Directions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]
DIRECTION_VECTORS = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0)
}


class SnakeGame:
    def __init__(self, width=200, height=200, render=False):
        self.WINDOW_WIDTH = width
        self.WINDOW_HEIGHT = height
        self.GRID_WIDTH = self.WINDOW_WIDTH // CELL_SIZE
        self.GRID_HEIGHT = self.WINDOW_HEIGHT // CELL_SIZE

        self.render_game = render
        if self.render_game:
            self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            pygame.display.set_caption('Snake Q-Learning Agent')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 24)
        self.reset()

    def reset(self):
        self.snake = [(self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.score = 0
        self.steps_since_last_pellet = 0
        self.total_steps = 0
        self.pellet = self.spawn_pellet()
        return self.get_state()

    def spawn_pellet(self):
        while True:
            pellet = (
                random.randint(0, self.GRID_WIDTH - 1),
                random.randint(0, self.GRID_HEIGHT - 1)
            )
            if pellet not in self.snake:
                return pellet

    def is_collision(self, position):
        x, y = position
        if (x, y) in self.snake[1:]:
            return True
        return False

    def get_state(self):
        head = self.snake[0]
        direction = self.direction

        danger_straight = self.is_collision(((head[0] + DIRECTION_VECTORS[direction][0]) % self.GRID_WIDTH,
                                             (head[1] + DIRECTION_VECTORS[direction][1]) % self.GRID_HEIGHT))

        danger_right = self.is_collision(((head[0] + DIRECTION_VECTORS[(direction + 1) % 4][0]) % self.GRID_WIDTH,
                                          (head[1] + DIRECTION_VECTORS[(direction + 1) % 4][1]) % self.GRID_HEIGHT))

        danger_left = self.is_collision(((head[0] + DIRECTION_VECTORS[(direction - 1) % 4][0]) % self.GRID_WIDTH,
                                         (head[1] + DIRECTION_VECTORS[(direction - 1) % 4][1]) % self.GRID_HEIGHT))

        food_direction = (
            int(self.pellet[0] < head[0]),
            int(self.pellet[0] > head[0]),
            int(self.pellet[1] < head[1]),
            int(self.pellet[1] > head[1])
        )

        return (danger_straight, danger_right, danger_left, direction) + food_direction

    def step(self, action):
        if action == 0:  # Straight
            pass
        elif action == 1:  # Right turn
            self.direction = (self.direction + 1) % 4
        elif action == 2:  # Left turn
            self.direction = (self.direction - 1) % 4

        dx, dy = DIRECTION_VECTORS[self.direction]
        new_head = (
            (self.snake[0][0] + dx) % self.GRID_WIDTH,
            (self.snake[0][1] + dy) % self.GRID_HEIGHT
        )

        reward = 0
        done = False

        if new_head in self.snake:
            reward = -10
            done = True
            return self.get_state(), reward, done, self.score

        self.snake.insert(0, new_head)
        if new_head == self.pellet:
            self.score += 1
            reward = 100
            self.pellet = self.spawn_pellet()
        else:
            self.snake.pop()

        return self.get_state(), reward, done, self.score

    def render(self):
        if not self.render_game:
            return
        self.screen.fill(BLACK)
        for segment in self.snake:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, GREEN, rect)

        pellet_rect = pygame.Rect(self.pellet[0] * CELL_SIZE, self.pellet[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, RED, pellet_rect)

        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (5, 5))

        pygame.display.flip()
        self.clock.tick(20)


class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.9, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def set_q_value(self, state, action, value):
        self.q_table[(state, action)] = value

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        q_values = [self.get_q_value(state, action) for action in range(self.action_size)]
        return np.argmax(q_values)

    def learn(self, state, action, reward, next_state, done):
        q_current = self.get_q_value(state, action)
        max_q_next = max([self.get_q_value(next_state, a) for a in range(self.action_size)]) if not done else 0
        q_target = reward + self.discount_factor * max_q_next
        self.set_q_value(state, action, q_current + self.learning_rate * (q_target - q_current))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def save_q_table(agent, filename="q_table.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(agent.q_table, f)


def load_q_table(filename="q_table.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)


def play_game(agent, game):
    game.render_game = True
    state = game.reset()
    done = False
    while not done:
        for event in pygame.event.get():  # Check for any Pygame events (like window close)
            if event.type == pygame.QUIT:
                pygame.quit()
                return  # Exit the game if the window is closed

        action = agent.choose_action(state)
        state, _, done, _ = game.step(action)
        game.render()
    print(f"Game over! Final score: {game.score}")


def train_agent(game, agent, episodes):
    total_reward = 0
    for e in range(episodes):
        state = game.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = game.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

    return total_reward / episodes  # Return average score


def stochastic_grid_search():
    # Define ranges for hyperparameters
    learning_rate_options = [0.01, 0.05, 0.1, 0.5]
    discount_factor_options = [0.7, 0.8, 0.9, 0.95]
    epsilon_decay_options = [0.9, 0.95, 0.99]
    epsilon_min_options = [0.01, 0.05, 0.1]

    best_score = float('-inf')
    best_params = None

    for _ in range(10):  # Stochastic search over 10 random combinations
        learning_rate = random.choice(learning_rate_options)
        discount_factor = random.choice(discount_factor_options)
        epsilon_decay = random.choice(epsilon_decay_options)
        epsilon_min = random.choice(epsilon_min_options)

        # Create the agent with random parameters
        agent = QLearningAgent(state_size=6, action_size=3, learning_rate=learning_rate,
                               discount_factor=discount_factor, epsilon_decay=epsilon_decay,
                               epsilon_min=epsilon_min)

        # Train the agent
        game = SnakeGame(width=200, height=200, render=False)
        avg_score = train_agent(game, agent, episodes=1000)

        print(f"Tested Params: lr={learning_rate}, gamma={discount_factor}, epsilon_decay={epsilon_decay}, "
              f"epsilon_min={epsilon_min} | Avg. Score: {avg_score}")

        if avg_score > best_score:
            best_score = avg_score
            best_params = (learning_rate, discount_factor, epsilon_decay, epsilon_min)

    print(f"\nBest Hyperparameters: {best_params} with Avg. Score: {best_score}")
    return best_params


def main():
    best_params = stochastic_grid_search()

    # Create the agent with the best hyperparameters
    agent = QLearningAgent(state_size=6, action_size=3, learning_rate=best_params[0],
                           discount_factor=best_params[1], epsilon_decay=best_params[2],
                           epsilon_min=best_params[3])

    # Train the agent using the best hyperparameters
    game = SnakeGame(width=200, height=200, render=True)
    train_agent(game, agent, episodes=3000)

    # Save the trained Q-table
    save_q_table(agent)
    print("Training complete.")

    # Load Q-table and play the game
    print("Playing the game using the learned policy...")
    trained_q_table = load_q_table()
    agent.q_table = trained_q_table
    game.render_game = True
    play_game(agent, game)

    # Quit Pygame after game has finished
    pygame.quit()
    print("Game over. Pygame quit.")


if __name__ == "__main__":
    main()
