import pygame
import random
import sys
import numpy as np
import tensorflow as tf
from collections import deque

# Initialize Pygame
pygame.init()

# Game Constants
CELL_SIZE = 20  # Cell size remains constant

# Colors
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
DIRECTION_VECTORS = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0)
}

# Action Constants
STRAIGHT = 0
RIGHT_TURN = 1
LEFT_TURN = 2

class SnakeGameRL:
    def __init__(self, width=200, height=200, render=False):
        self.WINDOW_WIDTH = width
        self.WINDOW_HEIGHT = height

        # Grid Dimensions
        self.GRID_WIDTH = self.WINDOW_WIDTH // CELL_SIZE
        self.GRID_HEIGHT = self.WINDOW_HEIGHT // CELL_SIZE

        self.render_game = render
        if self.render_game:
            self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            pygame.display.set_caption('Snake RL Agent')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 24)
        self.reset()

    def reset(self):
        self.snake = [(self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.score = 0
        self.frame = 0
        self.steps_since_last_pellet = 0
        self.pellets = []
        self.spawn_pellets(2)  # Initially spawn two pellets
        return self.get_state()

    def spawn_pellets(self, num_pellets=2):
        """
        Ensures that there are always 'num_pellets' pellets on the board.
        Adds new pellets without removing existing ones.
        """
        while len(self.pellets) < num_pellets:
            pellet = (
                random.randint(0, self.GRID_WIDTH - 1),
                random.randint(0, self.GRID_HEIGHT - 1)
            )
            if pellet not in self.snake and pellet not in self.pellets:
                self.pellets.append(pellet)

    def get_state(self):
        head = self.snake[0]
        dir_vector = DIRECTION_VECTORS[self.direction]
        left_dir = DIRECTION_VECTORS[(self.direction - 1) % 4]
        right_dir = DIRECTION_VECTORS[(self.direction + 1) % 4]

        # Find the closest pellet
        closest_pellet = min(self.pellets, key=lambda p: self.manhattan_distance(head, p))

        state = [
            # Danger straight
            self.is_collision(head, dir_vector),
            # Danger right
            self.is_collision(head, right_dir),
            # Danger left
            self.is_collision(head, left_dir),
            # Move direction
            self.direction == UP,
            self.direction == DOWN,
            self.direction == LEFT,
            self.direction == RIGHT,
            # Food location relative to head
            closest_pellet[0] < head[0],  # Food left
            closest_pellet[0] > head[0],  # Food right
            closest_pellet[1] < head[1],  # Food up
            closest_pellet[1] > head[1]   # Food down
        ]
        return np.array(state, dtype=int)

    def is_collision(self, position, direction):
        x, y = position
        dx, dy = direction
        new_x = (x + dx) % self.GRID_WIDTH
        new_y = (y + dy) % self.GRID_HEIGHT
        if (new_x, new_y) in self.snake[1:]:
            return True
        return False

    def manhattan_distance(self, a, b):
        dx = min(abs(a[0] - b[0]), self.GRID_WIDTH - abs(a[0] - b[0]))
        dy = min(abs(a[1] - b[1]), self.GRID_HEIGHT - abs(a[1] - b[1]))
        return dx + dy

    def step(self, action):
        self.frame += 1
        self.steps_since_last_pellet += 1
        # Update the direction based on action
        if action == STRAIGHT:
            pass  # Keep current direction
        elif action == RIGHT_TURN:
            self.direction = (self.direction + 1) % 4
        elif action == LEFT_TURN:
            self.direction = (self.direction - 1) % 4

        dx, dy = DIRECTION_VECTORS[self.direction]
        new_head = (
            (self.snake[0][0] + dx) % self.GRID_WIDTH,
            (self.snake[0][1] + dy) % self.GRID_HEIGHT
        )

        # Check for collision
        reward = 0
        done = False
        if self.is_collision(self.snake[0], DIRECTION_VECTORS[self.direction]):
            reward = -10
            done = True
            return self.get_state(), reward, done, self.score

        # Move snake
        self.snake.insert(0, new_head)
        if new_head in self.pellets:
            self.score += 1
            reward = 10
            self.pellets.remove(new_head)
            self.spawn_pellets(2)  # Ensure there are always two pellets
            self.steps_since_last_pellet = 0
        else:
            self.snake.pop()
            reward = 0  # No reward for just moving

        # Optional: Add a limit to the number of frames without eating
        if self.steps_since_last_pellet > 100:
            done = True
            reward = -10  # Penalty for taking too long

        return self.get_state(), reward, done, self.score

    def render(self):
        if not self.render_game:
            return
        self.screen.fill(BLACK)
        for segment in self.snake:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, GREEN, rect)

        # Draw pellets
        for pellet in self.pellets:
            pellet_rect = pygame.Rect(pellet[0] * CELL_SIZE, pellet[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, RED, pellet_rect)

        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (5, 5))

        pygame.display.flip()
        self.clock.tick(15)  # Limit to 15 FPS

class Agent:
    def __init__(self):
        self.state_size = 11
        self.action_size = 3  # [straight, right turn, left turn]
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.batch_size = 64
        self.train_start = 1000  # Start training after some experiences
        self.steps = 0  # Total steps taken

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.reshape(state, [1, self.state_size]))
        return np.argmax(act_values[0])  # Returns action with highest Q-value

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.steps += 1

    def replay(self):
        if len(self.memory) < self.train_start:
            return None
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        states = np.array([sample[0] for sample in minibatch])
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.array([sample[3] for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch])

        target = self.model.predict(states)
        target_next = self.target_model.predict(next_states)

        for i in range(len(minibatch)):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])

        history = self.model.fit(states, target, epochs=1, verbose=0)
        return history.history['loss'][0]

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main():
    game = SnakeGameRL(width=200, height=200, render=False)  # Set render=True to visualize
    agent = Agent()
    episodes = 1000
    scores = []
    survival_times = []
    losses = []
    epsilons = []
    for e in range(episodes):
        state = game.reset()
        done = False
        total_reward = 0
        total_loss = 0
        steps = 0

        while not done:
            game.render()
            action = agent.act(state)
            next_state, reward, done, score = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            state = next_state
            total_reward += reward
            steps += 1

            if loss is not None:
                total_loss += loss

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            if agent.steps % 1000 == 0:
                agent.update_target_model()

        agent.decay_epsilon()

        # Collect metrics
        scores.append(score)
        survival_times.append(steps)
        epsilons.append(agent.epsilon)
        if total_loss > 0:
            losses.append(total_loss / steps)
        else:
            losses.append(0)

        print(f"Episode {e+1}/{episodes}, Score: {score}, Steps: {steps}, Epsilon: {agent.epsilon:.4f}, Loss: {losses[-1]:.4f}")

    pygame.quit()

    # Plot the scores
    import matplotlib.pyplot as plt

    # Plot Scores
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Score over Episodes')

    # Plot Survival Times
    plt.subplot(1,2,2)
    plt.plot(survival_times)
    plt.xlabel('Episode')
    plt.ylabel('Survival Time (Steps)')
    plt.title('Survival Time over Episodes')
    plt.tight_layout()
    plt.show()

    # Plot Losses
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss over Episodes')
    plt.show()

    # Plot Epsilon
    plt.figure()
    plt.plot(epsilons)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon over Episodes')
    plt.show()

    # Save the model
    agent.model.save('snake_dqn_model.h5')
    print("Model saved as snake_dqn_model.h5")

if __name__ == "__main__":
    main()
