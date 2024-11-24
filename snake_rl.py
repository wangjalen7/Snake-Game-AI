import pygame
import random
import sys
import numpy as np
import tensorflow as tf
from collections import deque

# Initialize Pygame
pygame.init()

# Game Constants
WINDOW_WIDTH = 200  # Reduced size for faster training
WINDOW_HEIGHT = 200
CELL_SIZE = 20

# Grid Dimensions
GRID_WIDTH = WINDOW_WIDTH // CELL_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // CELL_SIZE

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
    def __init__(self, render=False):
        self.render_game = render
        if self.render_game:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption('Snake RL Agent')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 24)
        self.reset()

    def reset(self):
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.place_pellet()
        self.score = 0
        self.frame = 0
        self.steps_since_last_pellet = 0
        return self.get_state()

    def place_pellet(self):
        while True:
            self.pellet = (
                random.randint(0, GRID_WIDTH - 1),
                random.randint(0, GRID_HEIGHT - 1)
            )
            if self.pellet not in self.snake:
                break

    def get_state(self):
        head = self.snake[0]
        dir_vector = DIRECTION_VECTORS[self.direction]
        left_dir = DIRECTION_VECTORS[(self.direction - 1) % 4]
        right_dir = DIRECTION_VECTORS[(self.direction + 1) % 4]

        state = [
            # Danger straight
            self.is_collision(head, dir_vector),
            # Danger right
            self.is_collision(head, right_dir),
            # Danger left
            self.is_collision(head, left_dir),
            # Move direction
            dir_vector == DIRECTION_VECTORS[UP],
            dir_vector == DIRECTION_VECTORS[DOWN],
            dir_vector == DIRECTION_VECTORS[LEFT],
            dir_vector == DIRECTION_VECTORS[RIGHT],
            # Food location
            self.pellet[0] < head[0],  # Food left
            self.pellet[0] > head[0],  # Food right
            self.pellet[1] < head[1],  # Food up
            self.pellet[1] > head[1]   # Food down
        ]
        return np.array(state, dtype=int)

    def is_collision(self, position, direction):
        x, y = position
        dx, dy = direction
        new_x = x + dx
        new_y = y + dy
        if not (0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT):
            return True
        if (new_x, new_y) in self.snake[1:]:
            return True
        return False

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
        new_head = (self.snake[0][0] + dx, self.snake[0][1] + dy)

        # Check for collision
        reward = 0
        done = False
        if self.is_collision(self.snake[0], DIRECTION_VECTORS[self.direction]):
            reward = -10
            done = True
            return self.get_state(), reward, done, self.score

        # Move snake
        self.snake.insert(0, new_head)
        if new_head == self.pellet:
            self.score += 1
            reward = 10
            self.place_pellet()
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

        # Draw pellet
        pellet_rect = pygame.Rect(self.pellet[0] * CELL_SIZE, self.pellet[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
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
    game = SnakeGameRL(render=True)  # Set render=True to visualize
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


    # Save the model
    agent.model.save('snake_dqn_model.h5')
    print("Model saved as snake_dqn_model.h5")

if __name__ == "__main__":
    main()
