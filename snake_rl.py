# snake_rl.py
import os
import pygame
import random
import numpy as np
from collections import deque
import tensorflow as tf
from keras import layers, models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

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

# Actions
STRAIGHT = 0
RIGHT_TURN = 1
LEFT_TURN = 2

class SnakeGameRL:
    def __init__(self, width=200, height=200, render=False):
        self.WINDOW_WIDTH = width
        self.WINDOW_HEIGHT = height
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
        self.steps_since_last_pellet = 0
        self.total_steps = 0
        self.pellets = []
        self.spawn_pellets(2)
        return self.get_state()

    def spawn_pellets(self, num_pellets=2):
        while len(self.pellets) < num_pellets:
            pellet = (
                random.randint(0, self.GRID_WIDTH - 1),
                random.randint(0, self.GRID_HEIGHT - 1)
            )
            if pellet not in self.snake and pellet not in self.pellets:
                self.pellets.append(pellet)

    def manhattan_distance(self, a, b):
        dx = min(abs(a[0] - b[0]), self.GRID_WIDTH - abs(a[0] - b[0]))
        dy = min(abs(a[1] - b[1]), self.GRID_HEIGHT - abs(a[1] - b[1]))
        return dx + dy

    def bfs_shortest_path_to_pellet(self):
        if not self.pellets:
            return 0
        start = self.snake[0]
        goals = set(self.pellets)
        obstacles = set(self.snake[1:])

        queue = deque()
        queue.append((start, 0))
        visited = set([start])
        while queue:
            current, dist = queue.popleft()
            if current in goals:
                return dist
            for d in DIRECTIONS:
                dx, dy = DIRECTION_VECTORS[d]
                nx = (current[0] + dx) % self.GRID_WIDTH
                ny = (current[1] + dy) % self.GRID_HEIGHT
                next_pos = (nx, ny)
                if next_pos not in visited and next_pos not in obstacles:
                    visited.add(next_pos)
                    queue.append((next_pos, dist+1))
        return max(self.GRID_WIDTH, self.GRID_HEIGHT)  # Large penalty if no path

    def is_collision(self, position, direction):
        x, y = position
        dx, dy = direction
        new_x = (x + dx) % self.GRID_WIDTH
        new_y = (y + dy) % self.GRID_HEIGHT
        if (new_x, new_y) in self.snake[1:]:
            return True
        return False

    def get_state(self):
        head = self.snake[0]
        dir_vector = DIRECTION_VECTORS[self.direction]
        left_dir = DIRECTION_VECTORS[(self.direction - 1) % 4]
        right_dir = DIRECTION_VECTORS[(self.direction + 1) % 4]

        closest_pellet = min(self.pellets, key=lambda p: self.manhattan_distance(head, p))

        state = [
            int(self.is_collision(head, dir_vector)),   # Danger straight
            int(self.is_collision(head, right_dir)),    # Danger right
            int(self.is_collision(head, left_dir)),     # Danger left
            int(self.direction == UP),
            int(self.direction == DOWN),
            int(self.direction == LEFT),
            int(self.direction == RIGHT),
            int(closest_pellet[0] < head[0]),  # Food left
            int(closest_pellet[0] > head[0]),  # Food right
            int(closest_pellet[1] < head[1]),  # Food up
            int(closest_pellet[1] > head[1])   # Food down
        ]
        return np.array(state, dtype=int)

    def step(self, action):
        self.total_steps += 1
        old_distance = self.bfs_shortest_path_to_pellet()

        if action == STRAIGHT:
            pass
        elif action == RIGHT_TURN:
            self.direction = (self.direction + 1) % 4
        elif action == LEFT_TURN:
            self.direction = (self.direction - 1) % 4

        dx, dy = DIRECTION_VECTORS[self.direction]
        new_head = (
            (self.snake[0][0] + dx) % self.GRID_WIDTH,
            (self.snake[0][1] + dy) % self.GRID_HEIGHT
        )

        done = False
        reward = 0

        if new_head in self.snake[1:]:
            reward = -10
            done = True
            return self.get_state(), reward, done, self.score

        self.snake.insert(0, new_head)
        if new_head in self.pellets:
            self.score += 1
            reward += 10
            self.pellets.remove(new_head)
            self.spawn_pellets(2)
            self.steps_since_last_pellet = 0
        else:
            self.snake.pop()
            reward -= 0.01

        self.steps_since_last_pellet += 1
        # End episode if too long without eating
        if self.steps_since_last_pellet > (self.GRID_WIDTH * self.GRID_HEIGHT):
            reward -= 5
            done = True

        new_distance = self.bfs_shortest_path_to_pellet()
        if new_distance < old_distance:
            reward += 0.1
        elif new_distance > old_distance:
            reward -= 0.05

        reward += 0.001  # Small survival reward

        return self.get_state(), reward, done, self.score

    def render(self):
        if not self.render_game:
            return
        self.screen.fill(BLACK)
        for segment in self.snake:
            rect = pygame.Rect(segment[0]*CELL_SIZE, segment[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, GREEN, rect)

        for pellet in self.pellets:
            pellet_rect = pygame.Rect(pellet[0]*CELL_SIZE, pellet[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, RED, pellet_rect)

        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (5,5))

        pygame.display.flip()
        self.clock.tick(20)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.epsilon = 1e-5

    def add(self, transition, td_error):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.priorities[self.position] = max(max_priority, td_error + self.epsilon)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        probs = (priorities + 1e-6) ** self.alpha
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)  # Fix invalid values
        probs = np.clip(probs, 0, None)  # Ensure non-negative values
        if np.sum(probs) == 0:
            probs = np.ones(len(self.buffer)) / len(self.buffer)  # Fallback to uniform distribution
        else:
            probs /= np.sum(probs)  # Normalize probabilities

        # Now proceed with sampling
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = td_error + self.epsilon


def build_dueling_dqn(state_size, action_size, learning_rate):
    inputs = layers.Input(shape=(state_size,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)

    value = layers.Dense(1, activation=None)(x)
    advantage = layers.Dense(action_size, activation=None)(x)

    # Calculate average advantage manually without a Lambda layer
    average_advantage = tf.reduce_mean(advantage, axis=1, keepdims=True)
    advantage_minus_avg = advantage - average_advantage

    # Calculate Q-values
    q_values = value + advantage_minus_avg

    model = models.Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model


class DQNAgent:
    def __init__(self, state_size=11, action_size=3, capacity=25000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(capacity)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.batch_size = 16
        self.train_start = 1000
        self.model = build_dueling_dqn(self.state_size, self.action_size, self.learning_rate)
        self.target_model = build_dueling_dqn(self.state_size, self.action_size, self.learning_rate)
        self.update_target_model()
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        self.steps = 0

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        td_error = abs(reward)
        self.memory.add((state, action, reward, next_state, done), td_error)
        self.steps += 1

    def replay(self):
        if self.steps < self.train_start or len(self.memory.buffer) < self.batch_size:
            return

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        samples, indices, weights = self.memory.sample(self.batch_size, self.beta)
        states = np.array([sample[0] for sample in samples])
        actions = np.array([sample[1] for sample in samples])
        rewards = np.array([sample[2] for sample in samples])
        next_states = np.array([sample[3] for sample in samples])
        dones = np.array([sample[4] for sample in samples])

        states = states.reshape((self.batch_size, self.state_size))  # Reshape states to (batch_size, state_size)
        next_states = next_states.reshape((self.batch_size, self.state_size))  # Same for next_states

        target = self.model.predict(states, verbose=0)
        target_next = self.model.predict(next_states, verbose=0)
        target_val = self.target_model.predict(next_states, verbose=0)

        td_errors = np.zeros((self.batch_size,), dtype=np.float32)

        for i in range(len(samples)):
            if dones[i]:
                td_target = rewards[i]
            else:
                best_action = np.argmax(target_next[i])
                td_target = rewards[i] + self.gamma * target_val[i][best_action]
            td_errors[i] = td_target - target[i][actions[i]]
            target[i][actions[i]] = td_target

        self.model.fit(states, target, sample_weight=weights, verbose=0)
        self.memory.update_priorities(indices, td_errors)


def evaluate_agent(agent, env, eval_episodes=10):
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    total_score = 0
    for i in range(eval_episodes):
        state = env.reset()
        done = False
        episode_score = 0
        step_count = 0
        # Safety break in evaluation too
        max_eval_steps = 500
        while not done:
            step_count += 1
            if step_count > max_eval_steps:
                print("Evaluation episode forced to end due to step limit.")
                done = True
                break

            action = agent.act(state)
            state, reward, done, score = env.step(action)
            episode_score = score
        total_score += episode_score
    agent.epsilon = old_epsilon
    return total_score / eval_episodes


def main():
    print("Starting training...")
    game = SnakeGameRL(width=200, height=200, render=False)
    agent = DQNAgent()
    episodes = 1500
    eval_interval = 100
    eval_episodes = 10
    best_eval_score = -float('inf')

    max_episode_steps = 1000

    for e in range(episodes):
        if e % 100 == 0:
            print(f"Starting Episode {e+1}/{episodes}")
        state = game.reset()
        done = False
        episode_score = 0
        episode_step_count = 0

        while not done:
            episode_step_count += 1
            if episode_step_count % 100 == 0:
                print(f"Episode {e+1}, Step {episode_step_count}...")

            if episode_step_count > max_episode_steps:
                print("Forcing episode to end due to max step limit.")
                done = True
                break

            action = agent.act(state)
            next_state, reward, done, score = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            episode_score = score

            if agent.steps % 1000 == 0:
                print("Updating target model...")
                agent.update_target_model()

        # Periodically evaluate
        if (e + 1) % eval_interval == 0:
            avg_eval_score = evaluate_agent(agent, game, eval_episodes)
            print(f"Episode {e+1}, Evaluation Average Score: {avg_eval_score:.2f}")
            if avg_eval_score > best_eval_score:
                best_eval_score = avg_eval_score
                agent.model.save(f'{episodes}_snake_model.h5', overwrite=True)
                print(f"New best model saved with Avg Eval Score: {avg_eval_score:.2f}")

        if (e + 1) % 100 == 0:
            print(f"Episode: {e+1}/{episodes}, Current Score: {episode_score}, Epsilon: {agent.epsilon:.4f}, Best Eval Score: {best_eval_score:.2f}")

    pygame.quit()
    print(f"Training complete. Best evaluation score achieved: {best_eval_score:.2f}")


if __name__ == "__main__":
    main()
