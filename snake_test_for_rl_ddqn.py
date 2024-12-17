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
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = None  # The model will be loaded later for inference

    def load_model(self, model_path):
        """Load the trained model for inference."""
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def act(self, state):
        """Use the trained model to predict the best action."""
        if self.model is None:
            raise ValueError("Model has not been loaded. Use the 'load_model' method first.")
        state = np.reshape(state, [1, self.state_size])
        action_values = self.model.predict(state, verbose=0)  # Predict action values
        return np.argmax(action_values[0])  # Choose the action with the highest Q-value

def main():
    # Initialize the game and agent
    game = SnakeGameRL(width=200, height=200, render=True)  # Set render=True to visualize
    agent = Agent(state_size=11, action_size=6)

    # Load the trained model
    model_path = 'models/ddqn_snake_model.h5'  # Replace with the path to your .h5 file
    try:
        agent.load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Run the game with the loaded model
    total_episodes = 5  # Number of games to play
    for episode in range(total_episodes):
        state = game.reset()
        done = False
        total_score = 0

        print(f"Starting Episode {episode + 1}/{total_episodes}")
        while not done:
            game.render()
            action = agent.act(state)  # Get action from the model
            next_state, reward, done, score = game.step(action)
            state = next_state
            total_score = score

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        print(f"Episode {episode + 1}/{total_episodes} finished with Score: {total_score}")

    pygame.quit()
    print("Game over!")

if __name__ == "__main__":
    main()
