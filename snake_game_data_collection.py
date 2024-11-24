import pygame
import random
import sys
import numpy as np
import pandas as pd

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
UP = 'UP'
DOWN = 'DOWN'
LEFT = 'LEFT'
RIGHT = 'RIGHT'
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]
DIRECTION_VECTORS = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0)
}

# Direction Labels
DIR_LABELS = {
    UP: 0,
    DOWN: 1,
    LEFT: 2,
    RIGHT: 3
}

class SnakeGameDataCollector:
    def __init__(self, width=600, height=400):
        # Game Dimensions
        self.WINDOW_WIDTH = width
        self.WINDOW_HEIGHT = height

        # Grid Dimensions
        self.GRID_WIDTH = self.WINDOW_WIDTH // CELL_SIZE
        self.GRID_HEIGHT = self.WINDOW_HEIGHT // CELL_SIZE

        # Set up the display
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption('Snake Game Data Collection')

        # Set up the clock
        self.clock = pygame.time.Clock()

        # Font for score display
        self.font = pygame.font.SysFont('Arial', 24)

        # Game Variables
        self.reset()

        # Data collection variables
        self.data = []

    def reset(self):
        self.snake = [(self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)]
        self.direction = random.choice(DIRECTIONS)
        self.score = 0
        self.speed = 10  # Game speed in frames per second
        self.max_steps = 1000  # Maximum steps per game to prevent infinite loops
        self.steps = 0
        self.pellets = []
        self.spawn_pellets(2)  # Initially spawn two pellets
        self.board = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)

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

    def draw_cell(self, position, color):
        rect = pygame.Rect(
            position[0] * CELL_SIZE,
            position[1] * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE
        )
        pygame.draw.rect(self.screen, color, rect)

    def draw_grid(self):
        for x in range(0, self.WINDOW_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.WINDOW_HEIGHT))
        for y in range(0, self.WINDOW_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.WINDOW_WIDTH, y))

    def update_board(self):
        self.board.fill(0)
        for segment in self.snake[1:]:
            self.board[segment[1], segment[0]] = 2  # Snake body
        head = self.snake[0]
        self.board[head[1], head[0]] = 1  # Snake head
        for pellet in self.pellets:
            self.board[pellet[1], pellet[0]] = 3  # Pellets

    def handle_events(self):
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.save_data()
                pygame.quit()
                sys.exit()

    def move_snake(self):
        dx, dy = DIRECTION_VECTORS[self.direction]
        new_head = (
            (self.snake[0][0] + dx) % self.GRID_WIDTH,
            (self.snake[0][1] + dy) % self.GRID_HEIGHT
        )

        # Collision Detection with self
        if new_head in self.snake:
            return False  # Game Over

        self.snake.insert(0, new_head)

        # Check for pellet collision
        if new_head in self.pellets:
            self.score += 1
            self.pellets.remove(new_head)
            self.spawn_pellets(2)  # Ensure there are always two pellets
        else:
            self.snake.pop()  # Remove last segment

        return True  # Continue game

    def draw_elements(self):
        self.screen.fill(BLACK)
        self.draw_grid()
        # Draw Snake
        for segment in self.snake:
            self.draw_cell(segment, GREEN)
        # Draw Pellets
        for pellet in self.pellets:
            self.draw_cell(pellet, RED)
        # Draw Score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (5, 5))
        pygame.display.flip()

    def collect_data(self):
        self.update_board()
        state = self.board.copy()
        action = DIR_LABELS[self.direction]
        self.data.append({'state': state, 'action': action})

    def save_data(self):
        # Flatten the state arrays and prepare the data for saving
        records = []
        for entry in self.data:
            flat_state = entry['state'].flatten()
            record = flat_state.tolist()
            record.append(entry['action'])
            records.append(record)

        # Create a DataFrame
        columns = [f'cell_{i}' for i in range(self.GRID_WIDTH * self.GRID_HEIGHT)] + ['action']
        df = pd.DataFrame(records, columns=columns)

        # Save to CSV
        df.to_csv('snake_game_data.csv', index=False)
        print("Data saved to snake_game_data.csv")

    def run(self):
        while self.steps < self.max_steps:
            self.clock.tick(self.speed)
            self.handle_events()
            # Autonomous movement: Random direction
            self.direction = random.choice(DIRECTIONS)

            if not self.move_snake():
                print("Game Over!")
                break

            self.collect_data()
            self.draw_elements()
            self.steps += 1

        self.save_data()
        pygame.quit()
        sys.exit()

def main():
    # You can set the game board size here (width, height)
    game = SnakeGameDataCollector(width=800, height=600)
    game.run()

if __name__ == "__main__":
    main()
