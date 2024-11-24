import pygame
import random
import sys
import numpy as np
import pandas as pd

# Initialize Pygame
pygame.init()

# Game Constants
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400
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
    def __init__(self):
        # Set up the display
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
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
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice(DIRECTIONS)
        self.score = 0
        self.speed = 10  # Game speed in frames per second
        self.max_steps = 1000  # Maximum steps per game to prevent infinite loops
        self.steps = 0
        self.spawn_pellet()
        self.board = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)

    def spawn_pellet(self):
        while True:
            self.pellet = (
                random.randint(0, GRID_WIDTH - 1),
                random.randint(0, GRID_HEIGHT - 1)
            )
            if self.pellet not in self.snake:
                break

    def draw_cell(self, position, color):
        rect = pygame.Rect(
            position[0] * CELL_SIZE,
            position[1] * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE
        )
        pygame.draw.rect(self.screen, color, rect)

    def draw_grid(self):
        for x in range(0, WINDOW_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (WINDOW_WIDTH, y))

    def update_board(self):
        self.board.fill(0)
        for segment in self.snake[1:]:
            self.board[segment[1], segment[0]] = 2  # Snake body
        head = self.snake[0]
        self.board[head[1], head[0]] = 1  # Snake head
        self.board[self.pellet[1], self.pellet[0]] = 3  # Pellet

    def handle_events(self):
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.save_data()
                pygame.quit()
                sys.exit()

    def move_snake(self):
        dx, dy = DIRECTION_VECTORS[self.direction]
        new_head = (self.snake[0][0] + dx, self.snake[0][1] + dy)

        # Check for collision with walls
        if not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
            return False  # Game Over

        # Collision Detection with self
        if new_head in self.snake:
            return False  # Game Over

        self.snake.insert(0, new_head)

        # Check for pellet collision
        if new_head == self.pellet:
            self.score += 1
            self.spawn_pellet()
        else:
            self.snake.pop()  # Remove last segment

        return True  # Continue game

    def draw_elements(self):
        self.screen.fill(BLACK)
        self.draw_grid()
        # Draw Snake
        for segment in self.snake:
            self.draw_cell(segment, GREEN)
        # Draw Pellet
        self.draw_cell(self.pellet, RED)
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
        columns = [f'cell_{i}' for i in range(GRID_WIDTH * GRID_HEIGHT)] + ['action']
        df = pd.DataFrame(records, columns=columns)

        # Save to CSV
        df.to_csv('snake_game_data.csv', index=False)
        print("Data saved to snake_game_data.csv")

    def run(self):
        while self.steps < self.max_steps:
            self.clock.tick(self.speed)
            self.handle_events()
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
    game = SnakeGameDataCollector()
    game.run()

if __name__ == "__main__":
    main()
