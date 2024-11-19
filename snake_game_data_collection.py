# snake_game_data_collection.py

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
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

# Direction Labels
DIR_LABELS = {
    UP: 0,
    DOWN: 1,
    LEFT: 2,
    RIGHT: 3
}

# Set up the display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Snake Game Data Collection')

# Set up the clock
clock = pygame.time.Clock()

# Font for score display
font = pygame.font.SysFont('Arial', 24)

def draw_cell(position, color):
    rect = pygame.Rect(position[0] * CELL_SIZE, position[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, color, rect)

def draw_grid():
    for x in range(0, WINDOW_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (WINDOW_WIDTH, y))

def initialize_board():
    board = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
    return board

def update_board(board, snake, pellet):
    board.fill(0)
    for segment in snake[1:]:
        board[segment[1], segment[0]] = 2  # Snake body
    head = snake[0]
    board[head[1], head[0]] = 1  # Snake head
    board[pellet[1], pellet[0]] = 3  # Pellet
    # Borders can be added if needed
    return board

def main():
    # Game Variables
    snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
    direction = random.choice(DIRECTIONS)
    pellet = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
    score = 0
    speed = 10  # Game speed in frames per second
    max_steps = 1000  # Maximum steps per game to prevent infinite loops

    # Initialize the board
    board = initialize_board()

    # Data collection variables
    data = []

    running = True
    steps = 0
    while running and steps < max_steps:
        clock.tick(speed)
        screen.fill(BLACK)
        draw_grid()

        # Autonomous movement: Random direction
        direction = random.choice(DIRECTIONS)

        # Move Snake
        new_head = ((snake[0][0] + direction[0]) % GRID_WIDTH,
                    (snake[0][1] + direction[1]) % GRID_HEIGHT)

        # Collision Detection
        if new_head in snake:
            # Game Over
            running = False
            continue

        snake.insert(0, new_head)

        # Check for pellet collision
        if new_head == pellet:
            score += 1
            # Place new pellet
            while True:
                pellet = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
                if pellet not in snake:
                    break
        else:
            snake.pop()  # Remove last segment

        # Update the board
        board = update_board(board, snake, pellet)

        # Collect data
        state = board.copy()
        action = DIR_LABELS[direction]
        data.append({'state': state, 'action': action})

        # Draw Snake
        for segment in snake:
            draw_cell(segment, GREEN)

        # Draw Pellet
        draw_cell(pellet, RED)

        # Draw Score
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (5, 5))

        pygame.display.flip()

        steps += 1

    # Save data to CSV
    save_data_to_csv(data)

    pygame.quit()
    sys.exit()

def save_data_to_csv(data):
    # Flatten the state arrays and prepare the data for saving
    records = []
    for entry in data:
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

if __name__ == "__main__":
    main()
