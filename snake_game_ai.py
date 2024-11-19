# snake_game_ai.py

import pygame
import random
import sys
import numpy as np
import heapq

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
BLUE = (0, 0, 255)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

# Set up the display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Snake Game with AI')

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

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def bfs(snake, pellet, obstacles):
    from collections import deque
    start = snake[0]
    queue = deque()
    queue.append((start, []))
    visited = set()
    visited.add(start)

    while queue:
        current_pos, path = queue.popleft()
        if current_pos == pellet:
            return path  # Found the path

        for direction in DIRECTIONS:
            new_x = (current_pos[0] + direction[0]) % GRID_WIDTH
            new_y = (current_pos[1] + direction[1]) % GRID_HEIGHT
            new_pos = (new_x, new_y)

            if new_pos in visited or new_pos in obstacles:
                continue

            visited.add(new_pos)
            queue.append((new_pos, path + [direction]))
    return None  # No path found

def a_star(snake, pellet, obstacles):
    start = snake[0]
    heap = []
    heapq.heappush(heap, (0, start, []))
    visited = set()
    visited.add(start)

    while heap:
        cost, current_pos, path = heapq.heappop(heap)
        if current_pos == pellet:
            return path  # Found the path

        for direction in DIRECTIONS:
            new_x = (current_pos[0] + direction[0]) % GRID_WIDTH
            new_y = (current_pos[1] + direction[1]) % GRID_HEIGHT
            new_pos = (new_x, new_y)

            if new_pos in visited or new_pos in obstacles:
                continue

            visited.add(new_pos)
            new_cost = len(path) + 1 + manhattan_distance(new_pos, pellet)
            heapq.heappush(heap, (new_cost, new_pos, path + [direction]))
    return None  # No path found

def main():
    # Game Variables
    snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
    pellet = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
    score = 0
    speed = 10  # Game speed in frames per second
    path = []

    running = True
    while running:
        clock.tick(speed)
        screen.fill(BLACK)
        draw_grid()

        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if not running:
            break

        # Get obstacles (snake body excluding the head)
        obstacles = set(snake[1:])

        # Check if path is empty or invalid
        if not path:
            # Choose algorithm: bfs or a_star
            # path = bfs(snake, pellet, obstacles)
            path = a_star(snake, pellet, obstacles)
            if path is None:
                # No path to pellet, move randomly
                path = [random.choice(DIRECTIONS)]

        # Get next move
        direction = path.pop(0)

        # Move Snake
        new_head = ((snake[0][0] + direction[0]) % GRID_WIDTH,
                    (snake[0][1] + direction[1]) % GRID_HEIGHT)

        # Collision Detection
        if new_head in obstacles:
            print("Game Over! The snake collided with itself.")
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

        # Draw Snake
        for segment in snake:
            draw_cell(segment, GREEN)

        # Draw Pellet
        draw_cell(pellet, RED)

        # Draw Score
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (5, 5))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
