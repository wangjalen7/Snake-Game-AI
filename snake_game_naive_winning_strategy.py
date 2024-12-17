import pygame
import sys


# Note: we used chatGPT to debug this file as well as generate comments so that other team members could
# easily read through and know what was going on. As a result, some of the code is written by generative
# AI.


# Initialize Pygame
pygame.init()

# Game Constants
CELL_SIZE = 20  # Size of each cell in the grid

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
DIRECTION_VECTORS = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0)
}


class SnakeGameAI:
    def __init__(self, width=600, height=400):
        # Game Dimensions
        self.WINDOW_WIDTH = width
        self.WINDOW_HEIGHT = height

        # Grid Dimensions
        self.GRID_WIDTH = self.WINDOW_WIDTH // CELL_SIZE
        self.GRID_HEIGHT = self.WINDOW_HEIGHT // CELL_SIZE

        # Set up the display
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption('Snake Game with AI')

        # Set up the clock
        self.clock = pygame.time.Clock()

        # Font for score display
        self.font = pygame.font.SysFont('Arial', 24)

        # Game Variables
        self.reset()

    def reset(self):
        self.snake = [(self.GRID_WIDTH // 2, self.GRID_HEIGHT - 1)]
        self.score = 0
        self.speed = 10  # Game speed in frames per second
        self.pellets = []
        self.spawn_pellet()
        self.direction = UP
        self.up_moves = self.GRID_HEIGHT - 1
        self.right_moves = 0

    def spawn_pellet(self):
        """Randomly place a pellet on the grid."""
        import random
        while True:
            pellet = (
                random.randint(0, self.GRID_WIDTH - 1),
                random.randint(0, self.GRID_HEIGHT - 1)
            )
            if pellet not in self.snake:
                self.pellets = [pellet]
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
        for x in range(0, self.WINDOW_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.WINDOW_HEIGHT))
        for y in range(0, self.WINDOW_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.WINDOW_WIDTH, y))

    def draw_elements(self):
        self.screen.fill(BLACK)
        self.draw_grid()
        # Draw Snake
        for segment in self.snake:
            self.draw_cell(segment, GREEN)
        # Draw Pellet
        for pellet in self.pellets:
            self.draw_cell(pellet, RED)
        # Draw Score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (5, 5))
        pygame.display.flip()

    def handle_events(self):
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
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
            print("Game Over! The snake collided with itself.")
            pygame.quit()
            sys.exit()

        self.snake.insert(0, new_head)

        # Check for pellet collision
        if new_head in self.pellets:
            self.score += 1
            self.spawn_pellet()
        else:
            self.snake.pop()  # Remove last segment

    def update_direction(self):
        """Update the direction based on the fixed strategy."""
        if self.direction == UP and self.up_moves > 0:
            self.up_moves -= 1
        elif self.direction == UP and self.up_moves == 0:
            self.direction = RIGHT
            self.right_moves = 1
        elif self.direction == RIGHT and self.right_moves > 0:
            self.right_moves -= 1
            if self.right_moves == 0:
                self.direction = UP
                self.up_moves = self.GRID_HEIGHT - 2

    def run(self):
        while True:
            self.clock.tick(self.speed)
            self.handle_events()
            self.update_direction()
            self.move_snake()
            self.draw_elements()


def main():
    # You can set the game board size here (width, height)
    game = SnakeGameAI(width=200, height=200)
    game.run()


if __name__ == "__main__":
    main()
