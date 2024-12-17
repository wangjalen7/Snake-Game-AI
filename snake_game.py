import pygame
import random
import sys

# We used ChatGPT to generate the basic Snake game and then refactored the code a little bit to fine tune
# the features we needed as well as make it easier to work with for the agents we were testing.

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


class SnakeGame:
    def __init__(self, width=600, height=400):
        # Game Dimensions
        self.WINDOW_WIDTH = width
        self.WINDOW_HEIGHT = height

        # Grid Dimensions
        self.GRID_WIDTH = self.WINDOW_WIDTH // CELL_SIZE
        self.GRID_HEIGHT = self.WINDOW_HEIGHT // CELL_SIZE

        # Set up the display
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption('Snake Game')

        # Set up the clock
        self.clock = pygame.time.Clock()

        # Font for score display
        self.font = pygame.font.SysFont('Arial', 24)

        # Game Variables
        self.reset()

    def reset(self):
        self.snake = [(self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)]
        self.direction = RIGHT
        self.score = 0
        self.speed = 10  # Game speed in frames per second
        self.pellets = []
        self.spawn_pellets(2)  # Initially spawn two pellets

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

    def handle_events(self):
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and self.direction != DOWN:
                    self.direction = UP
                elif event.key == pygame.K_DOWN and self.direction != UP:
                    self.direction = DOWN
                elif event.key == pygame.K_LEFT and self.direction != RIGHT:
                    self.direction = LEFT
                elif event.key == pygame.K_RIGHT and self.direction != LEFT:
                    self.direction = RIGHT

    def move_snake(self):
        dx, dy = DIRECTION_VECTORS[self.direction]
        new_head = (
            (self.snake[0][0] + dx) % self.GRID_WIDTH,
            (self.snake[0][1] + dy) % self.GRID_HEIGHT
        )

        # Collision Detection with self
        if new_head in self.snake:
            print("Game Over! You collided with yourself.")
            pygame.quit()
            sys.exit()

        self.snake.insert(0, new_head)

        # Check for pellet collision
        if new_head in self.pellets:
            self.score += 1
            self.pellets.remove(new_head)
            self.spawn_pellets(2)  # Ensure there are always two pellets
        else:
            self.snake.pop()  # Remove last segment

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

    def run(self):
        while True:
            self.clock.tick(self.speed)
            self.handle_events()
            self.move_snake()
            self.draw_elements()


def main():
    # You can set the game board size here (width, height) but remember cell size is 20
    game = SnakeGame(width=800, height=600)
    game.run()


if __name__ == "__main__":
    main()
