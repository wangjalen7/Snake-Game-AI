import pygame
import random
import sys
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

class SnakeGameAI:
    def __init__(self):
        # Set up the display
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('Snake Game with AI')

        # Set up the clock
        self.clock = pygame.time.Clock()

        # Font for score display
        self.font = pygame.font.SysFont('Arial', 24)

        # Game Variables
        self.reset()

    def reset(self):
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.score = 0
        self.speed = 10  # Game speed in frames per second
        self.spawn_pellet()
        self.path = []

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

    def manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, position):
        neighbors = []
        for direction in DIRECTIONS:
            dx, dy = DIRECTION_VECTORS[direction]
            new_x = position[0] + dx
            new_y = position[1] + dy
            if 0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT:
                neighbors.append(((new_x, new_y), direction))
        return neighbors

    def a_star(self):
        start = self.snake[0]
        goal = self.pellet
        obstacles = set(self.snake[1:])
        open_set = []
        heapq.heappush(open_set, (0 + self.manhattan_distance(start, goal), 0, start, []))
        closed_set = set()

        while open_set:
            est_total_cost, cost_so_far, current_pos, path = heapq.heappop(open_set)

            if current_pos == goal:
                return path  # Found the path

            if current_pos in closed_set:
                continue
            closed_set.add(current_pos)

            for neighbor_pos, direction in self.get_neighbors(current_pos):
                if neighbor_pos in obstacles or neighbor_pos in closed_set:
                    continue
                new_cost = cost_so_far + 1
                est_total_cost = new_cost + self.manhattan_distance(neighbor_pos, goal)
                heapq.heappush(open_set, (est_total_cost, new_cost, neighbor_pos, path + [direction]))
        return None  # No path found

    def handle_events(self):
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def move_snake(self, direction):
        dx, dy = DIRECTION_VECTORS[direction]
        new_head = (self.snake[0][0] + dx, self.snake[0][1] + dy)

        # Check for collision with walls
        if not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
            print("Game Over! The snake hit a wall.")
            pygame.quit()
            sys.exit()

        # Collision Detection with self
        if new_head in self.snake:
            print("Game Over! The snake collided with itself.")
            pygame.quit()
            sys.exit()

        self.snake.insert(0, new_head)

        # Check for pellet collision
        if new_head == self.pellet:
            self.score += 1
            self.spawn_pellet()
        else:
            self.snake.pop()  # Remove last segment

    def run(self):
        while True:
            self.clock.tick(self.speed)
            self.handle_events()

            # Get obstacles (snake body excluding the head)
            obstacles = set(self.snake[1:])

            # Check if path is empty or invalid
            if not self.path:
                self.path = self.a_star()
                if not self.path:
                    # No path found; end the game
                    print("No path to pellet. Game Over!")
                    pygame.quit()
                    sys.exit()

            # Get next move
            direction = self.path.pop(0)
            self.move_snake(direction)
            self.draw_elements()

def main():
    game = SnakeGameAI()
    game.run()

if __name__ == "__main__":
    main()
