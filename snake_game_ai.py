import pygame
import random
import sys
import heapq

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
        self.snake = [(self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)]
        self.score = 0
        self.speed = 10  # Game speed in frames per second
        self.pellets = []
        self.spawn_pellets(2)  # Initially spawn two pellets
        self.path = []

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

    def manhattan_distance(self, a, b):
        dx = min(abs(a[0] - b[0]), self.GRID_WIDTH - abs(a[0] - b[0]))
        dy = min(abs(a[1] - b[1]), self.GRID_HEIGHT - abs(a[1] - b[1]))
        return dx + dy

    def get_neighbors(self, position):
        neighbors = []
        for direction in DIRECTIONS:
            dx, dy = DIRECTION_VECTORS[direction]
            new_x = (position[0] + dx) % self.GRID_WIDTH
            new_y = (position[1] + dy) % self.GRID_HEIGHT
            neighbors.append(((new_x, new_y), direction))
        return neighbors

    def a_star(self):
        start = self.snake[0]
        goals = self.pellets.copy()
        obstacles = set(self.snake[1:])
        open_set = []
        # Push tuples of (estimated_total_cost, cost_so_far, current_position, path)
        for goal in goals:
            heapq.heappush(open_set, (self.manhattan_distance(start, goal), 0, start, []))
        closed_set = set()

        while open_set:
            est_total_cost, cost_so_far, current_pos, path = heapq.heappop(open_set)

            if current_pos in goals:
                return path  # Found the path to a pellet

            if current_pos in closed_set:
                continue
            closed_set.add(current_pos)

            for neighbor_pos, direction in self.get_neighbors(current_pos):
                if neighbor_pos in obstacles or neighbor_pos in closed_set:
                    continue
                new_cost = cost_so_far + 1
                heuristic = min(self.manhattan_distance(neighbor_pos, goal) for goal in goals)
                est_total = new_cost + heuristic
                heapq.heappush(open_set, (est_total, new_cost, neighbor_pos, path + [direction]))
        return None  # No path found

    def handle_events(self):
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def move_snake(self, direction):
        dx, dy = DIRECTION_VECTORS[direction]
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
            self.pellets.remove(new_head)
            self.spawn_pellets(2)  # Ensure there are always two pellets
        else:
            self.snake.pop()  # Remove last segment

    def run(self):
        while True:
            self.clock.tick(self.speed)
            self.handle_events()

            # Check if path is empty or invalid
            if not self.path:
                self.path = self.a_star()
                if not self.path:
                    # No path found; end the game
                    print("No path to any pellet. Game Over!")
                    pygame.quit()
                    sys.exit()

            # Get next move
            direction = self.path.pop(0)
            self.move_snake(direction)
            self.draw_elements()

def main():
    # You can set the game board size here (width, height)
    game = SnakeGameAI(width=200, height=200)
    game.run()

if __name__ == "__main__":
    main()
