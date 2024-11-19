import random

# Constants for the game
GRID_WIDTH = 30  # Number of grid cells in width
GRID_HEIGHT = 20  # Number of grid cells in height

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

PELLET_REWARD = 100
TIME_PENALTY = 1


class SnakeGame:
    """Main class for Snake game logic."""

    def __init__(self):
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]  # Snake starts in the center
        self.direction = RIGHT  # Initial direction
        self.food = self.spawn_food()
        self.game_over = False
        self.score = 0
        self.total_score = 0  # Track sum of scores at each tick
        self.max_score = 0  # Track max score at any tick
        self.tick_count = 0  # Count the number of ticks the game took

    def spawn_food(self):
        """Spawn food in a random position not occupied by the snake."""
        while True:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food not in self.snake:
                return food

    def step(self, action):
        """Advance the game by one step, applying the given action."""
        self.change_direction(action)
        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)

        # Check collisions
        if (
                new_head in self.snake or  # Snake collides with itself
                new_head[0] < 0 or new_head[0] >= GRID_WIDTH or  # Wall collision
                new_head[1] < 0 or new_head[1] >= GRID_HEIGHT
        ):
            self.game_over = True
            return

        # Move the snake
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.food = self.spawn_food()
            self.score += PELLET_REWARD
        else:
            self.score -= TIME_PENALTY
            self.snake.pop()  # Remove the tail

        # Track score at each tick
        self.total_score += self.score
        self.max_score = max(self.max_score, self.score)
        self.tick_count += 1

    def change_direction(self, new_direction):
        """Change direction unless it's directly opposite to the current direction."""
        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction

    def get_snake(self):
        """Return the current snake position."""
        return self.snake

    def get_food(self):
        """Return the current food position."""
        return self.food

    def is_game_over(self):
        """Return whether the game is over."""
        return self.game_over

    def get_score(self):
        """Return the current score."""
        return self.score

    def get_average_score(self):
        """Return the average score per tick."""
        return self.total_score / self.tick_count if self.tick_count > 0 else 0

    def get_max_score(self):
        """Return the maximum score at any tick."""
        return self.max_score
    
    def get_reward(self):
        """Return the reward for the current state."""
        if self.is_collision():
            return -10  # Penalty for collision
        elif self.snake_eats_food():
            return 25   # Reward for eating food
        else:
            return 0   # Neutral reward for other moves
