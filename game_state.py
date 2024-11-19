from snake_game import GRID_WIDTH, GRID_HEIGHT


class GameState:
    """Convert the game state into a 2D grid representation, inlcuding direction."""

    def __init__(self, game):
        self.game = game

    def get_grid(self):
        """Return a 2D grid representation of the game."""
        grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

        # Mark the snake on the grid
        for x, y in self.game.get_snake():
            grid[y][x] = 1  # Snake body is represented as 1

        # Mark the food on the grid
        food_x, food_y = self.game.get_food()
        grid[food_y][food_x] = 2  # Food is represented as 2

        return grid
    
    def get_direction(self):
        """Return the current direction of the snake."""
        return self.game.direction
