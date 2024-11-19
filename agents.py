import random

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)


class Agent:
    """Base class for all agents."""

    def choose_action(self, game_state):
        raise NotImplementedError("This method should be overridden by subclasses.")


class RandomAgent(Agent):
    """An agent that picks a random valid action."""

    def __init__(self):
        self.actions = [UP, DOWN, LEFT, RIGHT]

    def choose_action(self, game_state):
        return random.choice(self.actions)


class GreedyAgent(Agent):
    """An agent that moves towards the food greedily while avoiding collisions."""
    def choose_action(self, game_state):
        snake_head = None
        food = None
        for y, row in enumerate(game_state):
            for x, cell in enumerate(row):
                if cell == 1 and not snake_head:  # Snake head
                    snake_head = (x, y)
                elif cell == 2:  # Food
                    food = (x, y)

        if not snake_head or not food:
            # If no valid game state is detected, move randomly as a fallback
            return random.choice([UP, DOWN, LEFT, RIGHT])

        # Get all valid moves
        valid_moves = get_valid_moves(game_state, snake_head)

        # Find the move that brings the snake closest to the food
        best_move = None
        best_distance = float('inf')
        for action, (nx, ny) in valid_moves.items():
            # Calculate Manhattan distance to the food
            distance = abs(food[0] - nx) + abs(food[1] - ny)
            if distance < best_distance:
                best_move = action
                best_distance = distance

        # If no valid move is found, fallback to a random move
        return best_move if best_move else random.choice(list(valid_moves.keys()))



def get_valid_moves(game_state, snake_head):
    """
    Return all valid moves for the snake based on the current game state.

    Args:
        game_state: A 2D grid representation of the game.
        snake_head: The current position of the snake's head (x, y).

    Returns:
        A dictionary of valid moves with their resulting positions.
    """
    moves = {
        UP: (snake_head[0], snake_head[1] - 1),
        DOWN: (snake_head[0], snake_head[1] + 1),
        LEFT: (snake_head[0] - 1, snake_head[1]),
        RIGHT: (snake_head[0] + 1, snake_head[1]),
    }

    valid_moves = {}
    for action, (nx, ny) in moves.items():
        if (
                0 <= nx < len(game_state[0])  # Check horizontal bounds
                and 0 <= ny < len(game_state)  # Check vertical bounds
                and game_state[ny][nx] != 1  # Avoid the snake's body
        ):
            valid_moves[action] = (nx, ny)

    return valid_moves
