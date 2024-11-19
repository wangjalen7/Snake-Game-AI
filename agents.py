import random
import heapq
from collections import deque


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


class PathfindingAgent(Agent):
    """An agent that uses BFS to find the shortest path to food."""
    def choose_action(self, game_state):
        snake_head = None
        food = None
        for y, row in enumerate(game_state):
            for x, cell in enumerate(row):
                if cell == 1 and not snake_head:
                    snake_head = (x, y)
                elif cell == 2:
                    food = (x, y)

        if not snake_head or not food:
            return random.choice([UP, DOWN, LEFT, RIGHT])

        # BFS to find the shortest path to food
        queue = deque([(snake_head, [])])
        visited = set()
        visited.add(snake_head)

        while queue:
            position, path = queue.popleft()

            if position == food:
                return path[0] if path else random.choice([UP, DOWN, LEFT, RIGHT])

            for action, next_pos in get_valid_moves(game_state, position).items():
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [action]))

        # If no path found, fallback to a random valid move
        return random.choice(list(get_valid_moves(game_state, snake_head).keys()))


class AStarAgent(Agent):
    """An agent that uses A* search to move towards the food."""
    def choose_action(self, game_state):
        snake_head = None
        food = None
        for y, row in enumerate(game_state):
            for x, cell in enumerate(row):
                if cell == 1 and not snake_head:
                    snake_head = (x, y)
                elif cell == 2:
                    food = (x, y)

        if not snake_head or not food:
            return random.choice([UP, DOWN, LEFT, RIGHT])

        # A* search
        open_set = []
        heapq.heappush(open_set, (0, snake_head, []))  # (priority, position, path)
        visited = set()

        def heuristic(pos):
            return abs(pos[0] - food[0]) + abs(pos[1] - food[1])  # Manhattan distance

        while open_set:
            _, position, path = heapq.heappop(open_set)

            if position in visited:
                continue
            visited.add(position)

            if position == food:
                return path[0] if path else random.choice([UP, DOWN, LEFT, RIGHT])

            for action, next_pos in get_valid_moves(game_state, position).items():
                if next_pos not in visited:
                    priority = len(path) + 1 + heuristic(next_pos)
                    heapq.heappush(open_set, (priority, next_pos, path + [action]))

        # If no path found, fallback to a random valid move
        valid_moves = list(get_valid_moves(game_state, snake_head).keys())
        return random.choice(valid_moves)

class DefensiveAgent(Agent):
    """An agent that prioritizes survival over reaching the food."""
    def choose_action(self, game_state):
        snake_head = None
        for y, row in enumerate(game_state):
            for x, cell in enumerate(row):
                if cell == 1 and not snake_head:  # Snake head
                    snake_head = (x, y)

        valid_moves = get_valid_moves(game_state, snake_head)

        # Prioritize moves that keep the snake alive
        safe_moves = {
            action: pos for action, pos in valid_moves.items()
            if game_state[pos[1]][pos[0]] != 1  # Avoid snake's body
        }

        # Default to random valid move if no "safe" move exists
        return random.choice(list(safe_moves.keys())) if safe_moves else random.choice(list(valid_moves.keys()))

class WallAvoidanceAgent(Agent):
    """An agent that avoids moving towards walls."""
    def choose_action(self, game_state):
        snake_head = None
        for y, row in enumerate(game_state):
            for x, cell in enumerate(row):
                if cell == 1 and not snake_head:  # Snake head
                    snake_head = (x, y)

        valid_moves = get_valid_moves(game_state, snake_head)

        # Avoid moves that lead closer to walls but also seek new routes
        def distance_from_wall(position):
            x, y = position
            return min(x, len(game_state[0]) - x - 1, y, len(game_state) - y - 1)

        best_move = None
        best_distance = -1
        for action, pos in valid_moves.items():
            dist = distance_from_wall(pos)
            if dist > best_distance:
                best_move = action
                best_distance = dist

        # Avoid situations where the snake continues in a loop
        # Re-evaluate position based on additional factors, such as food position
        if not best_move:
            best_move = random.choice(list(valid_moves.keys()))

        return best_move


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

    import heapq