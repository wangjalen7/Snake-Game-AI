import random
import numpy as np
import heapq
from collections import defaultdict, deque

from snake_game import GRID_HEIGHT, GRID_WIDTH


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
    """An agent that avoids moving towards walls while maintaining a buffer distance."""
    def __init__(self, buffer_distance=3):
        self.buffer_distance = buffer_distance  # Set the buffer distance to keep away from the walls

    def choose_action(self, game_state):
        snake_head = None
        food = None
        for y, row in enumerate(game_state):
            for x, cell in enumerate(row):
                if cell == 1 and not snake_head:
                    snake_head = (x, y)
                elif cell == 2:
                    food = (x, y)

        valid_moves = get_valid_moves(game_state, snake_head)

        # Debug: Print the valid moves to see what's available
        print(f"Valid moves: {valid_moves}")

        # Avoid moves that lead closer to walls but also seek new routes
        def distance_from_wall(position):
            x, y = position
            return min(x, len(game_state[0]) - x - 1, y, len(game_state) - y - 1)
        
        # Check if a move goes out of bounds or gets too close to a wall
        def is_safe_move(position):
            x, y = position
            return distance_from_wall(position) >= self.buffer_distance
    
        best_move = None
        best_score = float('-inf')

        for action, pos in valid_moves.items():
            if not is_safe_move(pos):  # Ignore moves that bring the snake too close to the wall
                continue

            food_distance = abs(food[0] - pos[0]) + abs(food[1] - pos[1]) if food else 0

            # Favor moves that bring the snake closer to food while avoiding walls
            score = -food_distance  # Negative so the agent moves closer to food

            if score > best_score or (score == best_score and random.random() < 0.5):
                best_move = action
                best_score = score

        # If no valid moves were safe, fallback to a random valid move
        if best_move is None:
            # Try to fallback to random but make sure it's a safe move
            for action, pos in valid_moves.items():
                if is_safe_move(pos):
                    best_move = action
                    break

        return best_move if best_move else random.choice(list(valid_moves.keys()))

    def get_current_direction(self, game_state, snake_head):
        """Get the current direction the snake is facing based on its head position and the direction it moved."""
        head_x, head_y = snake_head
        direction = game_state.get_direction()
        return direction
    
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # Q-table to store Q-values for state-action pairs

    def choose_action(self, state):
        """Choose an action based on epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: choose a random action
            return random.choice([UP, DOWN, LEFT, RIGHT])
        else:
            # Exploitation: choose the action with the highest Q-value for the current state
            state_key = self.get_state_key(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = {UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0}
            q_values = self.q_table[state_key]
            return max(q_values, key=q_values.get)

    def update_q_value(self, state, action, reward, next_state):
        """Update the Q-value for a given state-action pair."""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = {UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0}

        # Q-value update formula
        old_q_value = self.q_table[state_key][action]
        future_q_value = max(self.q_table[next_state_key].values())  # Max Q-value for next state
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * future_q_value - old_q_value)

        self.q_table[state_key][action] = new_q_value

    def get_state_key(self, state):
        """Convert a game state (2D grid) into a hashable state key."""
        return tuple(tuple(row) for row in state)  # Convert grid to a tuple of tuples for immutability
    
def valid_move(position, game_state):
    """Check if a move is valid (inside grid and not colliding with body)."""
    x, y = position
    return (0 <= x < GRID_WIDTH) and (0 <= y < GRID_HEIGHT) and game_state[y][x] != 1

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