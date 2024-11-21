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
        grid = game_state.get_grid()  # Get the grid representation
        snake_head = None
        food = None
        for y, row in enumerate(grid):
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
        grid = game_state.get_grid()
        snake_head = None
        food = None
        for y, row in enumerate(grid):
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

    def choose_action(self, state):
        snake_body = state.get_snake_body()
        snake_head = snake_body[0]
        food = state.get_food()

        if not snake_head or not food:
            return random.choice([UP, DOWN, LEFT, RIGHT])

        open_set = []
        initial_body = snake_body.copy()
        initial_path = []
        heapq.heappush(open_set, (0, snake_head, initial_body, initial_path))
        visited = set()

        actions = [UP, DOWN, LEFT, RIGHT]

        while open_set:
            priority, position, body, path = heapq.heappop(open_set)

            state_key = (position, tuple(body))
            if state_key in visited:
                continue
            visited.add(state_key)

            if position == food:
                return path[0] if path else random.choice(actions)

            for action in actions:
                dx, dy = action
                new_head = (position[0] + dx, position[1] + dy)

                # Check for wall collision
                if not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
                    continue

                # Check for body collision
                if new_head in body:
                    continue

                food_eaten = new_head == food

                if food_eaten:
                    # Snake grows
                    new_body = [new_head] + body
                else:
                    # Snake moves, tail moves forward
                    new_body = [new_head] + body[:-1]

                # No safety check here; we proceed to explore this path
                new_path = path + [action]
                priority = len(new_path) + self.heuristic(new_head, food)
                heapq.heappush(open_set, (priority, new_head, new_body, new_path))

        # If no path found, fallback to a random valid move
        valid_moves = self.get_valid_moves(state.get_grid(), snake_head, snake_body)
        return random.choice(list(valid_moves.keys()))

    def heuristic(self, pos, goal):
        """Return the heuristic value (estimated cost) from pos to goal."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def get_valid_moves(self, grid, snake_head, snake_body):
        """Return all valid moves for the snake based on the current game state."""
        moves = {
            UP: (snake_head[0], snake_head[1] - 1),
            DOWN: (snake_head[0], snake_head[1] + 1),
            LEFT: (snake_head[0] - 1, snake_head[1]),
            RIGHT: (snake_head[0] + 1, snake_head[1]),
        }

        valid_moves = {}
        for action, (nx, ny) in moves.items():
            if (
                0 <= nx < GRID_WIDTH  # Check horizontal bounds
                and 0 <= ny < GRID_HEIGHT  # Check vertical bounds
                and (nx, ny) not in snake_body  # Avoid the snake's body
            ):
                valid_moves[action] = (nx, ny)

        return valid_moves


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
        # print(f"Valid moves: {valid_moves}")

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
    """An agent that learns using the Q-learning algorithm."""

    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        self.q_table = defaultdict(lambda: {UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0})
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = 0.01

    def choose_action(self, state):
        """Choose an action using epsilon-greedy strategy."""
        state_key = self._state_to_key(state)
        if random.random() < self.exploration_rate:
            return random.choice([UP, DOWN, LEFT, RIGHT])
        else:
            return max(self.q_table[state_key], key=self.q_table[state_key].get)

    def learn(self, state, action, reward, next_state, food_position, game_over):
        """Update the Q-table based on the action taken and the reward received."""
        # Update the reward based on the custom reward function
        custom_reward = self.get_reward(state, state[0], food_position, game_over)

        # Update Q-values with the custom reward
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        # Q-value update logic
        best_next_action = max(self.q_table[next_state_key], key=self.q_table[next_state_key].get)
        target = custom_reward + self.discount_factor * self.q_table[next_state_key][best_next_action]
        self.q_table[state_key][action] += self.learning_rate * (target - self.q_table[state_key][action])

        # Decay the exploration rate
        self.decay_exploration()


    def decay_exploration(self):
        """Decay the exploration rate over time."""
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def get_reward(self, game_state, snake_head, food_position, game_over):
        """Define a custom reward function based on the game state."""
        reward = 0

        if game_over:
            reward = -100  # Big negative reward for game over (death)
        else:
            # Reward for eating food
            if snake_head == food_position:
                reward = 10  # Positive reward for eating food
            else:
                # Penalize for moving towards walls (getting closer to boundaries)
                distance_from_wall = min(snake_head[0], GRID_WIDTH - 1 - snake_head[0], snake_head[1], GRID_HEIGHT - 1 - snake_head[1])
                if distance_from_wall < 3:  # If snake is too close to the wall
                    reward -= 1  # Small negative reward for being close to walls

                # Reward for moving closer to food
                food_distance_before = abs(snake_head[0] - food_position[0]) + abs(snake_head[1] - food_position[1])
                food_distance_after = abs(snake_head[0] - food_position[0]) + abs(snake_head[1] - food_position[1])
                if food_distance_after < food_distance_before:
                    reward += 1  # Reward for getting closer to food

        return reward

    @staticmethod
    def _state_to_key(state):
        """Convert the 2D grid state into a hashable key for the Q-table."""
        return tuple(map(tuple, state))
    
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
    grid = game_state.get_grid()
    valid_moves = {}
    for action, (nx, ny) in moves.items():
        if (
                0 <= nx < len(grid[0])  # Check horizontal bounds
                and 0 <= ny < len(grid)  # Check vertical bounds
                and grid[ny][nx] != 1  # Avoid the snake's body
        ):
            valid_moves[action] = (nx, ny)

    return valid_moves
