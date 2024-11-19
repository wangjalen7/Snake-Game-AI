import pygame
import sys
import argparse
from snake_game import SnakeGame, LEFT, DOWN, UP, RIGHT, GRID_WIDTH, GRID_HEIGHT
from game_state import GameState
from agents import RandomAgent, GreedyAgent, QLearningAgent, WallAvoidanceAgent, PathfindingAgent, AStarAgent, DefensiveAgent

# Constants for rendering
GRID_SIZE = 20
SCREEN_WIDTH = GRID_SIZE * GRID_WIDTH
SCREEN_HEIGHT = GRID_SIZE * GRID_HEIGHT
FPS = 10

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

AGENTS = {
    "human": None,  # Human-controlled play
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "pathfinding": PathfindingAgent,
    "wall_avoidance": WallAvoidanceAgent,
    "a_star": AStarAgent,
    "defensive": DefensiveAgent,
    "qlearning": QLearningAgent,
}


def draw_game(screen, game):
    """Render the game on the screen."""
    screen.fill(BLACK)

    # Draw the snake
    for x, y in game.get_snake():
        pygame.draw.rect(screen, GREEN, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    # Draw the food
    food_x, food_y = game.get_food()
    pygame.draw.rect(screen, RED, (food_x * GRID_SIZE, food_y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    # Draw the score
    font = pygame.font.SysFont("Arial", 20)
    score_text = font.render(f"Score: {game.get_score()}", True, WHITE)
    screen.blit(score_text, (10, 10))  # Render score in the top-left corner

    pygame.display.flip()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Play Snake Game with different agents or as a human.")
    parser.add_argument("--agent", choices=AGENTS.keys(), default="human", help="Choose the agent to play the game.")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    # Initialize the game
    game = SnakeGame()

    # Initialize the selected agent
    agent_class = AGENTS[args.agent]
    agent = agent_class() if agent_class else None

    if args.agent == "qlearning":
        train_qlearning(agent)  # Train the Q-learning agent before starting the game

    while not game.is_game_over():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if args.agent == "human":
            # Handle human input
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                game.change_direction(UP)
            elif keys[pygame.K_DOWN]:
                game.change_direction(DOWN)
            elif keys[pygame.K_LEFT]:
                game.change_direction(LEFT)
            elif keys[pygame.K_RIGHT]:
                game.change_direction(RIGHT)

        elif agent:
            # Get the current game state as a 2D grid
            state = GameState(game).get_grid()

            # Agent chooses an action
            action = agent.choose_action(state)

            # Apply the agent's action
            game.change_direction(action)

        # Move the snake in the current direction
        game.step(game.direction)

        # Draw the game
        draw_game(screen, game)
        clock.tick(FPS)

    print(f"Game Over! Final score: {game.get_score():.2f}")
    print(f"Max score during the game: {game.get_max_score():.2f}")
    print(f"Average score per tick: {game.get_average_score():.2f}")
    pygame.quit()

# Train the Q-learning agent
def train_qlearning(agent, num_episodes=250000):
    for episode in range(num_episodes):
        game = SnakeGame()
        old_distance = None

        while not game.is_game_over():
            # Current state
            state = GameState(game).get_grid()
            snake_head = game.get_snake()[0]
            food_x, food_y = game.get_food()
            old_distance = abs(food_x - snake_head[0]) + abs(food_y - snake_head[1])

            # Choose an action
            action = agent.choose_action(state)
            game.change_direction(action)
            game.step(game.direction)

            # Next state
            next_state = GameState(game).get_grid()
            new_distance = abs(food_x - snake_head[0]) + abs(food_y - snake_head[1])

            # Define reward
            if game.is_game_over():
                reward = -50
            elif snake_head == (food_x, food_y):
                reward = 100
            else:
                reward = 10 if new_distance < old_distance else -1

            # Update Q-values
            agent.update_q_value(state, action, reward, next_state)

        # Log progress
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} completed. Epsilon: {agent.epsilon:.2f}. Score: {game.score}")

        # Decay epsilon
        agent.epsilon = max(0.01, agent.epsilon * 0.995)



if __name__ == "__main__":
    main()

