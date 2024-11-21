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

    # Initialize the selected agent
    agent_class = AGENTS[args.agent]
    agent = agent_class() if agent_class else None

    # If QLearning agent, train it before starting the game
    if args.agent == "qlearning":
        train_qlearning(agent)  # Train the Q-learning agent before starting the game

    # Initialize the game
    game = SnakeGame()

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
            state = GameState(game)

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
def train_qlearning(agent):
    episodes = 10000
    for episode in range(episodes):
        game = SnakeGame()  # Reset the game for each episode
        state = GameState(game).get_grid()

        while not game.is_game_over():
            action = agent.choose_action(state)
            previous_state = state

            game.step(action)
            state = GameState(game).get_grid()
            reward = game.get_reward()

            agent.learn(previous_state, action, reward, state, game.get_food(), game.is_game_over())

            # draw_game(screen, game)
            # clock.tick(FPS)

        agent.decay_exploration()

        print(f"Episode {episode + 1}/{episodes} completed. Exploration rate: {agent.exploration_rate:.4f}. Score: {game.score}")

    print("Training completed.")
    # pygame.quit()

if __name__ == "__main__":
    main()

