import pygame
import sys
import argparse
from snake_game import SnakeGame, LEFT, DOWN, UP, RIGHT, GRID_WIDTH, GRID_HEIGHT
from game_state import GameState
from agents import RandomAgent, GreedyAgent

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
    "greedy": GreedyAgent
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

    while not game.is_game_over():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if args.agent == "human":
            # Handle human input
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                game.step(UP)
            elif keys[pygame.K_DOWN]:
                game.step(DOWN)
            elif keys[pygame.K_LEFT]:
                game.step(LEFT)
            elif keys[pygame.K_RIGHT]:
                game.step(RIGHT)
        else:
            # Get the current game state as a 2D grid
            state = GameState(game).get_grid()

            # Agent chooses an action
            action = agent.choose_action(state)

            # Apply the agent's action
            game.step(action)

        # Draw the game
        draw_game(screen, game)
        clock.tick(FPS)

    print(f"Game Over! Final score: {game.get_score():.2f}")
    print(f"Max score during the game: {game.get_max_score():.2f}")
    print(f"Average score per tick: {game.get_average_score():.2f}")
    pygame.quit()


if __name__ == "__main__":
    main()
