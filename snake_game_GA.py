import pygame
import random
import numpy as np

# Initialize Pygame
pygame.init()

CELL_SIZE = 20

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREEN = (0, 180, 0)

# Directions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
DIRECTION_VECTORS = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0)
}


class SnakeGame:
    def __init__(self, width=200, height=200, render=False):
        self.WINDOW_WIDTH = width
        self.WINDOW_HEIGHT = height
        self.GRID_WIDTH = self.WINDOW_WIDTH // CELL_SIZE
        self.GRID_HEIGHT = self.WINDOW_HEIGHT // CELL_SIZE

        self.render_game = render
        if self.render_game:
            self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            pygame.display.set_caption('Snake Genetic Algorithm')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 24)
        self.reset()

    def reset(self):
        self.snake = [(self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.score = 0
        self.pellet = self.spawn_pellet()
        return self.get_state()

    def spawn_pellet(self):
        while True:
            pellet = (
                random.randint(0, self.GRID_WIDTH - 1),
                random.randint(0, self.GRID_HEIGHT - 1)
            )
            if pellet not in self.snake:
                return pellet

    def is_collision(self, position):
        x, y = position
        if (x, y) in self.snake[1:]:
            return True
        return False

    def get_state(self):
        head = self.snake[0]
        direction = self.direction

        danger_straight = self.is_collision((
            (head[0] + DIRECTION_VECTORS[direction][0]) % self.GRID_WIDTH,
            (head[1] + DIRECTION_VECTORS[direction][1]) % self.GRID_HEIGHT
        ))

        danger_right = self.is_collision((
            (head[0] + DIRECTION_VECTORS[(direction + 1) % 4][0]) % self.GRID_WIDTH,
            (head[1] + DIRECTION_VECTORS[(direction + 1) % 4][1]) % self.GRID_HEIGHT
        ))

        danger_left = self.is_collision((
            (head[0] + DIRECTION_VECTORS[(direction - 1) % 4][0]) % self.GRID_WIDTH,
            (head[1] + DIRECTION_VECTORS[(direction - 1) % 4][1]) % self.GRID_HEIGHT
        ))

        food_direction = (
            int(self.pellet[0] < head[0]),
            int(self.pellet[0] > head[0]),
            int(self.pellet[1] < head[1]),
            int(self.pellet[1] > head[1])
        )

        return (danger_straight, danger_right, danger_left, direction) + food_direction

    def step(self, action):
        if action == 1:  # Right turn
            self.direction = (self.direction + 1) % 4
        elif action == 2:  # Left turn
            self.direction = (self.direction - 1) % 4

        dx, dy = DIRECTION_VECTORS[self.direction]
        new_head = (
            (self.snake[0][0] + dx) % self.GRID_WIDTH,
            (self.snake[0][1] + dy) % self.GRID_HEIGHT
        )

        reward = 0
        done = False

        if new_head in self.snake:
            reward = -10
            done = True
            return self.get_state(), reward, done, self.score

        self.snake.insert(0, new_head)
        if new_head == self.pellet:
            self.score += 1
            reward = 10
            self.pellet = self.spawn_pellet()
        else:
            self.snake.pop()

        return self.get_state(), reward, done, self.score

    def render(self):
        if not self.render_game:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill(BLACK)
        for segment in self.snake:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, GREEN, rect)

        pellet_rect = pygame.Rect(self.pellet[0] * CELL_SIZE, self.pellet[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, RED, pellet_rect)

        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (5, 5))

        pygame.display.flip()
        self.clock.tick(20)


class GeneticAgent:
    def __init__(self, chromosome_length):
        self.chromosome = [random.uniform(-1, 1) for _ in range(chromosome_length)]

    def choose_action(self, state):
        inputs = np.array(state)
        weights = np.array(self.chromosome[:len(state)])
        weighted_sum = np.dot(inputs, weights)

        if weighted_sum < -0.5:
            return 2  # Left
        elif weighted_sum > 0.5:
            return 1  # Right
        else:
            return 0  # Straight


def evaluate_agent(agent, game, episodes=5):
    total_score = 0
    for _ in range(episodes):
        state = game.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            state, _, done, score = game.step(action)
        total_score += score
    return total_score / episodes


def genetic_algorithm(game, population_size=50, generations=20, mutation_rate=0.1):
    chromosome_length = len(game.get_state())
    population = [GeneticAgent(chromosome_length) for _ in range(population_size)]

    for generation in range(generations):
        fitness_scores = [evaluate_agent(agent, game) for agent in population]
        sorted_population = [agent for _, agent in sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)]

        print(f"Generation {generation}: Best Score = {max(fitness_scores)}")

        next_generation = sorted_population[:population_size // 2]

        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(sorted_population[:10], 2)
            child_chromosome = crossover(parent1.chromosome, parent2.chromosome)
            child_chromosome = mutate(child_chromosome, mutation_rate)
            next_generation.append(GeneticAgent(chromosome_length))
            next_generation[-1].chromosome = child_chromosome

        population = next_generation

    best_agent = population[0]
    return best_agent


def crossover(parent1, parent2):
    point = random.randint(0, len(parent1) - 1)
    return parent1[:point] + parent2[point:]


def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] += random.uniform(-0.1, 0.1)
    return chromosome


def main():
    game = SnakeGame(width=200, height=200, render=True)
    best_agent = genetic_algorithm(game)

    print("Testing the best agent...")
    for _ in range(1):
        state = game.reset()
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            action = best_agent.choose_action(state)
            state, _, done, _ = game.step(action)
            game.render()

    pygame.quit()


if __name__ == "__main__":
    main()
