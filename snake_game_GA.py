import pygame
import random
import time
import pickle  # To save and load the best snake

# Initialize Pygame and set game settings
pygame.init()

# Note: We used ChatGPT to debug a number of things in this file. Some of the code throughout the file is written by AI.

# Game settings
width, height = 200, 200
block_size = 20
fps = 10
clock = pygame.time.Clock()

# Colors
white = (255, 255, 255)
green = (0, 255, 0)
red = (255, 0, 0)
black = (0, 0, 0)


# Snake class
class Snake:
    def __init__(self, genes=None):
        self.body = [(100, 50), (80, 50), (60, 50)]
        self.direction = (block_size, 0)
        self.dead = False
        self.score = 0
        self.genes = genes if genes else self.random_genes()
        self.steps_since_last_food = 0

    def random_genes(self):
        # Genes represent movement actions, represented by a list of directions
        return random.choices([(0, -block_size), (0, block_size), (-block_size, 0), (block_size, 0)], k=100)

    def move(self):
        # Move snake according to the current gene (direction)
        head_x, head_y = self.body[0]
        dir_x, dir_y = self.genes[int(self.score) % len(self.genes)]  # Ensure index is an integer
        new_head = (head_x + dir_x, head_y + dir_y)

        # Wrap-around behavior for the snake's head
        if new_head[0] >= width:  # Right side
            new_head = (0, new_head[1])
        elif new_head[0] < 0:  # Left side
            new_head = (width - block_size, new_head[1])
        if new_head[1] >= height:  # Bottom side
            new_head = (new_head[0], 0)
        elif new_head[1] < 0:  # Top side
            new_head = (new_head[0], height - block_size)

        self.body = [new_head] + self.body[:-1]

    def grow(self):
        # Add a new body part to the snake
        tail_x, tail_y = self.body[-1]
        dir_x, dir_y = self.direction
        new_tail = (tail_x - dir_x, tail_y - dir_y)
        self.body.append(new_tail)

    def collision_check(self):
        # Self collision
        head_x, head_y = self.body[0]
        if (head_x, head_y) in self.body[1:]:
            self.dead = True

    def distance_to_food(self, food):
        # Calculate Manhattan distance to food
        head_x, head_y = self.body[0]
        food_x, food_y = food.position
        return abs(head_x - food_x) + abs(head_y - food_y)

    def draw(self, screen):
        # Draw snake on screen
        for segment in self.body:
            pygame.draw.rect(screen, green, pygame.Rect(segment[0], segment[1], block_size, block_size))


# Food class
class Food:
    def __init__(self):
        self.position = (random.randrange(1, (width // block_size)) * block_size,
                         random.randrange(1, (height // block_size)) * block_size)

    def spawn(self):
        self.position = (random.randrange(1, (width // block_size)) * block_size,
                         random.randrange(1, (height // block_size)) * block_size)

    def draw(self, screen):
        pygame.draw.rect(screen, red, pygame.Rect(self.position[0], self.position[1], block_size, block_size))


# Genetic Algorithm Class
class GeneticAlgorithm:
    def __init__(self, population_size, initial_threshold=5, threshold_increase=1):
        self.population_size = population_size
        self.snakes = [Snake() for _ in range(population_size)]
        self.score_threshold = initial_threshold  # Initial threshold for score
        self.threshold_increase = threshold_increase  # How much to increase the threshold each generation

    def evaluate_population(self):
        # Simulate each snake and calculate their score (fitness)
        for snake in self.snakes:
            snake.score = self.play_game(snake)  # Custom function to simulate game and return score

        # Sort the snakes by score (high to low)
        self.snakes.sort(key=lambda s: s.score, reverse=True)

        # Kill off snakes below the score threshold
        self.snakes = [snake for snake in self.snakes if snake.score >= self.score_threshold]

        print(f"Threshold: {self.score_threshold}, Population after threshold: {len(self.snakes)} snakes remaining.")

        # If not enough snakes remain, we need to fill the population
        while len(self.snakes) < self.population_size:
            self.snakes.append(Snake())  # Adding new random snakes if needed

    def play_game(self, snake):
        food = Food()
        food.spawn()
        snake.body = [(100, 50), (80, 50), (60, 50)]
        snake.direction = (block_size, 0)
        snake.score = 0
        snake.dead = False
        max_moves = 200  # Max moves before killing the snake (if it doesn't eat food)

        survival_time = 0
        total_proximity_reward = 0

        while not snake.dead:
            snake.move()
            snake.collision_check()

            # Check if snake eats food
            if snake.body[0] == food.position:
                snake.grow()
                snake.score += 1
                food.spawn()
                snake.steps_since_last_food = 0  # Reset steps since last food
            else:
                snake.steps_since_last_food += 1  # Increase move counter if no food is eaten

            # Reward or penalize based on proximity to food
            reward = self.calculate_proximity_reward(snake, food)
            total_proximity_reward += reward
            snake.score += reward

            # End the game if too many steps without eating food
            if snake.steps_since_last_food > max_moves:
                snake.dead = True
                print(f"Snake died due to timeout after {max_moves} moves without eating food.")
                break

            clock.tick(fps)
            survival_time += 1

        # Apply the enhanced fitness check
        fitness_score = self.calculate_fitness(snake.score, survival_time, total_proximity_reward)
        print(f"Snake score: {snake.score}, Survival time: {survival_time}, Proximity reward: {total_proximity_reward}")
        return fitness_score

    def calculate_fitness(self, score, survival_time, total_proximity_reward):
        # Fitness function that combines score, survival time, and proximity reward
        score_weight = 0.5
        survival_weight = 0.3
        proximity_weight = 0.2

        # Higher score and survival time contribute to better fitness
        fitness = (score * score_weight) + (survival_time * survival_weight) + (
                    total_proximity_reward * proximity_weight)
        return fitness

    def calculate_proximity_reward(self, snake, food):
        current_distance = snake.distance_to_food(food)
        if current_distance < 10:  # Close to food
            return 1
        elif current_distance < 20:  # Somewhat close to food
            return 0.5
        else:
            return -0.1  # Penalize for being far from food

    def select_parents(self):
        # Select parents based on fitness (tournament or roulette wheel)
        parents = []
        for i in range(self.population_size // 2):
            parent1, parent2 = random.choices(self.snakes[:self.population_size // 2], k=2)
            parents.append((parent1, parent2))
        return parents

    def crossover(self, parent1, parent2):
        # Combine genes of parent1 and parent2 to produce offspring
        crossover_point = random.randint(1, len(parent1.genes) - 1)
        child_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        return Snake(genes=child_genes)

    def mutate(self, child):
        # Mutate the child's genes (random mutation)
        if random.random() < 0.1:  # Mutation probability
            mutation_point = random.randint(0, len(child.genes) - 1)
            child.genes[mutation_point] = random.choice(
                [(0, -block_size), (0, block_size), (-block_size, 0), (block_size, 0)])

    def evolve(self):
        self.evaluate_population()
        parents = self.select_parents()
        next_generation = []
        for parent1, parent2 in parents:
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            next_generation.append(child)
        self.snakes = next_generation

        # Gradually increase the score threshold after each generation
        self.score_threshold += self.threshold_increase


# Save the best snake's genes
def save_best_snake(best_snake):
    with open("best_snake.pkl", "wb") as f:
        pickle.dump(best_snake.genes, f)
    print("Best snake's genes saved!")


# Load the best snake's genes
def load_best_snake():
    try:
        with open("best_snake.pkl", "rb") as f:
            genes = pickle.load(f)
        return Snake(genes)
    except FileNotFoundError:
        print("No saved snake found!")
        return None


# Main loop for evolving the snake
def game_loop():
    generations = 10
    population_size = 50
    ga = GeneticAlgorithm(population_size)

    # Run through all generations
    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        ga.evolve()

    # After all generations, save the best snake
    best_snake = ga.snakes[0]
    print(f"Best snake's score: {best_snake.score}")
    save_best_snake(best_snake)


# Main loop for running the best snake
def play_best_snake():
    # Load the best snake's genes
    best_snake = load_best_snake()
    if best_snake is None:
        return

    # Initialize Pygame window when starting to play the best snake
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Playing Best Evolved Snake")
    clock = pygame.time.Clock()

    food = Food()
    food.spawn()
    best_snake.body = [(100, 50), (80, 50), (60, 50)]
    best_snake.direction = (block_size, 0)
    best_snake.score = 0
    best_snake.dead = False

    # Event handling for closing the window
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        best_snake.move()
        best_snake.collision_check()

        # Check if snake eats food
        if best_snake.body[0] == food.position:
            best_snake.grow()
            best_snake.score += 1
            food.spawn()

        screen.fill(black)  # Clear the screen before drawing
        best_snake.draw(screen)
        food.draw(screen)
        pygame.display.update()

        clock.tick(fps)

    print(f"Final score of best snake: {best_snake.score}")
    pygame.quit()


# Ask user whether to evolve or play the best snake
def main():
    while True:
        choice = input("Enter 'e' to start evolving or 'p' to play the best snake: ").lower()
        if choice == 'e':
            game_loop()  # Run the genetic algorithm without opening Pygame
            break
        elif choice == 'p':
            play_best_snake()  # Open Pygame window to play the best snake
            break
        else:
            print("Invalid choice, please enter 'e' or 'p'.")


if __name__ == "__main__":
    main()
