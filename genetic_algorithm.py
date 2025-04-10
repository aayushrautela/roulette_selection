import random
import numpy as np
from functions import booth_2d, init_ranges

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        mutation_strength: float,
        crossover_rate: float,
        num_generations: int,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations

    def initialize_population(self):
        x_range, y_range = init_ranges[booth_2d]
        population = [
            (np.random.uniform(*x_range), np.random.uniform(*y_range))
            for _ in range(self.population_size)
        ]
        return population

    def evaluate_population(self, population):
        fitness = [1 / (1 + booth_2d(x, y)) for x, y in population]  # minimize cost → maximize fitness
        return fitness

    def selection(self, population, fitness_values):
        total_fitness = sum(fitness_values)
        probabilities = [f / total_fitness for f in fitness_values]
        selected = random.choices(population, weights=probabilities, k=self.population_size)
        return selected

    def crossover(self, parents):
        offspring = []
        for i in range(0, self.population_size, 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % self.population_size]
            if random.random() < self.crossover_rate:
                alpha = random.random()
                x = alpha * p1[0] + (1 - alpha) * p2[0]
                y = alpha * p1[1] + (1 - alpha) * p2[1]
                offspring.append((x, y))
                offspring.append((y, x))  # simple second child
            else:
                offspring.append(p1)
                offspring.append(p2)
        return offspring[:self.population_size]

    def mutate(self, individuals):
        mutated = []
        for x, y in individuals:
            if random.random() < self.mutation_rate:
                x += np.random.normal(0, self.mutation_strength)
                y += np.random.normal(0, self.mutation_strength)
            mutated.append((x, y))
        return mutated

    def evolve(self, seed: int):
        set_seed(seed)

        population = self.initialize_population()

        best_solutions = []
        best_fitness_values = []
        average_fitness_values = []

        for generation in range(self.num_generations):
            fitness_values = self.evaluate_population(population)

            best_idx = np.argmax(fitness_values)
            best_solutions.append(population[best_idx])
            best_fitness_values.append(fitness_values[best_idx])
            average_fitness_values.append(np.mean(fitness_values))

            parents = self.selection(population, fitness_values)
            offspring = self.crossover(parents)
            population = self.mutate(offspring)

        return best_solutions, best_fitness_values, average_fitness_values
