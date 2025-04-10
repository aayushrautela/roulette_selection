import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm
import numpy as np
import pandas as pd

def run_experiment_1():
    print("\n--- Experiment 1: Parameter Tuning ---")
    combinations = [
        (30, 0.1, 0.05, 0.7),
        (50, 0.2, 0.1, 0.8),
        (70, 0.3, 0.1, 0.9),
        (100, 0.1, 0.2, 0.6),
        (50, 0.4, 0.05, 0.8),
    ]

    results = []

    for i, (pop, mut_rate, mut_strength, cross_rate) in enumerate(combinations):
        ga = GeneticAlgorithm(
            population_size=pop,
            mutation_rate=mut_rate,
            mutation_strength=mut_strength,
            crossover_rate=cross_rate,
            num_generations=100,
        )
        best_solutions, best_fitness_values, _ = ga.evolve(seed=42)
        best_solution = best_solutions[-1]
        best_fitness = best_fitness_values[-1]
        results.append({
            "Combination": i + 1,
            "Population Size": pop,
            "Mutation Rate": mut_rate,
            "Mutation Strength": mut_strength,
            "Crossover Rate": cross_rate,
            "Best Fitness": round(best_fitness, 6),
            "Best Solution (x, y)": f"{best_solution[0]:.4f}, {best_solution[1]:.4f}"
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv("experiment_1_results.csv", index=False)

def run_experiment_2():
    print("\n--- Experiment 2: Random Seeds + Smaller Population ---")
    base_config = {
        "population_size": 100,
        "mutation_rate": 0.1,
        "mutation_strength": 0.2,
        "crossover_rate": 0.6,
        "num_generations": 100,
    }

    seeds = [1, 10, 23, 42, 99]
    fitness_results = []

    for seed in seeds:
        ga = GeneticAlgorithm(**base_config)
        _, best_fitness_values, _ = ga.evolve(seed)
        fitness_results.append(best_fitness_values[-1])

    best_fit = np.max(fitness_results)
    avg_fit = np.mean(fitness_results)
    std_fit = np.std(fitness_results)

    print(f"Best Fitness: {best_fit:.6f}")
    print(f"Average Fitness: {avg_fit:.6f}")
    print(f"Standard Deviation: {std_fit:.6f}")

    print("\nRunning with decreasing population sizes...")
    reduction_results = []

    for ratio in [0.5, 0.25, 0.1]:
        config = base_config.copy()
        config["population_size"] = int(config["population_size"] * ratio)
        ga = GeneticAlgorithm(**config)
        _, best_fitness_values, _ = ga.evolve(seed=42)
        reduction_results.append({
            "Population Size": config["population_size"],
            "Best Fitness": round(best_fitness_values[-1], 6)
        })

    df = pd.DataFrame(reduction_results)
    print(df.to_string(index=False))
    df.to_csv("experiment_2_results.csv", index=False)

def run_experiment_3():
    print("\n--- Experiment 3: Crossover Impact ---")
    crossover_rates = [0.2, 0.5, 0.8, 1.0]
    seeds = [10, 23, 42]
    config_base = {
        "population_size": 100,
        "mutation_rate": 0.1,
        "mutation_strength": 0.2,
        "num_generations": 100
    }

    plt.figure(figsize=(10, 6))
    for rate in crossover_rates:
        all_best = []
        all_avg = []
        for seed in seeds:
            config = config_base.copy()
            config["crossover_rate"] = rate
            ga = GeneticAlgorithm(**config)
            _, best_fitness_values, avg_fitness_values = ga.evolve(seed)
            all_best.append(best_fitness_values)
            all_avg.append(avg_fitness_values)

        avg_best = np.mean(all_best, axis=0)
        avg_avg = np.mean(all_avg, axis=0)

        plt.plot(avg_best, label=f"Best Fit (CR={rate})")
        plt.plot(avg_avg, linestyle="--", label=f"Avg Fit (CR={rate})")

    plt.yscale("log")
    plt.title("Experiment 3: Crossover Rate Impact")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (log scale)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("experiment_3_crossover_plot.png")
    plt.show()

def run_experiment_4():
    print("\n--- Experiment 4: Mutation Impact ---")
    mutation_configs = [
        (0.1, 0.1),
        (0.2, 0.2),
        (0.4, 0.3),
        (0.5, 0.5),
    ]
    seeds = [23, 42, 99]
    config_base = {
        "population_size": 100,
        "crossover_rate": 0.6,
        "num_generations": 100
    }

    plt.figure(figsize=(10, 6))
    for mut_rate, mut_strength in mutation_configs:
        all_best = []
        all_avg = []
        for seed in seeds:
            config = config_base.copy()
            config["mutation_rate"] = mut_rate
            config["mutation_strength"] = mut_strength
            ga = GeneticAlgorithm(**config)
            _, best_fitness_values, avg_fitness_values = ga.evolve(seed)
            all_best.append(best_fitness_values)
            all_avg.append(avg_fitness_values)

        avg_best = np.mean(all_best, axis=0)
        avg_avg = np.mean(all_avg, axis=0)

        label = f"MR={mut_rate}, MS={mut_strength}"
        plt.plot(avg_best, label=f"Best ({label})")
        plt.plot(avg_avg, linestyle="--", label=f"Avg ({label})")

    plt.yscale("log")
    plt.title("Experiment 4: Mutation Rate and Strength Impact")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (log scale)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("experiment_4_mutation_plot.png")
    plt.show()

if __name__ == "__main__":
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()
