"""
Robot Neuroevolution using DEAP library
Assignment 2 - Evolutionary Computing

This version uses the DEAP (Distributed Evolutionary Algorithms in Python) library
"""

import numpy as np
import mujoco
import matplotlib.pyplot as plt
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
import time
import random

# DEAP imports
from deap import base, creator, tools, algorithms

# ARIEL imports
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# Import our neural network controller
from neuroevolution_experiments import NeuralNetworkController, RobotSimulator


class DEAPEvolutionaryAlgorithm:
    """DEAP-based evolutionary algorithm for neural network evolution."""

    def __init__(self, algorithm_type: str, pop_size: int, input_size: int,
                 hidden_size: int, output_size: int, random_seed: int = 42):

        self.algorithm_type = algorithm_type
        self.pop_size = pop_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Calculate genome size
        self.controller = NeuralNetworkController(input_size, hidden_size, output_size)
        self.genome_size = self.controller.total_weights

        # Setup DEAP
        self._setup_deap()

    def _setup_deap(self):
        """Setup DEAP components."""

        # Create fitness class (maximization)
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # Create individual class
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        # Initialize toolbox
        self.toolbox = base.Toolbox()

        # Register functions
        self.toolbox.register("attr_float", random.gauss, mu=0, sigma=0.1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                             self.toolbox.attr_float, n=self.genome_size)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register genetic operators based on algorithm type
        if self.algorithm_type == "GA":
            self.toolbox.register("mate", tools.cxBlend, alpha=0.2)
            self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
            self.toolbox.register("select", tools.selTournament, tournsize=3)

        elif self.algorithm_type == "ES":
            # Evolution strategies - no crossover, only mutation
            self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=1.0)
            self.toolbox.register("select", tools.selBest)

        elif self.algorithm_type == "CMA":
            # We'll use (μ+λ)-ES selection for CMA-ES approximation
            self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=1.0)
            self.toolbox.register("select", tools.selBest)

    def evaluate_individual(self, individual, simulator: RobotSimulator) -> Tuple[float,]:
        """Evaluate a single individual."""
        try:
            # Ensure individual is proper numpy array
            individual_array = np.array(individual, dtype=float)

            if len(individual_array) != self.genome_size:
                print(f"Warning: Individual size mismatch: {len(individual_array)} vs {self.genome_size}")
                return (0.0,)

            controller = NeuralNetworkController(self.input_size, self.hidden_size,
                                               self.output_size, individual_array)
            fitness = simulator.evaluate_controller(controller)

            # Fitness tripping bruh
            if not isinstance(fitness, (int, float)) or np.isnan(fitness) or np.isinf(fitness):
                print(f"Warning: Invalid fitness value: {fitness}")
                return (0.0,)

            return (float(fitness),)  # DEAP expects tuple

        except Exception as e:
            print(f"Error in evaluate_individual: {e}")
            return (0.0,)

    def run_evolution(self, simulator: RobotSimulator, generations: int,
                     cxpb: float = 0.8, mutpb: float = 0.1) -> Tuple[List[float], List[float], np.ndarray]:
        """Run the evolutionary algorithm."""

        # Create initial population
        population = self.toolbox.population(n=self.pop_size)

        # Statistics tracking
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)
        stats.register("std", np.std)

        # Hall of fame to track best individual
        hof = tools.HallOfFame(1)

        # Set evaluation function
        self.toolbox.register("evaluate", self.evaluate_individual, simulator=simulator)

        # Track evolution
        best_fitness_history = []
        avg_fitness_history = []

        print(f"Running {self.algorithm_type} with {self.pop_size} individuals for {generations} generations")

        # Evaluate initial population
        print(f"Evaluating initial population of {len(population)} individuals...")
        fitnesses = []
        for i, ind in enumerate(population):
            try:
                fit = self.toolbox.evaluate(ind)
                fitnesses.append(fit)
                ind.fitness.values = fit
            except Exception as e:
                print(f"Warning: Individual {i} evaluation failed: {e}")
                # Assign very low fitness to failed individuals
                ind.fitness.values = (0.0,)
                fitnesses.append((0.0,))

        # Evolution loop
        for generation in range(generations):
            print(f"Generation {generation + 1}/{generations}")

            if self.algorithm_type == "GA":
                # Standard genetic algorithm
                offspring = algorithms.varAnd(population, self.toolbox, cxpb=cxpb, mutpb=mutpb)

                # Evaluate offspring that don't have valid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                for ind in invalid_ind:
                    try:
                        fit = self.toolbox.evaluate(ind)
                        ind.fitness.values = fit
                    except Exception as e:
                        print(f"Warning: Offspring evaluation failed: {e}")
                        ind.fitness.values = (0.0,)

                # Select next generation
                population = self.toolbox.select(offspring + population, self.pop_size)

            elif self.algorithm_type in ["ES", "CMA"]:
                # Evolution strategies
                # Create offspring through mutation only
                offspring = [self.toolbox.clone(ind) for ind in population]

                # Mutate all offspring
                for mutant in offspring:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values  # Invalidate fitness

                # Evaluate offspring
                for ind in offspring:
                    try:
                        fit = self.toolbox.evaluate(ind)
                        ind.fitness.values = fit
                    except Exception as e:
                        print(f"Warning: ES offspring evaluation failed: {e}")
                        ind.fitness.values = (0.0,)

                # (μ+λ) selection: select best from parents + offspring
                population = self.toolbox.select(population + offspring, self.pop_size)

            # Update statistics - ensure all fitness values are valid
            valid_population = [ind for ind in population if ind.fitness.valid and len(ind.fitness.values) > 0]

            if not valid_population:
                print(f"Warning: No valid individuals in generation {generation + 1}")
                best_fitness_history.append(0.0)
                avg_fitness_history.append(0.0)
                record = {'max': 0.0, 'avg': 0.0, 'std': 0.0}
            else:
                record = stats.compile(valid_population)
                best_fitness_history.append(record['max'])
                avg_fitness_history.append(record['avg'])

            # Update hall of fame
            hof.update(population)

            print(f"  Best: {record['max']:.3f}, Avg: {record['avg']:.3f}, Std: {record['std']:.3f}")

        return best_fitness_history, avg_fitness_history, np.array(hof[0])


def run_deap_experiments():
    """Run experiments using DEAP library."""

    print("=" * 60)
    print("DEAP-BASED NEUROEVOLUTION EXPERIMENTS")
    print("=" * 60)

    # Create results directory
    results_dir = Path("results_deap")
    results_dir.mkdir(exist_ok=True)

    # Robot simulator
    simulator = RobotSimulator(simulation_time=10.0)

    # Experiment configurations
    experiments = {
        "DEAP_GA": {"algorithm_type": "GA", "generations": 50, "pop_size": 30},
        "DEAP_ES": {"algorithm_type": "ES", "generations": 50, "pop_size": 30},
        "DEAP_CMA": {"algorithm_type": "CMA", "generations": 50, "pop_size": 20}
    }

    # Network dimensions
    input_size = 29  # 15 joint positions + 14 velocities
    hidden_size = 16
    output_size = 8  # 8 actuators

    num_repetitions = 3
    all_results = {}

    for exp_name, config in experiments.items():
        print(f"\n{'=' * 40}")
        print(f"Running {exp_name}")
        print(f"{'=' * 40}")

        experiment_results = []

        for rep in range(num_repetitions):
            print(f"\nRepetition {rep + 1}/{num_repetitions}")

            # Create algorithm
            alg = DEAPEvolutionaryAlgorithm(
                algorithm_type=config["algorithm_type"],
                pop_size=config["pop_size"],
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                random_seed=42 + rep
            )

            # Run evolution
            start_time = time.time()
            best_history, avg_history, best_individual = alg.run_evolution(
                simulator, config["generations"]
            )
            total_time = time.time() - start_time

            # Store results
            result = {
                'experiment_name': exp_name,
                'repetition': rep,
                'best_fitness_history': best_history,
                'avg_fitness_history': avg_history,
                'best_individual': best_individual,
                'best_fitness': max(best_history),
                'total_time': total_time,
                'config': config
            }

            experiment_results.append(result)

            # Save individual result
            result_file = results_dir / f"{exp_name}_rep{rep}_results.pkl"
            with open(result_file, 'wb') as f:
                pickle.dump(result, f)

            # Save best controller
            controller_file = results_dir / f"{exp_name}_rep{rep}_best_controller.pkl"
            best_controller = NeuralNetworkController(input_size, hidden_size, output_size, best_individual)
            with open(controller_file, 'wb') as f:
                pickle.dump(best_controller, f)

            print(f"Final best fitness: {max(best_history):.3f}")
            print(f"Time: {total_time:.1f} seconds")

        all_results[exp_name] = experiment_results

    # Generate summary plots
    plot_deap_results(all_results, results_dir)

    print(f"\n{'=' * 60}")
    print("DEAP EXPERIMENTS COMPLETED!")
    print(f"Results saved in: {results_dir}")
    print(f"{'=' * 60}")


def plot_deap_results(all_results: Dict, results_dir: Path):
    """Plot results from DEAP experiments."""

    plt.figure(figsize=(15, 5))

    # Plot 1: Evolution curves
    plt.subplot(1, 3, 1)
    colors = ['blue', 'red', 'green', 'orange']

    for i, (exp_name, results) in enumerate(all_results.items()):
        # Get all fitness histories
        fitness_histories = [r['best_fitness_history'] for r in results]

        # Ensure same length
        min_length = min(len(h) for h in fitness_histories)
        fitness_histories = [h[:min_length] for h in fitness_histories]

        # Calculate mean and std
        fitness_array = np.array(fitness_histories)
        mean_fitness = np.mean(fitness_array, axis=0)
        std_fitness = np.std(fitness_array, axis=0)

        generations = range(len(mean_fitness))
        color = colors[i % len(colors)]

        plt.plot(generations, mean_fitness, color=color, label=f'{exp_name}', linewidth=2)
        plt.fill_between(generations, mean_fitness - std_fitness,
                        mean_fitness + std_fitness, color=color, alpha=0.3)

    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Evolution Progress (DEAP)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Final performance comparison
    plt.subplot(1, 3, 2)
    exp_names = []
    final_fitness = []
    final_std = []

    for exp_name, results in all_results.items():
        fitness_values = [r['best_fitness'] for r in results]
        exp_names.append(exp_name.replace('DEAP_', ''))
        final_fitness.append(np.mean(fitness_values))
        final_std.append(np.std(fitness_values))

    bars = plt.bar(exp_names, final_fitness, yerr=final_std, capsize=5,
                   color=['lightblue', 'lightcoral', 'lightgreen'])
    plt.ylabel('Final Best Fitness')
    plt.title('Final Performance (DEAP)')
    plt.xticks(rotation=45)

    # Add value labels
    for bar, fitness, std in zip(bars, final_fitness, final_std):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{fitness:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10)

    # Plot 3: Computational efficiency
    plt.subplot(1, 3, 3)

    times = []
    for exp_name, results in all_results.items():
        avg_time = np.mean([r['total_time'] for r in results])
        times.append(avg_time)

    bars = plt.bar([name.replace('DEAP_', '') for name in all_results.keys()], times,
                   color=['lightblue', 'lightcoral', 'lightgreen'])
    plt.ylabel('Average Runtime (seconds)')
    plt.title('Computational Efficiency')
    plt.xticks(rotation=45)

    # Add time labels
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{time_val:.1f}s', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(results_dir / 'deap_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    print("\n" + "=" * 60)
    print("DEAP EXPERIMENTAL SUMMARY")
    print("=" * 60)

    for exp_name, results in all_results.items():
        print(f"\n{exp_name}:")
        fitness_values = [r['best_fitness'] for r in results]
        times = [r['total_time'] for r in results]

        print(f"  Best fitness: {np.mean(fitness_values):.3f} ± {np.std(fitness_values):.3f}")
        print(f"  Runtime: {np.mean(times):.1f} ± {np.std(times):.1f} seconds")
        print(f"  Generations: {len(results[0]['best_fitness_history'])}")
        print(f"  Population size: {results[0]['config']['pop_size']}")


if __name__ == "__main__":
    run_deap_experiments()
