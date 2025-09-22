"""
Quick experiment with reduced parameters to test everything works
"""

import sys
sys.path.append('../ariel')

from neuroevolution_experiments import *

def quick_experiment():
    """Run a quick experiment with small parameters."""

    print("Running quick experiment with small parameters...")

    # Create experiment runner
    runner = ExperimentRunner(results_dir="results_quick")

    configs = {
        'ES_quick': ExperimentConfig(
            algorithm='Evolution Strategies',
            population_size=10,
            generations=5,
            mutation_rate=0.1,
            crossover_rate=0.0,
            random_seed=42
        ),
        'GA_quick': ExperimentConfig(
            algorithm='Genetic Algorithm',
            population_size=10,
            generations=5,
            mutation_rate=0.1,
            crossover_rate=0.8,
            random_seed=42
        )
    }

    # Run baseline
    print("\n1. Running baseline...")
    runner.run_baseline_experiment(0, num_evaluations=10)

    # Run algorithms
    algorithms = {
        'ES_quick': EvolutionStrategies,
        'GA_quick': GeneticAlgorithm
    }

    for alg_name, alg_class in algorithms.items():
        print(f"\n2. Running {alg_name}...")
        config = configs[alg_name]
        runner.run_experiment(alg_class, config, alg_name, 0)

    print("\nGenerating plots...")
    plot_results("results_quick")

    print("Quick experiment completed! ✅")

if __name__ == "__main__":
    quick_experiment()