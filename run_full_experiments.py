"""
Run full experiments for Assignment 2 - Robot Neuroevolution
This script runs the complete experimental suite as required by the assignment.
"""

import sys
sys.path.append('../ariel')

from neuroevolution_experiments import *
import time

def main():
    """Run the complete experimental suite."""

    print("="*80)
    print("ROBOT NEUROEVOLUTION - FULL EXPERIMENTS")
    print("Assignment 2 - Evolutionary Computing")
    print("="*80)

    start_time = time.time()

    # Create experiment runner
    runner = ExperimentRunner(results_dir="results_final")

    # Get experiment configurations
    configs = create_experiment_configs()

    # Number of repetitions per experiment (as required by assignment)
    num_repetitions = 3

    print(f"\nExperiment Configuration:")
    print(f"- Algorithms: Evolution Strategies (ES), Genetic Algorithm (GA)")
    print(f"- Baseline: Random Controller")
    print(f"- Repetitions per experiment: {num_repetitions}")
    print(f"- Total runs: {len(configs) + 1} experiments × {num_repetitions} = {(len(configs) + 1) * num_repetitions} runs")
    print(f"- Neural Network: 29 → 16 → 8 (input → hidden → output)")
    print(f"- Gecko Robot: 8 actuators, 15 joints, simulation time = 15s per evaluation")

    # Run baseline experiments
    print(f"\n{'='*50}")
    print("1. BASELINE EXPERIMENTS (Random Controller)")
    print(f"{'='*50}")

    for rep in range(num_repetitions):
        print(f"\nBaseline Repetition {rep+1}/{num_repetitions}")
        runner.run_baseline_experiment(rep, num_evaluations=50)

    # Run evolutionary algorithm experiments
    algorithms = {
        'ES': EvolutionStrategies,
        'GA': GeneticAlgorithm
    }

    for alg_name, alg_class in algorithms.items():
        print(f"\n{'='*50}")
        print(f"2. {alg_name.upper()} EXPERIMENTS")
        print(f"{'='*50}")

        config = configs[alg_name]
        print(f"Configuration:")
        print(f"- Population size: {config.population_size}")
        print(f"- Generations: {config.generations}")
        print(f"- Mutation rate: {config.mutation_rate}")
        if hasattr(config, 'crossover_rate') and config.crossover_rate > 0:
            print(f"- Crossover rate: {config.crossover_rate}")
        print(f"- Tournament size: {config.tournament_size}")
        print(f"- Elitism: {config.elitism}")

        for rep in range(num_repetitions):
            print(f"\n{alg_name} Repetition {rep+1}/{num_repetitions}")

            # Use different random seed for each repetition
            config.random_seed = 42 + rep
            runner.run_experiment(alg_class, config, alg_name, rep)

    # Generate comprehensive results
    print(f"\n{'='*50}")
    print("3. RESULTS ANALYSIS")
    print(f"{'='*50}")

    print("\nGenerating plots and analysis...")
    plot_results("results_final")

    total_time = time.time() - start_time

    print(f"\n{'='*80}")
    print("EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"Total runtime: {total_time/60:.1f} minutes")
    print(f"Results saved in: results_final/")
    print(f"Best controllers saved as .pkl files for reproduction")
    print()
    print("Files generated:")
    print("- *_results.pkl: Complete experimental data")
    print("- *_best_controller.pkl: Best evolved controllers")
    print("- experiment_results.png: Comprehensive fitness plots")
    print()
    print("The results show fitness evolution curves with mean ± standard deviation")
    print("shading as required by the assignment, comparing:")
    print("- Random Baseline")
    print("- Evolution Strategies")
    print("- Genetic Algorithm")
    print()
    print("Ready for report writing! 📊🎯")

if __name__ == "__main__":
    main()