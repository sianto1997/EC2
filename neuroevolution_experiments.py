"""
Robot Neuroevolution Experiments
Assignment 2 - Evolutionary Computing

This module implements evolutionary algorithms to evolve neural network controllers
for the ARIEL Gecko robot. It includes baseline random controller, Evolution Strategies (ES),
and Genetic Algorithm (GA) implementations.
"""

import numpy as np
import mujoco
import matplotlib.pyplot as plt
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from tqdm import tqdm
import time

# ARIEL imports
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


@dataclass
class ExperimentConfig:
    """Configuration for evolutionary experiments."""
    algorithm: str
    population_size: int
    generations: int
    mutation_rate: float
    crossover_rate: float
    tournament_size: int = 3
    elitism: bool = True
    random_seed: int = 42


class NeuralNetworkController:
    """Neural network controller for the gecko robot."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, weights: Optional[np.ndarray] = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Calculate total number of weights needed
        self.w1_size = input_size * hidden_size
        self.b1_size = hidden_size
        self.w2_size = hidden_size * hidden_size
        self.b2_size = hidden_size
        self.w3_size = hidden_size * output_size
        self.b3_size = output_size
        self.total_weights = self.w1_size + self.b1_size + self.w2_size + self.b2_size + self.w3_size + self.b3_size

        if weights is None:
            self.weights = np.random.randn(self.total_weights) * 0.1
        else:
            assert len(weights) == self.total_weights, f"Expected {self.total_weights} weights, got {len(weights)}"
            self.weights = weights.copy()

    def _unpack_weights(self) -> Tuple[np.ndarray, ...]:
        """Unpack flat weight vector into matrices and biases."""
        idx = 0

        w1 = self.weights[idx:idx+self.w1_size].reshape(self.input_size, self.hidden_size)
        idx += self.w1_size

        b1 = self.weights[idx:idx+self.b1_size]
        idx += self.b1_size

        w2 = self.weights[idx:idx+self.w2_size].reshape(self.hidden_size, self.hidden_size)
        idx += self.w2_size

        b2 = self.weights[idx:idx+self.b2_size]
        idx += self.b2_size

        w3 = self.weights[idx:idx+self.w3_size].reshape(self.hidden_size, self.output_size)
        idx += self.w3_size

        b3 = self.weights[idx:idx+self.b3_size]

        return w1, b1, w2, b2, w3, b3

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the neural network."""
        w1, b1, w2, b2, w3, b3 = self._unpack_weights()

        # Layer 1
        h1 = np.tanh(np.dot(inputs, w1) + b1)

        # Layer 2
        h2 = np.tanh(np.dot(h1, w2) + b2)

        # Output layer - use tanh to get range [-1, 1], then scale to [-pi/2, pi/2]
        output = np.tanh(np.dot(h2, w3) + b3)

        return output * (np.pi / 2)  # Scale to [-pi/2, pi/2]


class RobotSimulator:
    """Handles robot simulation and fitness evaluation."""

    def __init__(self, simulation_time: float = 10.0, headless: bool = True):
        self.simulation_time = simulation_time
        self.headless = headless
        self.history = []

    def evaluate_controller(self, controller: NeuralNetworkController, seed: int = None) -> float:
        """Evaluate a controller and return its fitness."""
        if seed is not None:
            np.random.seed(seed)

        # Clear any existing control callback
        mujoco.set_mjcb_control(None)

        # Initialize world and robot
        world = SimpleFlatWorld()
        gecko_core = gecko()

        # Fix spawn point issue - use [0, 0, 0.1] to avoid clipping through ground
        world.spawn(gecko_core.spec, spawn_position=[0, 0, 0.1])

        model = world.spec.compile()
        data = mujoco.MjData(model)

        # Initialize data tracking
        geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
        to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

        # Reset history
        self.history = []

        def control_callback(model, data):
            """Control callback function."""
            # Get robot state (joint positions and velocities)
            inputs = np.concatenate([data.qpos, data.qvel])

            # Get control outputs from neural network
            outputs = controller.forward(inputs)

            # Ensure proper clipping to joint limits
            data.ctrl = np.clip(outputs, -np.pi/2, np.pi/2)

            # Track robot position
            if to_track:
                self.history.append(to_track[0].xpos.copy())

        # Set control callback
        mujoco.set_mjcb_control(control_callback)

        # Run simulation
        start_time = data.time
        while data.time - start_time < self.simulation_time:
            mujoco.mj_step(model, data)

        # Calculate fitness
        fitness = self._calculate_fitness()

        # Clear control callback
        mujoco.set_mjcb_control(None)

        return fitness

    def _calculate_fitness(self) -> float:
        """Calculate fitness based on robot movement."""
        if len(self.history) < 2:
            return 0.0

        positions = np.array(self.history)

        # Distance-based fitness: total distance traveled
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        total_distance = np.sum(distances)

        # Forward movement bonus (movement in positive x direction)
        forward_distance = positions[-1, 0] - positions[0, 0]

        # Stability penalty (penalize excessive z-axis movement)
        z_variation = np.std(positions[:, 2])
        stability_penalty = z_variation * 2.0

        # Combined fitness
        fitness = total_distance + forward_distance * 2.0 - stability_penalty

        return max(fitness, 0.0)  # Ensure non-negative fitness


class RandomBaselineController:
    """Baseline random controller for comparison."""

    def __init__(self, num_joints: int):
        self.num_joints = num_joints

    def get_actions(self) -> np.ndarray:
        """Get random actions."""
        return np.random.uniform(-np.pi/2, np.pi/2, self.num_joints)


class EvolutionStrategies:
    """Evolution Strategies (ES) implementation."""

    def __init__(self, config: ExperimentConfig, input_size: int, hidden_size: int, output_size: int):
        self.config = config
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Calculate genome size
        controller = NeuralNetworkController(input_size, hidden_size, output_size)
        self.genome_size = controller.total_weights

        # Initialize population
        self.population = []
        self.fitness_scores = []

        np.random.seed(config.random_seed)
        for _ in range(config.population_size):
            individual = np.random.randn(self.genome_size) * 0.1
            self.population.append(individual)

    def evolve_generation(self, simulator: RobotSimulator) -> Tuple[float, float, np.ndarray]:
        """Evolve one generation and return best fitness, average fitness, and best individual."""

        # Evaluate population
        self.fitness_scores = []
        for individual in tqdm(self.population, desc="Evaluating population"):
            controller = NeuralNetworkController(self.input_size, self.hidden_size, self.output_size, individual)
            fitness = simulator.evaluate_controller(controller)
            self.fitness_scores.append(fitness)

        # Get statistics
        best_fitness = max(self.fitness_scores)
        avg_fitness = np.mean(self.fitness_scores)
        best_individual = self.population[np.argmax(self.fitness_scores)].copy()

        # Selection and reproduction
        new_population = []

        # Elitism - keep best individual
        if self.config.elitism:
            new_population.append(best_individual.copy())

        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Select parent (tournament selection)
            parent = self._tournament_selection()

            # Create offspring with mutation
            offspring = parent.copy()
            mutation_mask = np.random.random(len(offspring)) < self.config.mutation_rate
            offspring[mutation_mask] += np.random.normal(0, 0.1, np.sum(mutation_mask))

            new_population.append(offspring)

        self.population = new_population
        return best_fitness, avg_fitness, best_individual

    def _tournament_selection(self) -> np.ndarray:
        """Tournament selection."""
        tournament_indices = np.random.choice(len(self.population), self.config.tournament_size, replace=False)
        tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx].copy()


class GeneticAlgorithm:
    """Genetic Algorithm implementation."""

    def __init__(self, config: ExperimentConfig, input_size: int, hidden_size: int, output_size: int):
        self.config = config
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Calculate genome size
        controller = NeuralNetworkController(input_size, hidden_size, output_size)
        self.genome_size = controller.total_weights

        # Initialize population
        self.population = []
        self.fitness_scores = []

        np.random.seed(config.random_seed)
        for _ in range(config.population_size):
            individual = np.random.randn(self.genome_size) * 0.1
            self.population.append(individual)

    def evolve_generation(self, simulator: RobotSimulator) -> Tuple[float, float, np.ndarray]:
        """Evolve one generation and return best fitness, average fitness, and best individual."""

        # Evaluate population
        self.fitness_scores = []
        for individual in tqdm(self.population, desc="Evaluating population"):
            controller = NeuralNetworkController(self.input_size, self.hidden_size, self.output_size, individual)
            fitness = simulator.evaluate_controller(controller)
            self.fitness_scores.append(fitness)

        # Get statistics
        best_fitness = max(self.fitness_scores)
        avg_fitness = np.mean(self.fitness_scores)
        best_individual = self.population[np.argmax(self.fitness_scores)].copy()

        # Create new population
        new_population = []

        # Elitism - keep best individual
        if self.config.elitism:
            new_population.append(best_individual.copy())

        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Select two parents
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            if np.random.random() < self.config.crossover_rate:
                offspring1, offspring2 = self._crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()

            # Mutation
            offspring1 = self._mutate(offspring1)
            offspring2 = self._mutate(offspring2)

            new_population.extend([offspring1, offspring2])

        # Trim to exact population size
        self.population = new_population[:self.config.population_size]
        return best_fitness, avg_fitness, best_individual

    def _tournament_selection(self) -> np.ndarray:
        """Tournament selection."""
        tournament_indices = np.random.choice(len(self.population), self.config.tournament_size, replace=False)
        tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single-point crossover."""
        crossover_point = np.random.randint(1, len(parent1))
        offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return offspring1, offspring2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Gaussian mutation."""
        mutated = individual.copy()
        mutation_mask = np.random.random(len(individual)) < self.config.mutation_rate
        mutated[mutation_mask] += np.random.normal(0, 0.1, np.sum(mutation_mask))
        return mutated


class ExperimentRunner:
    """Manages and runs evolutionary experiments."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def run_experiment(self, algorithm_class, config: ExperimentConfig, experiment_name: str,
                      repetition: int) -> Dict:
        """Run a single experiment."""

        print(f"Running {experiment_name} (repetition {repetition+1})")

        # Initialize components
        input_size = 29  # 15 joint positions + 14 velocities for gecko
        hidden_size = 16
        output_size = 8  # 8 joint actuators for gecko

        simulator = RobotSimulator(simulation_time=15.0)
        algorithm = algorithm_class(config, input_size, hidden_size, output_size)

        # Track results
        best_fitness_history = []
        avg_fitness_history = []
        generation_times = []

        print(f"Starting evolution with {config.population_size} individuals for {config.generations} generations")

        best_individual_ever = None
        best_fitness_ever = -float('inf')

        # Evolution loop
        for generation in range(config.generations):
            start_time = time.time()

            best_fitness, avg_fitness, best_individual = algorithm.evolve_generation(simulator)

            generation_time = time.time() - start_time
            generation_times.append(generation_time)

            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            if best_fitness > best_fitness_ever:
                best_fitness_ever = best_fitness
                best_individual_ever = best_individual.copy()

            print(f"Generation {generation+1}: Best={best_fitness:.3f}, Avg={avg_fitness:.3f}, Time={generation_time:.1f}s")

        # Save results
        results = {
            'experiment_name': experiment_name,
            'repetition': repetition,
            'config': config.__dict__,
            'best_fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history,
            'generation_times': generation_times,
            'best_individual': best_individual_ever,
            'best_fitness': best_fitness_ever,
            'total_time': sum(generation_times)
        }

        # Save individual experiment results
        results_file = self.results_dir / f"{experiment_name}_rep{repetition}_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

        # Save best controller
        controller_file = self.results_dir / f"{experiment_name}_rep{repetition}_best_controller.pkl"
        best_controller = NeuralNetworkController(input_size, hidden_size, output_size, best_individual_ever)
        with open(controller_file, 'wb') as f:
            pickle.dump(best_controller, f)

        print(f"Experiment completed. Best fitness: {best_fitness_ever:.3f}")
        print(f"Results saved to {results_file}")

        return results

    def run_baseline_experiment(self, repetition: int, num_evaluations: int = 50) -> Dict:
        """Run baseline random controller experiment."""

        print(f"Running Random Baseline (repetition {repetition+1})")

        simulator = RobotSimulator(simulation_time=15.0)
        fitness_scores = []

        # Evaluate random controllers
        for i in tqdm(range(num_evaluations), desc="Evaluating random controllers"):
            # Create random controller
            input_size = 29
            hidden_size = 16
            output_size = 8
            controller = NeuralNetworkController(input_size, hidden_size, output_size)

            fitness = simulator.evaluate_controller(controller, seed=i)
            fitness_scores.append(fitness)

        # Calculate statistics
        avg_fitness = np.mean(fitness_scores)
        std_fitness = np.std(fitness_scores)
        max_fitness = np.max(fitness_scores)

        results = {
            'experiment_name': 'Random_Baseline',
            'repetition': repetition,
            'fitness_scores': fitness_scores,
            'avg_fitness': avg_fitness,
            'std_fitness': std_fitness,
            'max_fitness': max_fitness,
            'num_evaluations': num_evaluations
        }

        # Save results
        results_file = self.results_dir / f"Random_Baseline_rep{repetition}_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

        print(f"Random baseline completed. Avg fitness: {avg_fitness:.3f} ± {std_fitness:.3f}")

        return results


def create_experiment_configs() -> Dict[str, ExperimentConfig]:
    """Create experiment configurations."""

    configs = {}

    # Evolution Strategies
    configs['ES'] = ExperimentConfig(
        algorithm='Evolution Strategies',
        population_size=50,
        generations=100,
        mutation_rate=0.1,
        crossover_rate=0.0,  # ES doesn't use crossover
        tournament_size=3,
        elitism=True,
        random_seed=42
    )

    # Genetic Algorithm
    configs['GA'] = ExperimentConfig(
        algorithm='Genetic Algorithm',
        population_size=50,
        generations=100,
        mutation_rate=0.05,
        crossover_rate=0.8,
        tournament_size=3,
        elitism=True,
        random_seed=42
    )

    return configs


def plot_results(results_dir: str = "results"): # NOT SURE IF IT WORKS
    """Generate plots from experimental results."""

    results_path = Path(results_dir)

    # Load all results
    all_results = {}

    for results_file in results_path.glob("*_results.pkl"):
        with open(results_file, 'rb') as f:
            result = pickle.load(f)

        exp_name = result['experiment_name']
        if exp_name not in all_results:
            all_results[exp_name] = []
        all_results[exp_name].append(result)

    # Create plots
    plt.figure(figsize=(15, 5))

    # Plot 1: Evolution curves with std shading
    plt.subplot(1, 3, 1)

    for exp_name, results_list in all_results.items():
        if exp_name == 'Random_Baseline':
            # Plot baseline as horizontal line
            baseline_fitness = np.mean([r['avg_fitness'] for r in results_list])
            baseline_std = np.std([r['avg_fitness'] for r in results_list])

            plt.axhline(y=baseline_fitness, color='red', linestyle='--',
                       label=f'Random Baseline ({baseline_fitness:.2f}±{baseline_std:.2f})')
            plt.axhspan(baseline_fitness - baseline_std, baseline_fitness + baseline_std,
                       alpha=0.2, color='red')
        else:
            # Plot evolution curves
            if results_list:
                # Extract fitness histories
                fitness_histories = [r['best_fitness_history'] for r in results_list]

                # Ensure all have same length
                min_length = min(len(h) for h in fitness_histories)
                fitness_histories = [h[:min_length] for h in fitness_histories]

                # Calculate mean and std
                fitness_array = np.array(fitness_histories)
                mean_fitness = np.mean(fitness_array, axis=0)
                std_fitness = np.std(fitness_array, axis=0)

                generations = range(len(mean_fitness))

                plt.plot(generations, mean_fitness, label=f'{exp_name} (mean)')
                plt.fill_between(generations, mean_fitness - std_fitness,
                               mean_fitness + std_fitness, alpha=0.3)

    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Final performance comparison
    plt.subplot(1, 3, 2)

    exp_names = []
    final_fitness = []
    final_std = []

    for exp_name, results_list in all_results.items():
        if exp_name == 'Random_Baseline':
            fitness_values = [r['avg_fitness'] for r in results_list]
        else:
            fitness_values = [r['best_fitness'] for r in results_list]

        exp_names.append(exp_name)
        final_fitness.append(np.mean(fitness_values))
        final_std.append(np.std(fitness_values))

    bars = plt.bar(exp_names, final_fitness, yerr=final_std, capsize=5)
    plt.ylabel('Final Fitness')
    plt.title('Final Performance Comparison')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, fitness, std in zip(bars, final_fitness, final_std):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                f'{fitness:.2f}±{std:.2f}', ha='center', va='bottom')

    # Plot 3: Convergence analysis
    plt.subplot(1, 3, 3)

    for exp_name, results_list in all_results.items():
        if exp_name != 'Random_Baseline' and results_list:
            # Calculate generations to reach certain fitness thresholds
            target_fitness = 0.8 * max([r['best_fitness'] for r in results_list])

            convergence_gens = []
            for result in results_list:
                fitness_hist = result['best_fitness_history']
                converged_gen = next((i for i, f in enumerate(fitness_hist) if f >= target_fitness), len(fitness_hist))
                convergence_gens.append(converged_gen)

            plt.hist(convergence_gens, alpha=0.7, label=f'{exp_name}', bins=10)

    plt.xlabel('Generations to Convergence')
    plt.ylabel('Frequency')
    plt.title('Convergence Analysis')
    plt.legend()

    plt.tight_layout()
    plt.savefig(results_path / 'experiment_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    for exp_name, results_list in all_results.items():
        print(f"\n{exp_name}:")
        print(f"  Number of repetitions: {len(results_list)}")

        if exp_name == 'Random_Baseline':
            fitness_values = [r['avg_fitness'] for r in results_list]
            print(f"  Average fitness: {np.mean(fitness_values):.3f} ± {np.std(fitness_values):.3f}")
        else:
            fitness_values = [r['best_fitness'] for r in results_list]
            print(f"  Best fitness: {np.mean(fitness_values):.3f} ± {np.std(fitness_values):.3f}")

            total_times = [r['total_time'] for r in results_list]
            print(f"  Average runtime: {np.mean(total_times):.1f} ± {np.std(total_times):.1f} seconds")


def visualize_best_controller(controller_file: str):
    """Visualize a trained controller in interactive mode.

    Args:
        controller_file: Path to saved controller pickle file
    """
    import pickle
    from mujoco import viewer

    print(f"Loading and visualizing controller from: {controller_file}")

    # Load best controller
    try:
        with open(controller_file, 'rb') as f:
            controller = pickle.load(f)
        print(f"✅ Controller loaded successfully")
    except FileNotFoundError:
        print(f"❌ Controller file not found: {controller_file}")
        print("Run experiments first to generate controllers!")
        return

    # Set up simulation with VIEWER
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0.1])

    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Track robot for fitness calculation
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    history = []

    def evolved_controller(model, data):
        """Control callback using evolved neural network."""
        inputs = np.concatenate([data.qpos, data.qvel])
        outputs = controller.forward(inputs)
        data.ctrl = np.clip(outputs, -np.pi/2, np.pi/2)

        # Track movement
        if to_track:
            history.append(to_track[0].xpos.copy())

    mujoco.set_mjcb_control(evolved_controller)

    print("\n🎮 CONTROLLER VISUALIZATION")
    print("="*50)
    print("🤖 Watch your evolved robot in action!")
    print("📊 Close the viewer window when done watching")
    print("⚡ Controller will run indefinitely until you close the window")
    print("="*50)

    # Launch viewer - ENOJAY DA VIEW! xD
    try:
        viewer.launch(model=model, data=data)
    except KeyboardInterrupt:
        print("Visualization interrupted by user")
    finally:
        # Calculate and show final fitness
        if len(history) > 1:
            positions = np.array(history)
            distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
            total_distance = np.sum(distances)
            forward_distance = positions[-1, 0] - positions[0, 0]
            print(f"\n📈 Performance metrics:")
            print(f"   Total distance traveled: {total_distance:.3f}")
            print(f"   Forward movement: {forward_distance:.3f}")
            print(f"   Final position: {positions[-1]}")


def compare_controllers_visually():
    """Compare random vs best evolved controllers visually."""
    from pathlib import Path
    from mujoco import viewer

    results_dir = Path("results")

    print("\n🔍 CONTROLLER COMPARISON")
    print("="*50)

    # Check for available controllers
    controller_files = list(results_dir.glob("*_best_controller.pkl"))

    if not controller_files:
        print("❌ No trained controllers found!")
        print("Run experiments first: python neuroevolution_experiments.py")
        return

    print(f"Found {len(controller_files)} trained controllers:")
    for i, file in enumerate(controller_files):
        print(f"  {i+1}. {file.name}")

    print(f"\n0. Random controller (baseline)")

    while True:
        try:
            choice = input(f"\nSelect controller to visualize (0-{len(controller_files)}, or 'q' to quit): ").strip()

            if choice.lower() == 'q':
                break

            choice = int(choice)

            if choice == 0:
                # Show random controller
                print("\n🎲 Showing random controller...")
                simulator = RobotSimulator(simulation_time=30.0)
                random_controller = NeuralNetworkController(29, 16, 8)

                # Set up visual simulation for random controller
                mujoco.set_mjcb_control(None)
                world = SimpleFlatWorld()
                gecko_core = gecko()
                world.spawn(gecko_core.spec, spawn_position=[0, 0, 0.1])

                model = world.spec.compile()
                data = mujoco.MjData(model)

                def random_control(model, data):
                    inputs = np.concatenate([data.qpos, data.qvel])
                    outputs = random_controller.forward(inputs)
                    data.ctrl = np.clip(outputs, -np.pi/2, np.pi/2)

                mujoco.set_mjcb_control(random_control)
                viewer.launch(model=model, data=data)

            elif 1 <= choice <= len(controller_files):
                controller_file = controller_files[choice - 1]
                visualize_best_controller(str(controller_file))
            else:
                print(f"❌ Invalid choice. Please enter 0-{len(controller_files)}")

        except ValueError:
            print("❌ Invalid input. Please enter a number or 'q'")
        except KeyboardInterrupt:
            print("\nVisualization interrupted")
            break


if __name__ == "__main__":
    import sys

    # Check for visualization mode
    if len(sys.argv) > 1:
        if sys.argv[1] == "--visualize":
            print("🎮 ROBOT VISUALIZATION MODE")
            print("="*50)
            compare_controllers_visually()
            sys.exit(0)
        elif sys.argv[1] == "--help":
            print("Robot Neuroevolution - Usage:")
            print("  python neuroevolution_experiments.py          # Run full experiments")
            print("  python neuroevolution_experiments.py --visualize  # Visualize controllers")
            print("  python neuroevolution_experiments.py --help   # Show this help")
            sys.exit(0)

    # Run complete experimental setup
    print("🤖 Starting Robot Neuroevolution Experiments")
    print("="*50)
    print("⚡ Running in HEADLESS mode for performance")
    print("💡 To see robot movement, use: python neuroevolution_experiments.py --visualize")
    print("="*50)

    # Create experiment runner
    runner = ExperimentRunner()

    # Get experiment configurations
    configs = create_experiment_configs()

    # Number of repetitions per experiment
    num_repetitions = 3

    # Run baseline experiments
    print("\n1. Running Random Baseline Experiments")
    for rep in range(num_repetitions):
        runner.run_baseline_experiment(rep)

    # Run evolutionary algorithm experiments
    algorithms = {
        'ES': EvolutionStrategies,
        'GA': GeneticAlgorithm
    }

    for alg_name, alg_class in algorithms.items():
        print(f"\n2. Running {alg_name} Experiments")
        config = configs[alg_name]

        for rep in range(num_repetitions):
            # Use different random seed for each repetition
            config.random_seed = 42 + rep
            runner.run_experiment(alg_class, config, alg_name, rep)

    print("\n" + "="*50)
    print("🎉 All experiments completed!")

    # Generate plots
    print("\n📊 Generating plots...")
    plot_results()

    print(f"\n📁 Results saved in: {runner.results_dir}")
    print("💾 Best controllers saved as .pkl files for each experiment")
    print("\n🎮 TO SEE YOUR ROBOTS IN ACTION:")
    print("   python neuroevolution_experiments.py --visualize")
    print("\n📈 Check the generated plots to see evolution progress!")
