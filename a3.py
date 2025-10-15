"""
Robot Olympics - Assignment 3
Evolutionary Computing

Co-evolution of robot morphology (NDE) and controller (NN weights)
for Olympic Arena racing task.
"""

import argparse
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Tuple
from multiprocessing import Pool
from rich.progress import track


import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt

from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

if TYPE_CHECKING:
    from networkx import DiGraph

SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5] 
SIMULATION_DURATION = 15 # Should we change this?
GENOTYPE_SIZE = 64
HIDDEN_SIZE = 8

GLOBAL_BEST_FITNESS = -1000.0

def race_fitness_function(history: List[List[float]], simulation_time: float) -> float:
    """
    Fitness rewarding forward progress toward finish line at Y=0,
    with exponential bonus for stronger progress.
    """
    if len(history) == 0:
        return -1000.0

    # Final position
    x_final, y_final, z_final = history[-1]
    y_start = SPAWN_POS[1]

    # Forward distance: positive if moving toward finish (Y decreases)
    y_distance = y_start - y_final
    if y_distance <= 0:
        return -1000.0  # punish moving backwards (increasing Y)

    # Exponential progress reward
    progress_bonus = (y_distance ** 1.2) * 20  

    # Base fitness
    fitness = progress_bonus

    # Huge bonus if robot crosses finish line (Y <= 0)
    if y_final <= 0.0:
        fitness += 1000.0

    return fitness



def show_xpos_history(history: List[List[float]], save_path: Path, data_dir: Path) -> None:
    """Plot robot path on Olympic Arena background (from A3 template)."""
    if history is None or len(history) == 0:
        console.log("[yellow]No path history to plot[/yellow]")
        return

    # Create a tracking camera (EXACTLY from template)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # Initialize world to get the background (EXACTLY from template)
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    bg_path = str(data_dir / "background.png")
    single_frame_renderer(
        model,
        data,
        # camera=camera,
        save_path=bg_path,
        save=True,
    )

    # Setup background image (EXACTLY from template)
    img = plt.imread(bg_path)
    _, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    w, h, _ = img.shape

    # Convert list of [x,y,z] positions to numpy array (EXACTLY from template)
    pos_data = np.array(history)

    # Calculate initial position (EXACTLY from template)
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]

    # Convert position data to pixel coordinates (EXACTLY from template)
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    # Plot x,y trajectory (EXACTLY from template)
    ax.plot(x0, y0, "kx", markersize=10, label="[0, 0, 0]")
    ax.plot(xc, yc, "go", markersize=10, label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", linewidth=2, label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", markersize=10, label="End")

    # Add labels and title
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    plt.title("Robot Path in XY Plane")

    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


class Individual:
    """Represents one robot: genotype (NDE vectors + NN weights) + fitness."""

    def __init__(self, nde_vectors: List[npt.NDArray[np.float32]],
                    controller_weights: npt.NDArray[np.float64],
                    sigma: float = 0.2):
            self.nde_vectors = nde_vectors
            self.controller_weights = controller_weights
            self.sigma = sigma  # self-adaptive mutation strength

            self.robot_graph = None
            self.fitness = -1000.0
            self.path_history = None
            self.evaluated = False


def create_controller_function(weights: npt.NDArray[np.float64],
                               input_size: int,
                               output_size: int):
    """
    Create a controller callback function with given weights.
    Based on nn_controller from A3 template but with evolved weights.
    """
    # Parse weights into layers
    w1_size = input_size * HIDDEN_SIZE
    w2_size = HIDDEN_SIZE * HIDDEN_SIZE
    w3_size = HIDDEN_SIZE * output_size

    idx = 0
    w1 = weights[idx:idx + w1_size].reshape(input_size, HIDDEN_SIZE)
    idx += w1_size
    w2 = weights[idx:idx + w2_size].reshape(HIDDEN_SIZE, HIDDEN_SIZE)
    idx += w2_size
    w3 = weights[idx:idx + w3_size].reshape(HIDDEN_SIZE, output_size)

    def controller(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
        """Neural network controller (from A3 template structure)."""
        inputs = data.qpos

        layer1 = np.tanh(np.dot(inputs, w1))
        layer2 = np.tanh(np.dot(layer1, w2))
        outputs = np.tanh(np.dot(layer2, w3))

        control = outputs * np.pi

        return control

    return controller

def evaluate_individual(individual: Individual, rng: np.random.Generator) -> float:
    """Decode body, construct once to validate, run ES for brain, assign fitness."""
    try:
        # Fresh decoder
        nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
        decoder = HighProbabilityDecoder(NUM_OF_MODULES)

        # Decode morphology
        p_matrices = nde.forward(individual.nde_vectors)
        robot_graph: DiGraph[Any] = decoder.probability_matrices_to_graph(
            p_matrices[0], p_matrices[1], p_matrices[2]
        )

        if robot_graph is None or len(robot_graph.nodes) < 2:
            individual.fitness = -1000.0
            return -1000.0

        individual.robot_graph = robot_graph

        # Try constructing once just to check validity
        try:
            _ = construct_mjspec_from_graph(robot_graph)
        except Exception as e:
            console.log(f"[yellow]Construction failed: {e}[/yellow]")
            individual.fitness = -1000.0
            return -1000.0

        # Run ES optimization
        best_w, best_f, best_path = optimize_brain_es(
            robot_graph, individual.controller_weights, rng,
            steps=5, pop_size=10, sigma=0.2
        )

        individual.controller_weights = best_w
        individual.fitness = best_f
        individual.path_history = best_path
        individual.evaluated = True
        return best_f

    except Exception as e:
        console.log(f"[red]Evaluation error: {e}[/red]")
        individual.fitness = -1000.0
        return -1000.0

def get_dynamic_duration(best_fitness: float) -> int:
    """Increase simulation time based on progress (fitness)."""
    if best_fitness < -500:   # still very poor
        return 15             # only 15s
    elif best_fitness < -100: # some progress
        return 45
    elif best_fitness < 0:    # close to target
        return 70
    else:
        return 100            # enough to reach finish

def create_random_individual(rng: np.random.Generator) -> Individual:
    """Create random individual."""
    # NDE vectors: 3 vectors of GENOTYPE_SIZE (from template)
    type_genes = rng.random(GENOTYPE_SIZE).astype(np.float32)
    conn_genes = rng.random(GENOTYPE_SIZE).astype(np.float32)
    rot_genes = rng.random(GENOTYPE_SIZE).astype(np.float32)

    nde_vectors = [type_genes, conn_genes, rot_genes]

    # Controller weights: estimate size (will adjust during evaluation)
    # 3-layer NN: input_size unknown, use max estimate
    # Using same initialization as template: loc=0.0138, scale=0.5
    max_weights = 50 * HIDDEN_SIZE + HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * 20
    controller_weights = rng.normal(loc=0.0138, scale=0.5, size=max_weights)

    return Individual(nde_vectors, controller_weights)


def tournament_selection(population: List[Individual], tournament_size: int,
                        rng: np.random.Generator) -> Individual:
    """Tournament selection."""
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    tournament = [population[i] for i in indices]
    winner = max(tournament, key=lambda ind: ind.fitness)
    return winner


def crossover_blend(parent1: Individual, parent2: Individual,
                   alpha: float, rng: np.random.Generator) -> Tuple[Individual, Individual]:
    """
    Blend crossover for body (NDE vectors) and single-point for controller.
    """
    # Crossover NDE vectors (blend)
    child1_nde = []
    child2_nde = []

    for i in range(3):
        # Blend crossover
        beta = rng.uniform(-alpha, 1 + alpha)
        child1_vec = beta * parent1.nde_vectors[i] + (1 - beta) * parent2.nde_vectors[i]
        child2_vec = (1 - beta) * parent1.nde_vectors[i] + beta * parent2.nde_vectors[i]

        # Clip to [0, 1] as in NDE requirements
        child1_nde.append(np.clip(child1_vec, 0, 1).astype(np.float32))
        child2_nde.append(np.clip(child2_vec, 0, 1).astype(np.float32))

    # Crossover controller weights (single-point)
    min_len = min(len(parent1.controller_weights), len(parent2.controller_weights))
    point = rng.integers(1, min_len)

    child1_weights = np.concatenate([
        parent1.controller_weights[:point],
        parent2.controller_weights[point:]
    ])
    child2_weights = np.concatenate([
        parent2.controller_weights[:point],
        parent1.controller_weights[point:]
    ])

    child_sigma = (parent1.sigma + parent2.sigma) / 2
    return (
        Individual(child1_nde, child1_weights, sigma=child_sigma),
        Individual(child2_nde, child2_weights, sigma=child_sigma),
    )


def mutate_gaussian(individual: Individual, mutation_rate: float,
                   rng: np.random.Generator) -> Individual:
    """
    Gaussian mutation with self-adaptive sigma (log-normal update).
    """
    # Update sigma
    n = GENOTYPE_SIZE * 3 + len(individual.controller_weights)
    tau = 1 / np.sqrt(2 * np.sqrt(n))
    individual.sigma *= np.exp(tau * rng.normal(0, 1))
    individual.sigma = max(0.01, min(individual.sigma, 1.0))  # clamp range

    sigma = individual.sigma

    # Mutate NDE vectors
    for i in range(3):
        mask = rng.random(GENOTYPE_SIZE) < mutation_rate
        if np.any(mask):
            noise = rng.normal(0, sigma, GENOTYPE_SIZE).astype(np.float32)
            individual.nde_vectors[i][mask] += noise[mask]
            individual.nde_vectors[i] = np.clip(individual.nde_vectors[i], 0, 1)

    # Mutate controller weights
    mask = rng.random(len(individual.controller_weights)) < mutation_rate
    if np.any(mask):
        noise = rng.normal(0, sigma, len(individual.controller_weights))
        individual.controller_weights[mask] += noise[mask]

    return individual


def simulate_with_brain(robot_graph, weights, rng):
    """Simulate a robot defined by robot_graph with given brain weights, return fitness + path."""
    mj.set_mjcb_control(None)

    # Rebuild robot spec
    robot = construct_mjspec_from_graph(robot_graph)
    world = OlympicArena()
    world.spawn(robot.spec, SPAWN_POS)

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    # Tracker
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    # Controller setup
    input_size = len(data.qpos) if len(data.qpos) > 0 else 1
    output_size = model.nu if model.nu > 0 else 1
    required_weights = input_size * HIDDEN_SIZE + HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * output_size
    if len(weights) < required_weights:
        extra = rng.normal(0, 0.5, required_weights - len(weights))
        weights = np.concatenate([weights, extra])

    controller_func = create_controller_function(weights, input_size, output_size)
    ctrl = Controller(controller_callback_function=controller_func, tracker=tracker)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    # Dynamic duration based on global best fitness
    duration = get_dynamic_duration(GLOBAL_BEST_FITNESS)
    simple_runner(model, data, duration=duration)

    # Collect path
    path = tracker.history["xpos"][0] if "xpos" in tracker.history and len(tracker.history["xpos"]) > 0 else []
    fitness = race_fitness_function(path, duration)
    return fitness, path

def optimize_brain_es(robot_graph, init_weights, rng, steps=5, pop_size=32, sigma=0.2):
    global GLOBAL_BEST_FITNESS

    mean = init_weights.copy()
    best_w = mean.copy()
    best_f = -1e9
    best_path = []
    sigma_val = sigma

    for step in range(steps):
        # Generate candidate solutions
        candidates = [mean + rng.normal(0, sigma_val, size=mean.shape) for _ in range(pop_size)]

        # Run simulations in parallel
        with Pool(processes=min(pop_size, 8)) as pool:  # adjust #workers for your CPU
            results = pool.starmap(
                simulate_with_brain,
                [(robot_graph, w, np.random.default_rng(rng.integers(1e9))) for w in candidates]
            )

        # Attach weights to results
        results = list(zip(candidates, [r[0] for r in results], [r[1] for r in results]))

        # Sort by fitness
        results.sort(key=lambda x: x[1], reverse=True)
        elites = results[:max(1, pop_size // 5)]

        # Update mean
        mean = np.mean([w for w, _, _ in elites], axis=0)

        # Update best
        if elites[0][1] > best_f:
            best_w, best_f, best_path = elites[0]
            sigma_val *= 0.95
        else:
            sigma_val *= 1.05

        if best_f > GLOBAL_BEST_FITNESS:
            GLOBAL_BEST_FITNESS = best_f

    return best_w, best_f, best_path



from rich.progress import track

def run_evolution(population_size: int, num_generations: int,
                 crossover_prob: float, mutation_prob: float,
                 tournament_size: int, baseline: bool,
                 random_seed: int) -> Tuple[Individual, List[float], List[float], List[Individual]]:
    """
    Main evolutionary algorithm using only numpy and ARIEL components.
    Cleaner logging with progress bars and per-generation summaries.
    """
    # Setup
    rng = np.random.default_rng(random_seed)
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    decoder = HighProbabilityDecoder(NUM_OF_MODULES)

    # Tracking
    best_fitness_history = []
    avg_fitness_history = []
    best_individuals_history = []

    # Initialize population
    console.log(f"[cyan]Initializing population of {population_size}[/cyan]")
    population = [create_random_individual(rng) for _ in range(population_size)]

    # Evaluate initial population with progress bar
    console.log("[cyan]Evaluating generation 0...[/cyan]")
    for ind in track(population, description="Evaluating Gen 0"):
        evaluate_individual(ind, rng)

    fitnesses = [ind.fitness for ind in population]
    best_fitness_history.append(max(fitnesses))
    avg_fitness_history.append(np.mean(fitnesses))
    best_ind = max(population, key=lambda x: x.fitness)
    best_individuals_history.append(best_ind)

    console.log(f"Gen 0: Best={max(fitnesses):.2f}, Avg={np.mean(fitnesses):.2f}")

    # Evolution loop
    for gen in range(1, num_generations + 1):
        console.log(f"[bold cyan]Generation {gen}/{num_generations}[/bold cyan]")

        if baseline:
            # Baseline: random controllers each generation (no learning)
            for ind in population:
                ind.controller_weights = rng.normal(0, 0.1, len(ind.controller_weights))
                ind.evaluated = False

            for ind in track(population, description=f"Baseline Gen {gen} eval"):
                evaluate_individual(ind, rng)

        else:
            offspring = []

            # Check if population collapsed
            valid_count = sum(1 for ind in population if ind.fitness > -1000)
            if valid_count < 2:
                console.log(f"[yellow]Population collapsed ({valid_count} valid). Re-initializing...[/yellow]")
                population = [create_random_individual(rng) for _ in range(population_size)]

                for ind in track(population, description=f"Re-init Gen {gen} eval"):
                    evaluate_individual(ind, rng)

                fitnesses = [ind.fitness for ind in population]
                best_fitness_history.append(max(fitnesses))
                avg_fitness_history.append(np.mean(fitnesses))
                best_ind = max(population, key=lambda x: x.fitness)
                best_individuals_history.append(best_ind)
                continue

            # Generate offspring
            while len(offspring) < population_size:
                parent1 = tournament_selection(population, tournament_size, rng)
                parent2 = tournament_selection(population, tournament_size, rng)

                if rng.random() < crossover_prob:
                    child1, child2 = crossover_blend(parent1, parent2, alpha=0.5, rng=rng)
                else:
                    child1 = Individual([v.copy() for v in parent1.nde_vectors],
                                        parent1.controller_weights.copy(),
                                        sigma=parent1.sigma)
                    child2 = Individual([v.copy() for v in parent2.nde_vectors],
                                        parent2.controller_weights.copy(),
                                        sigma=parent2.sigma)

                if rng.random() < mutation_prob:
                    mutate_gaussian(child1, mutation_rate=0.20, rng=rng)
                if rng.random() < mutation_prob:
                    mutate_gaussian(child2, mutation_rate=0.20, rng=rng)

                offspring.extend([child1, child2])

            offspring = offspring[:population_size]

            # Evaluate offspring with progress bar
            console.log("[cyan]Evaluating offspring...[/cyan]")
            for ind in track(offspring, description=f"Evaluating Gen {gen} offspring"):
                evaluate_individual(ind, rng)

            # Survival selection (elitism)
            combined = population + offspring
            combined.sort(key=lambda x: x.fitness, reverse=True)
            population = combined[:population_size]

        # Track statistics (once per generation)
        fitnesses = [ind.fitness for ind in population]
        best_fitness_history.append(max(fitnesses))
        avg_fitness_history.append(np.mean(fitnesses))
        best_ind = max(population, key=lambda x: x.fitness)
        best_individuals_history.append(best_ind)

        console.log(f"Gen {gen}: Best={max(fitnesses):.2f}, Avg={np.mean(fitnesses):.2f}")

    # Return best individual
    best_individual = max(population, key=lambda x: x.fitness)
    return best_individual, best_fitness_history, avg_fitness_history, best_individuals_history



def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Robot Olympics - Assignment 3")
    parser.add_argument("--baseline", action="store_true",
                       help="Run baseline with random controllers")
    parser.add_argument("--generations", type=int, default=20,
                       help="Number of generations (default: 20)")
    parser.add_argument("--population", type=int, default=20,
                       help="Population size (default: 20)")
    parser.add_argument("--repetitions", type=int, default=1,
                       help="Number of repetitions (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--render-video", action="store_true",
                       help="Render video of best robot")

    args = parser.parse_args()

    script_name = "robot_olympics"
    if args.baseline:
        script_name += "_baseline"

    cwd = Path.cwd()
    data_dir = cwd / "results_olympics" / script_name
    data_dir.mkdir(parents=True, exist_ok=True)

    console.log("[bold]Robot Olympics - Assignment 3[/bold]")
    console.log(f"Mode: {'Baseline' if args.baseline else 'Evolution'}")
    console.log(f"Generations: {args.generations}, Population: {args.population}")
    console.log(f"Results: {data_dir}")

    # Run repetitions
    for rep in range(args.repetitions):
        console.log(f"\n[bold green]=== Repetition {rep + 1}/{args.repetitions} ===[/bold green]")

        # Run evolution
        best_ind, best_hist, avg_hist, best_inds = run_evolution(
            population_size=args.population,
            num_generations=args.generations,
            crossover_prob=0.8,
            mutation_prob=0.2,
            tournament_size=3,
            baseline=args.baseline,
            random_seed=args.seed + rep
        )

        console.log(f"[green]Best fitness: {best_ind.fitness:.2f}[/green]")

        # Save results
        result = {
            'repetition': rep,
            'best_fitness': best_ind.fitness,
            'best_fitness_history': best_hist,
            'avg_fitness_history': avg_hist,
            'best_individual': best_ind,
            'parameters': vars(args)
        }

        with open(data_dir / f"rep_{rep}_results.pkl", 'wb') as f:
            pickle.dump(result, f)

        # Plot fitness
        plt.figure(figsize=(10, 6))
        plt.plot(best_hist, 'b-', linewidth=2, label='Best')
        plt.plot(avg_hist, 'r--', linewidth=2, label='Average')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(data_dir / f"rep_{rep}_fitness.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Plot best path
        if best_ind.path_history:
            show_xpos_history(best_ind.path_history,
                            data_dir / f"rep_{rep}_best_path.png", data_dir)

        # Save best robot as JSON (PHENOTYPE for submission)
        if best_ind.robot_graph:
            save_graph_as_json(best_ind.robot_graph, data_dir / f"rep_{rep}_best_robot.json")

            # Save controller weights
            with open(data_dir / f"rep_{rep}_best_controller.pkl", 'wb') as f:
                pickle.dump(best_ind.controller_weights, f)

        # Render video
        if args.render_video and best_ind.robot_graph:
            console.log("[cyan]Rendering video...[/cyan]")
            try:
                # Following A3 template pattern
                robot = construct_mjspec_from_graph(best_ind.robot_graph)
                mj.set_mjcb_control(None)
                world = OlympicArena()
                world.spawn(robot.spec, SPAWN_POS)
                model = world.spec.compile()
                data = mj.MjData(model)
                mj.mj_resetData(model, data)

                input_size = len(data.qpos) if len(data.qpos) > 0 else 1
                output_size = model.nu if model.nu > 0 else 1

                controller_func = create_controller_function(
                    best_ind.controller_weights, input_size, output_size
                )
                ctrl = Controller(controller_callback_function=controller_func)

                args_list: list[Any] = []
                kwargs_dict: dict[Any, Any] = {}
                mj.set_mjcb_control(
                    lambda m, d: ctrl.set_control(m, d, *args_list, **kwargs_dict)
                )

                video_dir = str(data_dir / "videos")
                video_recorder = VideoRecorder(output_folder=video_dir)
                video_renderer(model, data, duration=SIMULATION_DURATION,
                             video_recorder=video_recorder)

                console.log(f"[green]Video saved to {video_dir}[/green]")
            except Exception as e:
                console.log(f"[red]Video rendering failed: {e}[/red]")

    console.log(f"\n[bold green]Complete! Results in {data_dir}[/bold green]")


if __name__ == "__main__":
    main()

