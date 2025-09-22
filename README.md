# Robot Neuroevolution - Assignment 2

**Course**: Evolutionary Computing
**Assignment**: A2 - Robot Neuroevolution (ARIEL Gecko Robot)


## Installation & Setup

### Prerequisites

1. **ARIEL Framework**: The experiments require the ARIEL robotics simulation framework.
2. **UV Package Manager**: Recommended for managing Python dependencies.
3. **Python 3.11+**: Required for ARIEL compatibility.

### Step-by-Step Installation

1. **Install UV (if not already installed)**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   ```

2. **Clone ARIEL** (should be in parent directory):
   ```bash
   cd ..
   git clone https://github.com/ci-group/ariel
   cd ariel
   ```

3. **Set up ARIEL virtual environment**:
   ```bash
   uv venv
   uv sync
   ```

4. **Install additional dependencies**:
   ```bash
   uv add tqdm deap matplotlib seaborn
   ```

5. **Test ARIEL installation**:
   ```bash
   source .venv/bin/activate
   python examples/_robogen_build.py
   ```

6. **Return to project directory and test**:
   ```bash
   cd ../EC2
   python A2_try.py  # Test implementation
   ```

## Evolutionary Algorithms Implemented

### 1. Baseline: Random Controller
- **Description**: Random neural network weights for comparison
- **Purpose**: Establish performance baseline
- **Implementation**: Random sampling from Gaussian distribution

### 2. Evolution Strategies (ES)
- **Type**: (μ+λ) Evolution Strategies
- **Key Features**:
  - Mutation-only evolution (no crossover)
  - Gaussian perturbation of weights
  - Tournament selection
  - Elitism (best individual preserved)
- **Parameters**:
  - Population size: 50
  - Generations: 100
  - Mutation rate: 0.1
  - Tournament size: 3

### 3. Genetic Algorithm (GA)
- **Type**: Standard Genetic Algorithm
- **Key Features**:
  - Single-point crossover
  - Gaussian mutation
  - Tournament selection
  - Elitism
- **Parameters**:
  - Population size: 50
  - Generations: 100
  - Crossover rate: 0.8
  - Mutation rate: 0.05
  - Tournament size: 3

### 4. DEAP-based Algorithms (Alternative Implementation)
- **Library**: DEAP (Distributed Evolutionary Algorithms in Python)
- **Variants**: GA, ES, CMA-ES approximation
- **Purpose**: Comparison with established EA library

## Neural Network Architecture

### Controller Structure
- **Input Layer**: 29 neurons
  - 15 joint positions (qpos)
  - 14 joint velocities (qvel)
- **Hidden Layer**: 16 neurons (tanh activation)
- **Output Layer**: 8 neurons (8 actuators)
  - Output range: [-π/2, π/2] for joint control

### Gecko Robot Specifications
- **Actuators**: 8 servo motors
  1. robot-neckservo
  2. robot-spineservo
  3. robot-fl_legservo (front left leg)
  4. robot-fl_flipperservo (front left flipper)
  5. robot-fr_legservo (front right leg)
  6. robot-fr_flipperservo (front right flipper)
  7. robot-bl_legservo (back left leg)
  8. robot-br_legservo (back right leg)

## Fitness Function

The fitness function combines multiple objectives to encourage effective locomotion:

```python
def calculate_fitness():
    # Distance-based fitness: total distance traveled
    distances = sqrt(sum(diff(positions)^2))
    total_distance = sum(distances)

    # Forward movement bonus (positive x direction)
    forward_distance = final_x - initial_x

    # Stability penalty (excessive z-axis variation)
    stability_penalty = std(z_positions) * 2.0

    # Combined fitness
    fitness = total_distance + forward_distance * 2.0 - stability_penalty
    return max(fitness, 0.0)  # Ensure non-negative
```

## Running Experiments


### Full Experiments (3 algorithms × 3 repetitions)
```bash
source ../ariel/.venv/bin/activate
python neuroevolution_experiments.py
```

```bash
source ../ariel/.venv/bin/activate

# Run experiments first (headless for speed)
python neuroevolution_experiments.py

# Then visualize the results
python neuroevolution_experiments.py --visualize
```


**Why Two Modes?** (I just wanted to see it move xD)
- **Headless Mode** (default): Fast batch processing for experiments (~5-6 evaluations/second)
- **Visual Mode** (`--visualize`): Interactive viewing for analysis (~1 evaluation/second)

### DEAP-based Experiments
```bash
source ../ariel/.venv/bin/activate
python neuroevolution_deap.py
```

### Individual Components Testing
```bash
python quick_experiment.py


python A2_try.py
```

### Command Line Options
```bash
python neuroevolution_experiments.py          # Run experiments (headless)
python neuroevolution_experiments.py --visualize  # Interactive robot viewer
```

## Results and Evaluation

### Experiment Output
Each run generates:
- **Results files**: `{algorithm}_rep{n}_results.pkl`
- **Best controllers**: `{algorithm}_rep{n}_best_controller.pkl`
- **Fitness plots**: Evolution curves with mean ± std shading
- **Summary statistics**: Final performance comparison

### Reproducing Results
All experiments are reproducible using fixed random seeds:
- Base seed: 42
- Repetition seeds: 42 + repetition_number

### Evaluation Metrics
1. **Best fitness achieved**: Maximum fitness across all generations
2. **Convergence rate**: Generations needed to reach target performance
3. **Stability**: Standard deviation across repetitions
4. **Computational efficiency**: Runtime per generation


### Evolutionary Computing Libraries
```
deap>=1.3.3          # DEAP evolutionary algorithms
tqdm>=4.60.0         # Progress bars
```

### ARIEL Dependencies
Handled automatically by UV when installing ARIEL:
```
ariel                 # Robot simulation framework
```

## External Libraries Used

Following the assignment guidelines, we utilized these allowed external libraries:

1. **DEAP**: Professional evolutionary algorithms library
   - Citation: Fortin, F.A., et al. (2012). "DEAP: Evolutionary algorithms made easy." JMLR.
   - Usage: Alternative implementation for comparison

2. **NumPy/SciPy**: Scientific computing
   - Standard numerical operations and statistics

3. **Matplotlib**: Plotting and visualization
   - Generation of fitness plots with error shading

## Scientific Approach & Methodology

### Experimental Design
- **Comparative study**: Multiple EA approaches on same problem
- **Statistical rigor**: 3 repetitions per experiment for significance testing
- **Baseline comparison**: Random controller performance as reference
- **Reproducibility**: Fixed seeds and documented parameters

### Algorithm Selection Rationale
1. **Evolution Strategies**: Well-suited for continuous optimization of neural weights
2. **Genetic Algorithm**: Classical approach for comparison, includes crossover
3. **Random Baseline**: Essential for determining if learning occurs

### Evaluation Protocol
1. Fixed simulation time (15 seconds) per evaluation
2. Consistent fitness function across all experiments
3. Same neural network architecture for fair comparison
4. Multiple repetitions for statistical validity


## References & Citations

1. **ARIEL Framework**: https://github.com/ci-group/ariel
2. **DEAP Library**: Fortin, F.A., et al. (2012). "DEAP: Evolutionary algorithms made easy." Journal of Machine Learning Research.
3. **MuJoCo Physics**: Todorov, E., et al. (2012). "MuJoCo: A physics engine for model-predictive control."
4. **Evolution Strategies**: Hansen, N., & Ostermeier, A. (2001). "Completely derandomized self-adaptation in evolution strategies."



---
