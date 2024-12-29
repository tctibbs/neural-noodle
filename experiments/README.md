# Experiments

This directory contains self-contained subdirectories, each dedicated to a specific experiment designed to achieve a unique objective.

## Experiment Goals

### [`01 Avoid Collision`](./01_avoid_collision)
   - **Objective**: Train the snake to survive for at least **10,000 steps** without colliding with itself or the walls.
   - **Focus**: Build basic survival instincts and reward the snake for staying alive.

### [`02 Apple Hunting`](./02_apple_hunting)
   - **Objective**: Train the snake to collect at least **10 apples** per episode while avoiding collisions.
   - **Focus**: Encourage exploration and optimize reward collection by eating apples.

## Structure of Each Experiment

Each experiment folder includes:
- `config.yaml`: Configuration file containing hyperparameters and environment settings.
- `README.md`: Detailed description of the experiment's objectives, parameters, and results.
- `results/`: Folder to store logs, trained models, and generated plots.

## How to Run an Experiment

   ``` bash
   python -m main --config experiments/<experiment_directory>/config.yaml
   ```
