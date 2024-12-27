# Experiments

This directory contains self-contained subdirectories, each dedicated to a specific experiment designed to achieve a unique objective.

## Experiment Goals

### [`01 Avoid Collision`](./01_avoid_collision)
   - **Objective**: Train the snake to survive by avoiding collisions with itself or the walls.
   - **Focus**: Build basic survival instincts and reward the snake for staying alive.

### [`02 Apple Hunting`](./02_apple_hunting)
   - **Objective**: Train the snake to actively seek and collect apples.
   - **Focus**: Encourage exploration and optimize reward collection by eating apples.

## Structure of Each Experiment
Each experiment folder includes:
- `config.yaml`: Configuration file containing hyperparameters and environment settings.
- `train.py`: Training script specific to the experiment's goals.
- `README.md`: Detailed description of the experiment's objectives, parameters, and results.
- `results/`: Folder to store logs, trained models, and generated plots.

## How to Run an Experiment
1. Navigate to the desired experiment directory:
    ``` bash
    cd experiments/<experiment_name>
    ```

2.	Execute the training script:
    ``` bash
    python -m train
    ```
