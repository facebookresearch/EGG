Here you can find a set of the games/tasks that are used in this [paper](arxiv.org).

# Structure

Each game directory contains the definition of the agents' architecture, training logic, hyperparameter grids compatible
with the nest tool, and notebooks that were used to obtain the figures in the text.

```text
guess_number/ # Guess Number task
    ./hyperparam_grid/ # json files with hyperparameter grids
mnist_classification/ # Image Classification task
mnist_overfit/ # MNIST overfitting experiments
mnist_adv/ # adversarial robustness experiments
```

# Running
Launching each game is as simple as regular EGG games, e.g.
```bash
python -m egg.zoo.language_bottleneck.mnist_classification
```

For the set of game-specific parameters, please check each game's `train.py` script.
