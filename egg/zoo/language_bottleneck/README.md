Here you can find a set of the games/tasks that are used the following paper:
 * _Information Minimization In Emergent Languages_. Eugene Kharitonov, Rahma Chaabouni, Diane Bouchacourt, Marco Baroni.
 [arxiv](https://arxiv.org/abs/1905.13687).

To get a glimpse of some of the interesting things that can be done with Information Bottlenecks, see this tiny MNIST style transfer notebook: [notebook](/egg/zoo/language_bottleneck/mnist-style-transfer-via-bottleneck.ipynb) / [colab](https://colab.research.google.com/github/facebookresearch/EGG/blob/master/egg/zoo/language_bottleneck/mnist-style-transfer-via-bottleneck.ipynb).

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
