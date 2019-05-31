## NEST

This is a simply utility to run a hyperparameter grid search for EGG-based games. We support two modes:
* Local search (via `nest_local.py`): runs multiple jobs on a single machine, leveraging all available CUDA devices (if any);
* Preemptable cluster-based (via `nest.py`).
 
Both modes have similar behaviour, however, the latter is only implemented for the FAIR environment and uses internal libraries.

# How to use

To run a hyperparameter search, we firstly need to define a json file with a list of parameters to be searched
through. The syntax is simple:
```json
{
"hyperparameter_name_1": ["value1", "value2"],
"hyperparameter_name_2": [1, 2, 3],
}
```
You can find an example in [example.json](example.json).

Next, the module that implements the game should have a function `main(params)` which accepts command-line parameters and 
runs the game logic. Hence, the typical implementation should follow this pattern:

```python
...
def main(params):
    egg.core.init(params=params)
    ...
    
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
```
An example game that meets this requirement is [mnist_autoenc](./../zoo/mnist_autoenc/train.py).


# Local mode
For a local search mode, we assume that (a) the game requires only single GPU, (b) it does not assign GPU devices 
directly.

After the json specification is set, we can run the search as follows:
```bash
python -m egg.nest.nest_local --game egg.zoo.mnist_autoenc.train --sweep egg/nest/example.json --n_workers=1
```

This command will spawn `n_workers` processes, playing the `egg.zoo.mnist_autoenc.train` game over a grid of hyperparameters
defined in `egg/nest/example.json `. If `n_workers` is more than the number of available GPU devices, the devices would 
be shared (e.g. if the number of workers is 2x the number of GPUs, each GPU would be shared by two workers).

`nest_local` has the following parameters:
 * `game` is a path to the game module to be run (e.g. egg.zoo.mnist_autoenc.train);
 * `sweep` is a path to the json file defining the grid of hyper parameters to be search through;
 * `preview/dry_run` suppress running any jobs, simply produces output and creates the output directory layout;
 * `n_workers` specifies the number of worker jobs;
 * `name` optional name of the search, which is used for naming the directory with the outputs of the runs. If not specified,
     `nest` would infer it from the `game` parameter (e.g. `mnist_autoenc` if the `game` parameter is set to `egg.zoo.mnist_autoenc.train`)
 * `root_dir` sets the output directory for the hyperparameter search. If not specified, `nest_local` would create a directory
    in `~/nest_local/{name}/{timestamp}/`. The stdout/stderr and the checkpoints of the jobs would be stored there;
 * `checkpoint_freq` sets how often the checkpoints would be set (default: 0). `checkpoint_dir` is set automatically by
 `nest_local` to be `{root_dir}/{hyperparameter_combination_id}/`.
 
