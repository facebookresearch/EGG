# Dealing with command-line (CL) parameters

Most games have a set of similar command-line parameters, such as size of the batch, size of the vocabulary
that is used to communicate over the channel, where the checkpoint models are stored, the random seed to be used, etc.

To simplify this boilerplate configuration, EGG provides a pre-defined set of common parameters. When relevant, those parameters 
are automatically used by the training logic.

In order to set these parameters, two steps are needed:
 * If your game requires some of its own configuration, you need to create an instance of `argparse.ArgumentParser` and
    add your required parameters to it;
 * Call `egg.core.init()` to parse them.
 
Behind the scenes, EGG will populate the list of parameters and will call the `parse_args()` method of the ArgumentParser instance.

Overall, the standard usage is supposed to look as follows:
```python
import argparse
import egg.core as core
...

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_specific_parameter', type=int, default=10,
                        help='A game-specific parameter (default: 10)')
    ...
    args = core.init(parser)
    return args

if __name__ == '__main__':
    args = get_args() 
    
```
    
The list of the supported "common" parameters can be found in [util.py](https://github.com/facebookresearch/EGG/blob/master/egg/core/util.py) and it includes both parameters that are read 
by EGG itself and those that could be used by the user. Below we briefly discuss them.

### EGG-used parameters
These parameters are used by EGG internally:

* `random_seed` - used by EGG to set the random seed for both CPU and CUDA (if available). If not specified from CL,
    it will store an auto-generated value, that can be retrieved and re-used later (e.g., if, after a run, you want to replicate it);
* `no_cuda` - disables the use of CUDA even when it is available. By default, EGG uses CUDA when present;
* `load_from_checkpoint` - if specified, EGG loads model, optimizer, and trainer state from the specified file;
* `checkpoint_dir` and `checkpoint_freq` - if specified, checkpoints wil be stored every `checkpoint_freq` epochs to 
    `checkpoint_dir`. The names of the checkpoints would be `{number_of_epochs}.tar`;

### Pre-defined convenience parameters
These parameters are simply defined for the user's convenience. They can be used or ignored; although we advice to use 
them if they fit your game, so that the names are uniform across games. The pre-implemented games in the _zoo_ use those parameters when 
appropriate.

* `n_epochs` should be used to set the number of training epochs;
* `vocab_size` (default: 10) and `max_len` (default: 1) should set the vocabulary size and the maximal message length in the channel;
* `batch_size` (default: 32) should set size of the training batch;
* `optimizer/lr` select an optimizer and a learning rate. `adam` (default), `sgd`, `adagrad` are supported. Default value for 
    `lr` is `1e-2`. 
To use optimizer parameters, EGG provides an utility constructor:
```python
import egg.core as core
...
model = build_my_model(...)
optimizer = core.get_optimizer(model.parameters())
```

In this case, it would be set according to the CL-specified `optimizer` and `lr` parameters.
