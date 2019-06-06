# Using Tensorboard

### Setting up
First, you need a fresh (>= 1.1.0) pytorch version installed, as tensorboard was supported only recently.
Tensorboard itself has to be installed separately, see [instructions](https://pytorch.org/docs/stable/tensorboard.html).

### Using with EGG
By default, Tensorboard reporting is disabled; to enable it, you need
to set the flag `--tensorboard` when launching your game. Another useful flag sets the path for the output logs,
`--tensorboard_dir` (default is `./runs/`). Make sure that different experiments use different directories, 
otherwise the results would be mixed.

An example command then would be:
```bash
python -m egg.zoo.mnist_autoenc.train --tensorboard --tensorboard_dir=./runs/mnist_example/
```

After training is finished, the logs can be analyzed by running tensorboard and opening its page on
a browser (http://127.0.0.1:6006/):
```bash
tensorboard --logdir=./runs/ 
```

### Using in your game

To get access to the `SummaryWriter` instance (see [pytorch description](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter)),users have to call the `egg.core.utils.get_summary_writer()` function.
Since the writer writes the statistics asynchronously, it is necessary to call `egg.core.close()` at the end of a program - 
otherwise the last datapoints might be lost.
