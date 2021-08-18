This code implements the signalling game described in [1]. The game proceeds as follows:
 * Sender is shown a target image alongside with one or many distractor images,
 * Sender sends a one-symbol message to Receiver,
 * Receiver obtains Sender's message and all images in random order,
 * Receiver predicts which of the received images is the target one and agents are rewarded if the prediction is correct.

To run the game, you need to pre-install `h5py` (`pip install h5py`).


The game can be launched with the following command (with appropriate path to the data):
```bash
python -m egg.zoo.signal_game.train --root=/private/home/kharitonov/work/egg/data/concepts/
```

The data used in the paper can be downloaded from [this link](https://dl.fbaipublicfiles.com/signaling_game_data).

The game can be configured with the following command-line parameters:
 * `--root` specifies the root folder of the data set
 * `--tau_gs` set the softmax temperature for Sender; valid for both Reinforce and Gumbel-Softmax training
        (defaults to 10.0)
 * `--game_size` set the number of the images used (one of them is the target, the rest are distractors, default: 2)
 * `--same` whether the distractor images should be sampled from the same concept as the target (either 0 or 1, default: 0)
 * `--vocab_size` sets the number of symbols for communication (default: 100)
 * `--batch_size` sets the batch size (default: 32)
 * `--embedding_size` sets the size of the symbol embeddings used by Receiver (default: 50)
 * `--hidden_size` the hidden layer size used by both agents (default: 20)
 * `--batches_per_epoch` how many batches per epoch (default: 100)
 * `--mode` specifies which training mode will be used - either Gumbel Softmax relaxation (`--mode=gs`) or Reinforce 
 (`--mode=rf`) (default: `rf`)
 * `--gs_tau` sets the Gumbel Softmax relaxation temperature (default: 1.0)
 
 It also accepts parameters that are common for all games, e.g.
 * `--n_epochs` the number of training epochs to run (default: 10)
 * `--random_seed` sets the random seed
 * `--lr` , `--optimizer`, ...
 see [this doc](https://github.com/facebookresearch/EGG/blob/main/docs/CL.md).

 
 
[1] *"Multi-agent cooperation and the emergence of (natural) language*, A. Lazaridou, A. Peysakhovich, M. Baroni 
[[arxiv]](https://arxiv.org/abs/1612.07182)
 
