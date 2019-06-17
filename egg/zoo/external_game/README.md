We call this the `external_game` because it is a Sender/Receiver game where inputs and labels (target outputs) come from an external CSV file (as in [this example](classification.data)). This allows the user to exploit EGG functionalities entirely from the command line, with no need to directly adapt the code.

Communication takes place via variable-length messages. Both agents are implemented as single-layer recurrent cells. The game supports training with Gumbel-Softmax relaxation of the communication
channel (in which case we optimize a differentiable loss) or by REINFORCE (with 0-1 reward based on accuracy of the Receiver output). Other parameters can be controlled via command-line options, as specified below.

As illustrated by [this example](classification.data), the input files contain two fields, separated by a semi-colon `;`. The first field specifies a vector-valued input to Sender (vector values are space-delimited), the second specifies a multi-class label (an integer), that must be predicted by the Receiver. Note that the script assumes that class labels range from 0 to the largest integer found in the relevant field of the training file. For example, even if the training file contains only class labels 2 and 4, the script will assume that this is a 5-way classification problem, with class labels ranging from 0 to 4.

The `game.py` script of this game supports two exclusive regimes: `train` and `dump`. The first is used to train a model and the second is used for evaluation and analysis given a pre-trained model and some input data: In `dump` regime, the messages sent and Receiver outputs are saved ("dumped"). The script assumes the training regime unless a dataset for dumping is provided (with the `--dump_data` parameter and loading the trained model from a checkpoint created in training mode). When in `dump` regime and if some non-default model parameters were used at training time, one needs to specify them again. In both regimes, the number of input features and output classes is automatically inferred from the training set (which means that `train_data` argument must be passed in `dump` mode as well).

The game can be run as follows (training regime):
```bash
python egg/zoo/external_game/game.py --train_data=./egg/zoo/external_game/classification.data \
    --validation_data=./egg/zoo/external_game/classification.data --n_epoch=150 --train_mode=gs --random_seed=21 \
    --lr=1e-2 --max_len=2 --checkpoint_dir=./
```
After training, a model file named `150.tar` (which refers to the ```{n_epochs}.tar```) will be saved in the current directory.
Next, switching to the `dump` regime, the trained model can be run on new data, with output behaviour printed to standard output (as in this example), or to a text file (as shown below).
```bash
python egg/zoo/external_game/game.py --train_data=./egg/zoo/external_game/classification.data   \
    --dump_data=./egg/zoo/external_game/classification.data --train_mode=gs --max_len=2 \
    --load_from_checkpoint=150.tar

```
Note that here we re-used the training data as test data for convenience, but a more typical case will be one in which the latter are from a separate test file. Note also that we still have to specify the model parameters (`--train_mode=gs --max_len=2` in this case).

Training with REINFORCE is similar, and several other options are illustrated by the following example:
```bash
python egg/zoo/external_game/game.py --train_data=./egg/zoo/external_game/binary_classification.data \
    --validation_data=./egg/zoo/external_game/binary_classification.data --n_epoch=250 --train_mode=rf --random_seed=21 \
     --lr=0.005 --max_len=2 --vocab_size=4 --receiver_hidden=30 --sender_hidden=30 --sender_entropy_coeff=1e-1 --checkpoint_dir=./

```
To output the results to the file `out.txt` (again, recycling the same data we did for training for dumping), we run:
```bash
python egg/zoo/external_game/game.py --dump_data=./egg/zoo/external_game/binary_classification.data
     --train_data=./egg/zoo/external_game/binary_classification.data \
     --train_mode=rf --vocab_size=4 --receiver_hidden=30 --sender_hidden=30 \
    --max_len=2 --load_from_checkpoint=250.tar --dump_output=out.txt
```
The format of the dumped output (printed to stdout or to a file) is as follows:
```
input_vector;sender_message;receiver_output;gold_label_value
```
where:
* `input_vector` is the input vector, in comma-delimited format
* `sender_message` is the message sent by the Sender, in comma-delimited format
* `receiver_output` is the label value produced by the Receiver
* `gold_label_value` is the gold label for the current input


## Game configuration parameters:
 * `train_path/validation_path` -- paths for the train and validation set files. The validation set is optional, and if specified it is used to compute and print validation loss periodically.
 * `random_seed` -- random seed to be used (if not specified, a random value will be used).
 * `checkpoint_dir` and `checkpoint_freq` specify where the model and optimizer state will be checkpointed and how often. For instance, `--checkpoint_dir=./checkpoints --checkpoint_freq=10`, configures the checkpoints to be stored in 
     `./checkpoints` every 10 epochs.
     If `checkpooint_dir` is specified, then the model obtained at the end of training will be saved there under the name
     `{n_epochs}.tar`.
 * `load_from_checkpoint` -- loads the model and optimizer state from a checkpoint. If `--n_epochs` is larger than that
    in the checkpoint, training will continue.
 * `dump_data`/`dump_output` -- if a `dump_data` file is specified, input/message/output/gold label are produced for the inputs in the file. If `dump_output` is not used, the dump is printed to stdout, otherwise, it is written into the speficied `data_output` file.

## Model parameters:
 * `sender_cell/receiver_cell` -- the recurrent cell type used by the agents; can be any of {rnn, gru, lstm} (default: rnn).
 * `sender_layers/receiver_layers` -- the number of RNN, GRU, or LSTM layers used by Sender and Receiver (default: 1). At the moment, these options are only supported if training with REINFORCE.
 * `sender_hidden/receiver_hidden` -- the size of the hidden layers for the cells (default: 10).
 * `sender_embedding/receiver_embedding` -- the size of the embedding space (default: 10).
 * `vocab_size` -- the number of unique symbols in the vocabulary (inluding `<eos>`!). `<eos>` is conventionally mapped to 0. Default: 10.
 * `max_len` -- the maximal length of the message. Receiver's output is processed either after the `<eos>` symbol is received
 or after `max_len` symbols, and further symbols are ignored.
 * `force_eos` -- forces that each message sequence is terminated by a EOS. If not set (by default), the communication stops
 either when Sender sends an EOS symbol or when `max_len` is reached. Hence, it might happen that the last symbol consumed by Receiver is not 
 EOS. If set, all messages are forced to have the EOS. This is achieved by reducing `max_len` by 1 and appending the EOS symbol to each sequence.

## Training hyper-parameters:
 * `lr` -- the learning rates for the agents' parameters (it might be useful to have Sender's learning rate
 lower, as Receiver has to adjust to the changes in Sender) (default: 1e-1).
 * `optimizer` -- selects the optimizer (`adam/adagrad/sgd`, defaults to Adam).
 * `no_cuda` -- disable usage of CUDA even if it is available (by default, CUDA is used if present).
 * `train_mode` -- whether Gumbel-Softmax (`gs`) or Reinforce-based (`rf`) training is used (default: gs). If Reinforce 
    is specified, then both Sender and Receiver sample their outputs and the 0-1 accuracy of the Receiver label compared to the gold one is optimized. In case of 
    Gumbel-Softmax relaxation, Receiver is deterministic and the optimized loss is cross-entropy.
 * `sender_entropy_coeff/receiver_entropy_coeff` -- if Reinforce is used, the regularisation coefficients for the
 entropy term in the loss (default: 1e-2).
 * `temperature` -- if Gumbel-Softmax is used, temperature parameter (default: 1.0)
 * `n_epochs` -- number of training epochs (default: 10).
 * `batch_size` -- size of a batch. Note that it will be capped by dataset size (e.g., if the training dataset has
    1000 rows, batch cannot be larger than 1000) (default: 32).

