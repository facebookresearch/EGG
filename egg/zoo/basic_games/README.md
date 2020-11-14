## Introduction

Many language emergence games use a *reconstruction* or a *discrimination* task. In the reconstruction task, Sender gets an input item, it sends a message to Receiver, and Receiver must generate an output item identical to Sender's input. In the discrimination task, Sender gets an input item (the *target*); Receiver gets multiple input items (the same target and a number of *distractors*, in random order). Sender sends a message to Receiver, and Receiver has to output the location of the target in the array of its inputs.

The `play.py` script in this directory implements both types of tasks, as they share much of the same infrastructure. In particular, we consider the case here when inputs are vectors of discrete elements (interpretable as lists of categorical attribute-value pairs), and we let the user pass these inputs through text files. [This directory](data_generation_scripts) contains scripts that can generate input files in the right format, as well as samples from their outputs.

Both the main and the auxiliary game scripts are thoroughly commented, as we hope they will serve as starting points to acquaint yourself with EGG, and personalize it for your purposes.

## Reconstruction game

The reconstruction game reads input from files that have an input item (attribute-value list) on each line, as in [this example file](data_generation_scripts/example_reconstruction_input.txt), containing a list of 3-valued 5-attribute items. Although values are represented by integers, they are treated as categorical (that is, a value of 3 is as different from 4 as it is from 45).

Here is an example of how to run the reconstruction game (we do not need to specify that we are playing this type of game, as it is the default option):

```bash
python -m egg.zoo.basic_games.play --mode 'gs' --train_data "train_file.txt" --validation_data "valid_file.txt" --n_attributes 2 --n_values 10 --n_epochs 50 --batch_size 512 --validation_batch_size 1000 --max_len 4 --vocab_size 100 --sender_hidden 256 --receiver_hidden 512 --sender_embedding 5 --receiver_embedding 30 --receiver_cell "gru" --sender_cell "gru" --lr 0.01 --print_validation_events
```

In this particular instance, the following parameters are invoked:
 * `mode` -- tells whether to use Reinforce (`rf`) or Gumbel-Softmax (`gs`) for training.
 * `train_data/validation_data` -- paths to the files containing training data and validation data (the latter used at each epoch to track the progress of training); both files are in the same format.
 * `n_attributes` -- this is the number of "fields" that the input file vectors have: for example, given the input file linked above, `n_attributes` should be set to 5.
 * `n_values` -- number of distinct values that each input file vector field can take. As we are counting from 0, if the maximum value is 2 (as in the example file above), `n_values` should be set to 3 (and 0 constitutes a possible value).
 * `n_epochs` -- how many times the data in the input training file will be traversed during training: note that they will be traversed in a different random order each time.
 * `batch_size` -- batch size for training data (can't be smaller than number of items in training file).
 * `validation_batch_size` -- batch size for validation data, provided as a separate argument as it is often convenient to traverse the whole validation set in a single step.
 * `max_len` -- after `max_len` symbols without `<eos>` have been emitted by the Sender, an `<eos>` is forced; consequently, the longest possible message will contain `max_len` symbols, followed by `<eos>`.
 * `vocab_size` -- the number of unique symbols in the Sender vocabulary (inluding `<eos>`!).
 * `sender_hidden/receiver_hidden` -- the size of the hidden layers of the agents.
 * `sender_embedding/receiver_embedding` -- output dimensionality of the layer that embeds symbols produced at previous step by the Sender message-emitting/Receiver message-processing recurrent networks, respectively.
 * `sender_cell/receiver_cell` -- type of cell of recurrent networks agents use to emit/process the message.
 * `lr` -- learning rate.
 * `print_validation_events` -- if this flag is passed, after training is done the script will print the validation input, as well as the corresponding messages emitted by Sender and the corresponding Receiver outputs.
 
 To see all arguments that can be passed (and for more information on the ones above), run:
 
 ```bash
python -m egg.zoo.basic_games.play -h
```

## Discrimination game

The discrimination game reads the input from files that have, on each line, a tuple of items (attribute-value lists), followed by the index of the target in this sequence (counting from 0), as in [this example file](data_generation_scripts/example_discriminative_input.txt), representing a setup with 3-valued 5-attribute items arranged in tuples of two (a target and a distractor, in random order, with the last field indicating the position of the target). Items are period-delimited, and attributes are space-delimited. Note that the same number of items is expected on each line, that is, the number of distractors cannot change from episode to episode. As in the reconstruction game, although values are represented by integers, they are categorical, that is, for the purpose of the game, they are converted to a one-hot vector representation where there is not inherent similarity between numerically close values.

Here is an example of a discrimination game run:

```bash
python -m egg.zoo.basic_games.play --game_type 'discri' --mode 'rf' --train_data "discri_train_file.txt" --validation_data "discri_valid_file.txt" --n_values 10 --n_epochs 50 --batch_size 512 --validation_batch_size 10 --max_len 4 --vocab_size 100 --sender_hidden 256 --receiver_hidden 512 --sender_embedding 5 --receiver_embedding 30 --lr 0.01 --receiver_cell "gru" --sender_cell "gru" --random_seed 111 --print_validation_events
```

Most parameters were explained above, but notice the following:
 * `game_type` -- this determines whether we're going to play a discrimination (`discri`) or recognition (`reco`, the default) game.
 * `n_values` -- number of distinct values that each attribute can take. In the discrimination game, the number of attributes and the number of distractors is automatically computed from the input file.
 * `random_seed` -- a random seed can be passed to ensure that exactly the same experiment (same initializations, same shuffling of training data, etc.) can be reproduced.
 
## Output

The script will write output to STDOUT, including information on the chosen parameters, and training and validation statistics at each epoch. The exact output might change depending on the chosen parameters and independent EGG development, but it will always contain information about the loss and accuracy (proportion of successful game rounds).

When the `printer_validation_events` flag is passed, the script prints detailed information about the last validation pass. In particular, for all inputs in the validation set, the script prints the following lists: the inputs, the corresponding gold labels, the messages produced by Sender and the outputs of Receiver. The exact nature of each of these lists will change depending on game type and other parameters, but the following considerations hold in general:
* **INPUTS** -- The input items are printed in one-hot vector format. In discrimination games, only Sender inputs (that is, the target items) are printed.
* **LABELS** -- In discrimination games, these are the same indices of target location present in the input file. In recognition games, they are identical to the input items (in the original input format).
* **MESSAGES** -- For technical reasons, these are represented in integer format when using Reinforce (with 0 as `<eos>` delimiter) and as one-hot vectors for Gumbel-Softmax (with the 0-th position denoting `<eos>`). Note that, at validation/test time, any symbol following the first occurrence of `<eos>` can be ignored.
* **OUTPUTS** -- These are the non-normalized (pre-softmax) scores produced by the Receiver. In reconstruction games, each output will be the concatenation of the distributions over the values of each input attribute: for example, if inputs are two-attribute vectors with 5 possible values, each output item will be a list of 10 numbers, the first half to be interpreted as a non-normalized probability distribution over the values of the first attribute, and the second half as the same for the second attribute. In discrimination games, the output represents non-normalized probabilities for the possible position of the target in the input item array. When training with Gumbel-Softmax, this distribution is printed for each symbol produced by Sender. The one that is taken as the effective Receiver output for evaluation purposes is the distribution emitted in correspondance to the first `<eos>` emitted by Sender.
 
