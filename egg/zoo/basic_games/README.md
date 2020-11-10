## Introduction

Many language emergence games use a *reconstruction* or a *discrimination* task. In the reconstruction task, Sender gets an input item, it sends a message to Receiver, and Receiver must generate an output item identical to Sender's input. In the discrimination task, Sender gets an input item (the *target*); Receiver gets multiple input items (the same target and a number of *distractors*). Sender sends a message to Receiver, and Receiver has to output the location of the target in the array of its inputs.

The `play.py` script in this directory implements both types of tasks, as they share much of the same infrastructure. In particular, we consider the case here when inputs are vectors of discrete elements (interpretable as lists of categorical attribute-value pairs, where no value of an attribute is more or less similar to another), and we let the user pass these inputs through text files.

Both the main and the auxiliary scripts are thoroughly commented, as we hope this can serve as a starting points to acquaint yourself with EGG, and start personalizing it for your purposes.

## Reconstruction game

The reconstruction game reads input from files that have an input item (attribute-value list) on each line, as in this example file ADD!!!!. NB: although values are represented by integers, they are treated as categorical (that is, a value of 3 is as different from 4 as it is from 45).

Here is an example of how to run the reconstruction game (note that we do not need to specify that we are playing the reconstruction game, as this is the default):

```bash
python -m egg.zoo.basic_games.play --mode 'gs' --train_data "train_file.txt" --validation_data "valid_file.txt" --n_attributes 2 --n_values 10 --n_epochs 50 --batch_size 512 --validation_batch_size 1000 --max_len 4 --vocab_size 100 --sender_hidden 256 --receiver_hidden 512 --sender_embedding 5 --receiver_embedding 30 --receiver_cell "gru" --sender_cell "gru" --lr 0.01 --print_validation_events
```

In this particular instance, the following parameters are invoked:
 * `mode` -- tells whether to use reinforce (`rf`) or Gumbel-Softmax (`gs`) for training.
 * `train_data/validation_data` -- paths to the files containing training data and validation data (the latter used at each epoch to track the progress of training); both files are in the same format.
 * `n_attributes` -- this is the number of elements that the input file vectors have: for example, given this ADD file, `n_attributes` should be set to XXX.
 * `n_values` -- number of distinct values that each input file vector element can take: as we are counting from 0, if the maximum value is 3, `n_values` should be set to 4 (and 0 should be used as a possible values).
 * `n_epochs` -- how many times the data in the input training file should be traversed: note that they will be traversed in a different random order each time.
 * `batch_size` -- batch size for training data (can't be smaller than number of item items in training file).
 * `validation_batch_size` -- batch size for validation data, provided as a separate argument as it is often convenient to traverse the whole validation set in a single step.
 * `max_len` -- after `max_len` symbols without `<eos>` have been emitted by the Sender, an `<eos>` is forced; consequently, the longest possible message will contain `max_len` symbols, followed by `<eos>`.
 * `vocab_size` -- the number of unique symbols in the Sender vocabulary (inluding `<eos>`!).
 * `sender_hidden/receiver_hidden` -- the size of the hidden layers of the agents.
 * `sender_embedding/receiver_embedding` -- output dimensionality of the layer that embeds symbols produced at previous step by the Sender message-emitting/Receiver message-processing recurrent networks, respectively.
 * `sender_cell/receiver_cell` -- type of cell of recurrent networks agents use to emit/process the message.
 * `lr` -- learning rate.
 * `print_validation_events` -- if this flag is passed, the script will print the validation input, with the corresponding messages emitted by Sender and Receiver outputs, after the last epoch of training.
 
 To see all arguments that can be passed (and for some more information on the ones above), run:
 
 ```bash
python -m egg.zoo.basic_games.play -h
```
## Discrimination game

The discrimination game reads the input from files that have, on each line, a sequence of items (attribute-value lists), followed by the index of the target in this sequence (counting from 0), as in this example file ADD!!!!. Items are period-delimited, and values are space-delimited. As in the reconstruction game, although values are represented by integers, they are categorical, that is, for the purpose of the game, they are converted to a one-hot vector representation where there is not inherent similarity between numerically close values.

Here is an example of a discrimination game run:

```bash
python -m egg.zoo.basic_games.play --game_type 'discri' --mode 'rf' --train_data "discri_train_file.txt" --validation_data "discri_valid_file.txt" --n_values 10 --n_epochs 50 --batch_size 512 --validation_batch_size 10 --max_len 4 --vocab_size 100 --sender_hidden 256 --receiver_hidden 512 --sender_embedding 5 --receiver_embedding 30 --lr 0.01 --receiver_cell "gru" --sender_cell "gru" --random_seed 111 --print_validation_events
```

Most parameters were explained above, but notice here the following:
 * ...
 
## Output

The script will write output to STDOUT, including information on the chosen parameters, and training and validation statistics at each epoc. The exact output might change depending on the chosen parameters and independent EGG development, but it will always contain information about the loss and accuracy (proportion of successful game rounds). Whe the `printer_validation_events` flag is passed, the script also prints, for all inputs in the validation set, and as separate lists represented in text format: the inputs, the corresponding gold labels, the messages produced by the Sender and the outputs of the Receiver (after full training). The exact nature of each of these lists will change depending on game type and other parameters. In particular,  under Gumbel-Softmax training the script prints the outputs emitted by the ...
ONE-HOT!
 
