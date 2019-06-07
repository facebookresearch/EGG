# OBJECTS GAME

*objects_game* is a simple Sender/Receiver game where Sender sees a target vector of discrete properties
-e.g. [1, 2, 3, 1] for a game with 4 dimensions and features
 in the [1, 3] range- and sends a variable length message
 to Receiver. Receiver, conditioned on that message, has to 
point to the right target among a set of distractors.

This game is a variant of [1], and we call it the object game because the vectors can be thought of as attribute-value descriptions of different objects.

A picture that describes the architecture of the game can be found [here](pics/archs.jpeg) (the no-image option is not currently implemented).

Example command:
```bash
python -m egg.zoo.objects_game.train \
    --perceptual_dimensions "[4, 4, 4, 4, 4]" \
    --vocab_size 1100 \
    --n_distractors 1 \
    --n_epoch 50 \
    --max_len 1  \
    --sender_lr 0.001 \
    --receiver_lr 0.001 \
    --batch_size 32 \
    --random_seed 111 \
    --data_seed 222 \
    --train_samples 1e6 \
    --validation_samples 1e4\
    --test_samples 1e3 \
    --evaluate \
    --dump_msg_folder '../messages' \
    --shuffle_train_data
```

The basic structure of this game is similar to the one of the [external_game](../external_game/README.md). Please refer to that for game configurations and training hyper-parameters. However, the input is different, as here it consists of discrete-valued vectors where the values that each dimension take can vary and can be specified as a parameter to the game. 
Additionally, the receiver is shown the target AND a fixed number of distractors and has to point to the position of the target among the randomly shuffled distractors.

Please refer to [this script](../../core/util.py) for a comprehensive list of not-game-specific parameters.

## Vector parameters:
Parameters to customize the default ([4, 4, 4, 4 ,4]) vectors for the game.

#### Data Generation
 * `--perceptual_dimensions` -- the prototype vector: the discrete values that each dimension can take. For instance if we set --perceptual_dimensions [3, 4, 5, 1], the game will be played with 4-dimensional vectors where the first dimension can be one of {1, 2, 3}, the second one of {1, 2, 3, 4} etc., leading to 3 * 4 * 5 * 1 = 60 distinct possible vectors. (default: [4, 4, 4, 4, 4])
 * `--n_distractors` -- number of additional vectors that the receiver will see. If the target vector (shown to Sender) [3, 4, 3] and n_distractors is 2, Receiver will have to choose the position of the target given the following input (where vector order is randomly determined): [ [4, 2, 5], [3, 4, 3], [1, 4, 2] ] (correct answer 2, second position). (default: 3)
 * `--train_samples` -- number of distinct tuples in the training set. A tuple is a unique combination of target+distractor(s). (default: 1e6)
 * `--validation_samples` -- number of distinct tuples in the validation set. (default: 1e4)
 * `--test_samples` -- number of distinct tuples in the test set. (default: 1e3)

#### Data Loading
Instead of generating train, validation and test splits at run time through the parameters listed in the previous section, an external data file can be loaded with the following argument:

 * `--load_data_path` -- path to a .npz data file containing splits in compressed numpy array format. 

When  `--load_data_path` is not set, the vectors populating the splits are generated according to the parameters described in the previous section. Such datasets can be saved in the correct format using `--dump_data_folder` (see below) and specifying the directory where to save them, and then loaded with `--load_data_path`.
Otherwise, a manually generated data file can be loaded as well.  It must be in .npz format and should have the following fields:
 * {train|valid|test} -- these fields contain the tensor with the tuples seen by the receiver. Each tensor has dimension split_size X n_distractors+1 X n_features. split_size is the total number of trials played by the agents in the {train|valid|test} phase. Note that the +1 term added to n_distractors is due to the presence of the target vector. n_features is simply the number of dimensions that each vector has, in the default case of [4, 4, 4, 4, 4] it would be 5. 
 * {train|valid|test}\_labels -- a 1-D array of size split_size that contains the idx of the target in the lineup of target+distractor(s) for each trial.
 * n_distractors -- the number of distractors used.
 
 Note that if `--load_data_path` is used, the `--{train|validation|test}\_samples` and `--n_distractors` command-line options will be ignored, as the relevant information will be inferred from the data file.
 
`--load_data_path` and `--perceptual_dimensions`are mutually exclusive. They cannot be used at the same time as it would not be clear if the game should be using the loaded data or generate new vectors and tuples according to the prototype specified through `--perceptual_dimensions`.

## Agents architecture
 * `--sender_hidden` -- Size of the hidden layer of Sender (default: 50)
 * `--receiver_hidden` -- Size of the hidden layer of Receiver (default: 50)
 * `--sender_embedding` -- Dimensionality of the embedding hidden layer for Sender (default: 10)
 * `--receiver_embedding` -- Dimensionality of the embedding hidden layer for Sender (default: 10)
 * `--sender_cell` -- Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)
 * `--receiver_cell` -- Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)
 
## Learning parameters
 * `--sender_lr` -- Learning rate for Sender's parameters (default: 1e-1)
 * `--receiver_lr` -- Learning rate for Receiver's parameters (default: 1e-1)
 * `--temperature` -- GS temperature for the sender (default: 1.0)

## Training parameters
 * `--shuffle_train_data` -- if set, training data will be shuffled before every epoch (default: False)
 * `--evaluate` -- if set, evaluation on the test will be performed (default: False) 
 * `--batch_size` -- number of samples (distinct tuples) in each mini-batch. (default: 32)
 * `--n_epochs` -- number of epoch in the training run. (default: 10)
 * `--validation_freq` -- the validation would be run every `validation_freq` epochs. (default: 1)

## Communication channel parameters
The communication channel can be tuned with the following parameters that are the same ones that can be used throughout all the games implemented in EGG so far. <br/>

 * `vocab_size` -- the number of unique symbols in the vocabulary (inluding `<eos>`!). `<eos>` is conventionally mapped to 0. (default: 10)
 * `max_len` -- the maximal length of the message. Receiver's output is processed either after the `<eos>` symbol is received
 or after `max_len` symbols, and further symbols are ignored. (default: 1)

## Seeding
 * `data_seed` -- a seed value to control the pseudo-random process that generates train, validation and test splits (for reproducibility purposes). (default: 111)
 * `random_seed` -- a seed value to control the pseudo-random process that initializes the game and network(s) parameters (for reproducibility purposes). (default: 
 
 Note that if you want to keep the data generation process stable, i.e, using the same datasets across training runs while varying the game network initialization, you can do so by keeping `--data_seed` fixed and varying `--random_seed`.

## Dumping
 * `--output_json` -- If set, EGG will output validation stats in json format (default: False)
 * `--dump_data_folder` -- path to folder where .npz file with dumped data will be created
 * `--dump_msg_folder` -- path to folder where a file with the output of the run will be saved
 
The name of the created file will be:
```
messages_{perceptual_dimensions}_vocab_{vocab_size}' \
                        '_maxlen_{max_len}_bsize_{batch_size}' \
                        '_n_distractors_{n_distractors}_train_size_{train_samples}' \
                        '_valid_size_{validation_samples}_test_size_{test_samples}' \
                        '_slr_{sender_lr}_rlr_{receiver_lr}_shidden_{sender_hidden}' \
                        '_rhidden_{receiver_hidden}_semb_{sender_embedding}' \
                        '_remb_{receiver_embedding}_mode_{mode}' \
                        '_scell_{sender_cell}_rcell_{receiver_cell}.msg
```

The file contains the list of arguments given to the command line parser as first line.
The format of the dumped output is as follows:

```
input_vector -> receiver_input -> sender_message -> receiver_output (label=gold_label_value)
```
where:
* `input_vector` is the input target vector, in comma-delimited format
* `receiver_input` is the input to Receiver, that is, the set of target+distractor vectors, delimited by colons (note that the vectors are in the same random order in which they are presented to Receiver, and the target is not explicitly marked)
* `sender_message` is the message sent by Sender, in comma-delimited format
* `receiver_output` is the label value produced by the Receiver
* `gold_label_value` is the gold label for the current input

Statistics about the overall number of unique messages and their distribution are printed at the end of the file.
 
## Debug
 * `debug` -- used to run egg with pdb (python debugger) enabled

## Help
Please run `python -m egg.zoo.objects_game.train -h` for a more comprehensive list of command line arguments

<br/>
<br/>

[1] *"EMERGENCE OF LINGUISTIC COMMUNICATION FROM REFERENTIAL GAMES WITH SYMBOLIC AND PIXEL INPUT"*, Angeliki Lazaridou, Karl Moritz Hermann, Karl Tuyls, Stephen Clark 
[[arxiv]](https://arxiv.org/pdf/1804.03984.pdf)
