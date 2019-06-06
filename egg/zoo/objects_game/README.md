# OBJECTS GAME

*objects_game* is a simple Sender/Receiver game where Sender sees a target vector of discrete properties
-e.g. [1, 2, 3, 0] for a game with 4 dimensions and features
 in the [0, 3] range- and sends a variable length message
 to Receiver. Receiver, conditioned on that message, has to 
point to the right target among a set of distractors.

This game is a variant of [1].

A picture that describes the architecture of the game can be found [here](pics/archs.jpeg)

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
 * `--perceptual_dimension` -- the prototypcal vector: the discrete values that each dimension can take. For instance if we set --perceptual_dimension [3, 4, 5, 1], the game will be played with 4-dimensional vectors where the first dimension can be one of {1, 2, 3}, the second one of {1, 2, 3, 4} etc., leading to 3 * 4 * 5 * 1 = 60 distinct possible vectors. (default: [4, 4, 4, 4, 4])
 * `--n_distractors` -- number of additional vectors that the receiver will see. If the target vector (shown to Sender) [3, 4, 3] and n_distractors is 2, Receiver will have to choose the position of the target given the following input (where vector order is randomly determined): [ [4, 2, 5], [3, 4, 3], [1, 4, 2] ] (correct answer 2, second position). (default: 3)
 * `--train_samples` -- number of distinct tuples in the training set. A tuple is a unique combination of target+distractor(s). (default: 1e6)
 * `--validation_samples` -- number of distinct tuples in the validation set. (default: 1e4)
 * `--test_samples` -- number of distinct tuples in the test set. (default: 1e3)

#### Data Loading
 * `--load_data_path` -- path to .npz data file containing splits in numpy array format. Each split has dimension of `split_size X n_distracotrs X n_features`.

n_features is simply the number of dimensions that each vector has, in the default case of [4, 4, 4, 4, 4] it will be 5.`

The file can be manually created but it's safer to use the dumped file created when `--dump_data_folder` is set.
If manually created it should contain train, validation and test split in the described format plus the prototypical vector of perceptual_dimensions used in the splits (e.g. [3, 3, 3] if the splits only contains 3-D vectors with each dimension taking values in the range [1, 3] ), and the number of distractors seen by the Receiver. 
 
 *Note* that if `--load_data_path` is used {train|validation|test}_samples and n_distractors can be set but they will be ignored as they will be inferred from the data file.
 
 `--load_data_path` is a mutual exclusive argument with `--perceptual_dimensions`. They cannot be used at the same time as it would not be clear if the game should be using the loaded data or generate new vectors and tuples according to the prototypical one.

A sample data file is provided [here](sample_data).


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
The communication channel can be tuned with the following parameters that are the same ones that can be used used throughout all the games implemented in EGG so far. <br/>

 * `vocab_size` -- the number of unique symbols in the vocabulary (inluding `<eos>`!). `<eos>` is conventionally mapped to 0. (default: 10)
 * `max_len` -- the maximal length of the message. Receiver's output is processed either after the `<eos>` symbol is received
 or after `max_len` symbols, and further symbols are ignored. (default: 1)

## Seeding
 * `data_seed` -- a seed value to control the pseudo-random process that generate train, validation and test splits (for reproducibility purposes). (default: 111)
 * `random_seed` -- a seed value to control the pseudo-random process that initializes the game and network(s) parameters (for reproducibility purposes). (default: 
 
 `Note` that if you want to keep the data_generation process stable *i.e* having the same datasets across training run while varying the game network initialization you could so by keeping `--data_seed` fixed and vary `--random_seed`

## Dumping
 * `--output_json` -- If set, egg will output validation stats in json format (default: False)
 * `--dump_data_folder` -- path to folder where .npz file with dumped data will be created
 * `--dump_msg_folder` -- path to folder where a file with the output of the model will be saved
 
The name of the file created will be:
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

The file contain the namespace (list of arguments given to the command line parser) as the first line.
The format of the dumped output is as follows:

```
input_vector -> sender_message -> receiver_output (label=gold_label_value)
```
where:
* `input_vector` is the input vector, in comma-delimited format
* `sender_message` is the message sent by the Sender, in comma-delimited format
* `receiver_output` is the label value produced by the Receiver
* `gold_label_value` is the gold label for the current input

At the end of the file the number of unique messages with their occurrences will be printed.
 
## Debug
 * `debug` -- used to run egg with pdb (python debugger) enabled

## Help
Please run `python -m egg.zoo.objects_game.train -h` for a more comprehensive list of command line arguments

<br/>
<br/>

[1] *"EMERGENCE OF LINGUISTIC COMMUNICATION FROM REFERENTIAL GAMES WITH SYMBOLIC AND PIXEL INPUT"*, Angeliki Lazaridou, Karl Moritz Hermann, Karl Tuyls, Stephen Clark 
[[arxiv]](https://arxiv.org/pdf/1804.03984.pdf)
