## Introduction

Many language emergence games use a *reconstruction* or a *discrimination* task. In the reconstruction task, Sender gets an input item, it sends a message to Receiver, and Receiver must generate an output item identical to Sender's input. In the discrimination task, Sender gets an input item (the *target*); Receiver gets multiple input items (the same target and a number of *distractors*). Sender sends a message to Receiver, and Receiver has to output the location of the target in the array of its inputs.

The `play.py` script in this directory implements both types of tasks, as they share much of the same infrastructure. In particular, we consider the case here when inputs are vectors of discrete elements (interpretable as lists of categorical attribute-value pairs), and we let the user pass these inputs through text files.

Both the main and the auxiliary scripts are thoroughly commented, as we hope this can serve as a starting points to acquaint yourself with EGG, and start personalizing it for your purposes.

## Reconstruction game

The reconstruction game reads input from files that have an input item (attribute-value list) on each line, as in this example file ADD!!!!. NB: although values are represented by integers, they are treated as categorical (that is, a value of 3 is as different from 4 as it is from 45).

Here is an example of how to run the reconstruction game (note that we do not need to specify that we are playing the reconstruction game, as this is the default):

```bash
python -m egg.zoo.basic_games.play --mode 'gs' --train_data "train_file.txt" --validation_data "valid_file.txt" --n_attributes 2 --n_values 10 --n_epochs 50 --batch_size 512 --validation_batch_size 10 --max_len 4 --vocab_size 100 --sender_hidden 256 --receiver_hidden 512 --sender_embedding 5 --receiver_embedding 30 --lr 0.01 --receiver_cell "gru" --sender_cell "gru" --random_seed 111 --print_validation_events
```

In this particular instance, the following parameters are invoked:
 * `max_len` -- the maximal length of the message. Receiver's output is checked either after `<eos>` symbol is received or after `max_len` symbols, where `<eos>`
 * `vocab_size` -- the number of unique symbols in the vocabulary (inluding `<eos>`!)
 * `sender_cell/receiver_cell` -- the types of the cells that are used by the agents; can be any of {rnn, gru, lstm}
 * `n_features` -- the dimensionality of the vectors that are auto-encoded
 * `n_hidden` -- the size of the hidden space for the RNN cells
 * `embed_dim` -- the size of the hidden space for the RNN cells
 * `sender_entropy_coeff/receiver_entropy_coeff` -- the regularisation coefficients for the
 entropy term in the loss; those are used to encourage exploration in Reinforce
 * `sender_hidden/receiver_hidden` -- the size of the hidden layers for the cells
 * `sender_lr/receiver_lr` -- the learning rates for the agents' parameters (it might be useful to have Sender's learning rate
 lower, as Receiver has to adjust to the changes in Sender)
 
