`simple_autoenc` is a simple Sender/Receiver game where a pair of agents is trained to 
encode and decode (i.e. autoencode) a one-hot vector of a fixed dimension.

The communication is performed by mean of variable-length messages; the training is done by Reinforce.

The game can be run as follows:

```bash
python -m egg.zoo.simple_autoenc.train --vocab_size=3 --n_features=6 --n_epoch=50 --max_len=10 --batch_size=512 --random_seed=21
```

The game accepts the following game-specific parameters:
 * `max_len` -- the maximal length of the message. Receiver's output is checked either after `<eos>` symbol is received
 or after `max_len` symbols;
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
 
