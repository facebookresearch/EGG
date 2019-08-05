`channel` is a Sender/Receiver game where a pair of agents is trained to 
encode and decode (i.e. autoencode) a one-hot vector of a fixed dimension and transmit it over channels with different
properties. The vectors to be auto-encoded might come from uniform, powerlaw or some specified
distribution. The communication is performed by mean of variable-length messages; the training is done by Reinforce/GS. 

This code was used in the experiments of the following paper: 
* _Anti-efficient encoding in emergent communication._ Rahma Chaabouni, Eugene Kharitonov, Emmanuel Dupoux, Marco Baroni.
[arxiv](https://arxiv.org/abs/1905.12561)

The game can be run as follows:

```bash
python -m egg.zoo.channel.train --vocab_size=3 --n_features=6 --n_epoch=50 --max_len=10 --batch_size=512 --random_seed=21
```

The game accepts the following game-specific parameters:
 * `max_len` -- the maximal length of the message. Receiver's output is checked either after `<eos>` symbol is received
 or after `max_len` symbols;
 * `vocab_size` -- the number of unique symbols in the vocabulary (inluding `<eos>`!)
 * `sender_cell/receiver_cell` -- the cells used by the agents; can be any of {rnn, gru, lstm}
 * `n_features` -- the dimensionality of the vectors that are auto-encoded
 * `n_hidden` -- the size of the hidden space for the RNN cells
 * `embed_dim` -- the size of the hidden space for the RNN cells
 * `sender_entropy_coeff/receiver_entropy_coeff` -- the regularisation coefficients for the
 entropy term in the loss, used to encourage exploration in Reinforce
 * `sender_hidden/receiver_hidden` -- the size of the hidden layers for the cells
 * `sender_lr/receiver_lr` -- the learning rates for the agents' parameters (it might be useful to have Sender's learning rate
 lower, as Receiver has to adjust to the changes in Sender)
 * `mode={gs/rf}` -- training either via GS or Reinforce
 * `probs={p1,p2,...}` or `'probs=powerlaw'` select the prior distribution over concepts
 
