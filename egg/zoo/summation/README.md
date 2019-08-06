`summation` is a simple Sender/Receiver game where a pair of agents is jointly trained to recognize the `a^n b^n` grammar.
Sender is presented with non-empty sequences of form {`a`, `aabb`, `abb`, ...}, sends a variable-length message to Receiver. 
In turn, for each sequence, Receiver has to output a binary indicator if the sequence has equal number of `a`s and `b`s.

Can you guess if Sender will actually count and just pass a decision to Receiver? Or will it simply reproduce the sequence
and it is Receiver who answers? Or they invent some in-between code? Does it depend on the parameters?

To get an answer, you can launch the game as follows:

```bash
python -m egg.zoo.summation.train --vocab_size=20 --max_n=10  --n_epoch=5 --max_len=10 --batch_size=32 --random_seed=21 --batches_per_epoch=100 --temperature=0.50 --sender_cell=lstm --receiver_cell=lstm --random_seed=21
```

The game first outputs training scores and then dumps the communication protocol for both agents:
```
inputs sequence -> message -> Receiever's output (correct label)
```
In this dump `a` and `b` are encoded as 1 and 2. The training is done via Gumbel-Softmax relaxation.

The game accepts the following game-specific parameters:
 * `max_n` -- the maximal value of `n` in `a^nb^n`. The maximal length of the input string will be `2 * n + 1`
 * `max_len` -- the maximal length of the message between Sender and Receiver
 * `vocab_size` -- the number of unique symbols in the communication vocabulary (inluding `<eos>`!)
 * `sender_cell/receiver_cell` -- the cells that are used by the agents; can be any of {rnn, gru, lstm}
 * `n_hidden` -- the size of the hidden space for the RNN cells
 * `embed_dim` -- the size of the hidden space for the RNN cells
 * `lr` -- learning rate
 
