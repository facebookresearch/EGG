This code implements the experiments reported in the following paper:
* _Compositionality and Generalization in Emergent Languages._ Rahma Chaabouni, Eugene Kharitonov, Diane Bouchacourt, Emmanuel Dupoux, Marco Baroni. ACL 2020. [[arxiv]](https://arxiv.org/abs/2004.09124)

The game can be run as follows:

```bash
python -m egg.zoo.compo_vs_generalization.train  --n_values=3 --n_attributes=5 --vocab_size=200 --max_len=2 --batch_size=5120 --sender_cell=lstm --receiver_cell=lstm --random_seed=1
```
Please refer to the paper to the details of the game. The hyperparameters used in the paper are provided in `./hyperparams`.


The game accepts the following game-specific parameters:
 * `max_len` -- the length of the messages.
 * `vocab_size` -- the number of unique symbols in the vocabulary
 * `sender_cell/receiver_cell` -- the cells used by the agents; can be any of {rnn, gru, lstm}
 * `n_attributes`/`n_values` -- the number of attributes/values in the attribute/value world
 * `sender_emb/receiver_emb` -- the size of the embeddings for Sender and Receiver
 * `sender_entropy_coeff` -- the regularisation coefficients for the
 entropy term in the loss, used to encourage exploration in Reinforce (only for Sender)
 * `sender_hidden/receiver_hidden` -- the size of the hidden layers for the cells
 
# Reproducibility
If you want to recover results maximally close to those reported in the paper, please use EGG v1.0. This can be done by running the following command:
```bash
git checkout v1.0
```
In later versions of EGG, some metrics are aggregated differently, which might lead to small discrepancies.
