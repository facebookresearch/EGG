This code illustrates that (naive) compositionality of an emergent language might be not connected to the agents' performance. In particular, it is not the case that having a perfectly compositional language would necessarily result in a fast language acquisition or a good generalization on a downstream task.

For more details, refer to _Eugene Kharitonov and Marco Baroni._ Emergent Language Generalization and Acquisition Speed are not tied to Compositionality." [arXiv](https://arxiv.org/abs/2004.03420).

`discrete.py` and `continuous.py` implement the first and the second experiments from the text. `hypergrids/` contains the hyperparameters and command-line arguments to obtain the reported results.

Example commands for running:

* the discrete experiment:
```bash

python -m egg.zoo.compositional_efficiency.discrete --language=identity --loss_type=autoenc --random_seed=1 \
  --receiver_cell=lstm --receiver_layers=0 --cell_layers=1  --receiver_hidden=100 --receiver_emb=50 \
  --vocab_size=50 --batch_size=32 --n_a=2 --n_v=31 --n_epochs=500 
```

* the continuous experiment
```bash
python -m egg.zoo.compositional_efficiency.continuous --vocab_size=100 --batch_size=32 \
  --n_epochs=100 --random_seed=2 --receiver_hidden=100 --receiver_emb=50 --receiver_cell=lstm \
  --receiver_layers=0 --lenses=1 --lr=1e-3
```


# Reproducibility
If you want to recover results maximally close to those reported in the paper, please use EGG v1.0. This can be done by running the following command:
```bash
git checkout v1.0
```
In later versions of EGG, some metrics are aggregated differently, which might lead to small discrepancies.
