This code implements the experiments reported in the following paper:

* *Defending Compositionality in Emergent Languages*. Michal Auersperger, Pavel Pecina. NAACL SRW 2022. [[arxiv]](https://arxiv.org/abs/2206.04751)

and is an extension of the experiments in `egg/zoo/compo_vs_generalization`

A single experiment of the full communication game can be run as follows:

```bash
python -m egg.zoo.compo_vs_generalization_ood.train  --n_values=50 --n_attributes=2 --vocab_size=50 --max_len=3  --receiver=ModifReceiver --sender=ModifSender --hidden=50 --batch_size=64 --random_seed=1
```

To run an experiment with a single agent on half of the problem only (i.e., a *learning alone* experiment from the paper), run e.g.: 
```bash
python -m egg.zoo.compo_vs_generalization_ood.learning_alone.train  --n_values=50 --n_attributes=2 --vocab_size=50 --max_len=5 --archpart=sender --model=OrigSenderDeterministic --hidden=50 --batch_size=64 --random_seed=1 
```

See the article for further details.

To replicate the *full experiments*, use the hyperparameters in `./hyperparams/modified_arch.json` and `./hyperparams/orig_arch.json`.

To replicate the *learning alone experiments*, use the hyperparameters in `./hyperparams/learning_alone/receiver.json` and `./hyperparams/learning_alone/sender.json`.


For convenience, we attach the log files of our full experiment runs in `results/orig_arch/220520T115546` and `results/modified_arch/220517T231916`. The notebook `ntb-results-full.ipynb` processes the logs and produces Table 2 and Figure 1 from the paper.
