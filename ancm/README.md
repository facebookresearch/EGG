# ANCM 2024 – final project

> Emergence of Linguistic Communication from Referential Games with Symbolic and Pixel Input


## TODO
- [ ] Run for more epochs...
- [x] Make sure Reinforce is implemented correctly
- [ ] Run GS for more epochs
- [ ] Reinforce: Gibbs sampling? How to implement it using the wrapper?
- [ ] Reinforce: length cost = 1e-2 by default.

## Training
```bash
run_commands/train.sh
```

```bash
python3 train.py --n_distractors 4 --n_samples 30 --n_epochs 100 --vocab_size 100 --max_len 10 \
  --batch_size 32 --sender_lr 1e-2 --receiver_lr 1e-2 --lr_scheduler 0.1 \
  --sender_hidden 50 --receiver_hidden 50 --evaluate --output_json --mode rf 
```

Training will automatically export the training data for a given number of distractors and samples (unless such dataset already exists in the `input_data/` directory).

* LR scheduler: if multiplier value is provided, LR will be linearly scaled, multiplier determines LR value in the last epoch. 

## Exporting the data

To get `data/visa.csv`:
```bash
python3 export_visa.py
```

To manually export a NPZ file to the `data/input_data/` directory:
```bash
python3 reformat_visa.py -d <num. distractors> -s <num. samples per concept>
```

A sample is a set of target concept + distractor concepts, sampled from the same category or the same categories (that can be changed by editing `reformat_visa.py`). Each concept can only be a target object in one of train/test/val, but it can be a distractor in any subset.



