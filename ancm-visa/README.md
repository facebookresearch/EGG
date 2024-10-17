# ANCM 2024 – final project

> Emergence of Linguistic Communication from Referential Games with Symbolic and Pixel Input


## TODO
- [x] Make sure Reinforce is implemented correctly
- [x] Single category data + training
- [ ] Allow resampling of distractors (e.g. **X** A B A C)? 
- [ ] Run GS for more epochs...
- [ ] Run Reinforce more epochs...
- [ ] Reinforce: Gibbs sampling? How to implement it using the wrapper?
- [ ] Reinforce: length cost = 1e-2 by default.

## Setup
```bash
python3 -m pip install -r requirements.txt
cd .. && python3 -m pip install --editable egg/ && cd ancm/
```

## Training
```bash
run_commands/train.sh
```

```bash
python3 train.py --n_distractors 4 --n_samples 30 --n_epochs 1000 --vocab_size 100 --max_len 10 \
  --batch_size 32 --sender_lr 1e-1 --receiver_lr 1e-1 --lr_decay 0.05 \
  --sender_hidden 50 --receiver_hidden 50 --evaluate --output_json --seed 42 --mode rf 
```

Training only on concepts belonging to one of the categories:
```bash
python3 train.py --n_distractors 4 --n_samples 30 --category animals --n_epochs 5000 --vocab_size 30 --max_len 10 \
  --batch_size 32 --sender_lr 1e-3 --receiver_lr 1e-3 --lr_decay 0.1 \
  --sender_hidden 16 --receiver_hidden 16 --evaluate --output_json --mode rf --seed 42
```

| **Category** | **Num. concepts**  |
|--------------|-------------------:|
| animals      | 135                |
| appliances   | 18                 |
| artefacts    | 37                 |
| clothing     | 40                 |
| container    | 11                 |
| device       | 8                  |
| food         | 59                 |
| home         | 49                 |
| instruments  | 19                 |
| material     | 4                  |
| plants       | 7                  |
| structures   | 26                 |
| tools        | 27                 |
| toys         | 3                  |
| vehicles     | 37                 |
| weapons      | 23                 |

Training will automatically export the training data for a given number of distractors and samples (unless such dataset already exists in the `input_data/` directory).

* LR scheduler: if multiplier value is provided, LR will be linearly scaled, multiplier determines LR value in the last epoch. 

## Exporting the data

To get CSV files:
```bash
python3 export_visa.py
```

To manually export a NPZ file to the `data/input_data/` directory:
```bash
python3 reformat_visa.py -d <num. distractors> -s <num. samples per concept> -c <category, optional>
```

A sample is a set of target concept + distractor concepts, sampled from the same category or the same categories (that can be changed by editing `reformat_visa.py`). Each concept can only be a target object in one of train/test/val, but it can be a distractor in any subset.



