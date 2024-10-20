# ANCM 2024 – final project

> Emergence of Linguistic Communication from Referential Games with Symbolic and Pixel Input


## Setup
```bash
python3 -m pip install -r requirements.txt
cd .. && python3 -m pip install --editable egg/ && cd ancm/
```

## Training

To generate a dataset for a given set of perceptual dimensions:
```bash
python3 train.py --perceptual_dimensions '[4, 4, 4, 4, 4]' --n_distractors 4 \
  --vocab_size 12 --n_epochs 10 --max_len 5 --length_cost 0.001 \
  --train_samples 1e5 --sender_cell lstm --receiver_cell lstm \
  --sender_hidden 50 --receiver_hidden 50 --sender_embedding 10 --receiver_embedding 10 \
  --sender_lr 1e-3 --receiver_lr 2e-4 --sender_entropy_coeff 0.01 --receiver_entropy_coeff 0.001 \
  --mode rf --evaluate --output_json --dump_data_folder data/input_data/ --dump_results_folder runs/ --filename baseline
```

To use an existing NPZ dataset:
```bash
python3 train.py --load_input_data data/input_data/visa-5-200.npz --n_distractors 4 \
  --vocab_size 12 --n_epochs 15 --max_len 5 --length_cost 0.001 \
  --train_samples 1e5 --sender_cell lstm --receiver_cell lstm \
  --sender_hidden 50 --receiver_hidden 50 --sender_embedding 10 --receiver_embedding 10 \
  --sender_lr 1e-3 --receiver_lr 2e-4 --sender_entropy_coeff 0.01 --receiver_entropy_coeff 0.001 \
  --mode rf --evaluate --output_json --dump_data_folder data/ --dump_results_folder runs/ --filename baseline
```

## Data

Exporting VISA to CSV:
```bash
python3 data/export_visa.py
```

Exporting CSV to NPZ:
```
python3 reformat_visa.py -d <num. distractors> -s <num. samples> [-c <a category>]
```
A sample is a set of target concept + distractor concepts, sampled from the same category or the same categories (that can be changed by editing `reformat_visa.py`). 

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


## Results

Generated data:
- BASELINE: 
  - acc: 96.07
  - 402 unique messages for 636 unique objects 
  - MI: 8.22
  - alignment: 93.91 
  - training time: ~10 min (cezary)
