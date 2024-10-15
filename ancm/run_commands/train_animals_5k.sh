python3 train.py --n_distractors 4 --n_samples 30 --category animals --n_epochs 5000 --vocab_size 30 --max_len 10 \
  --batch_size 32 --sender_lr 1e-3 --receiver_lr 1e-3 --lr_decay 0.1 \
  --sender_hidden 16 --receiver_hidden 16 --evaluate --output_json --mode rf --seed 42
