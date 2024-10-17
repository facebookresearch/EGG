python3 train.py --n_distractors 4 --n_samples 30 --n_epochs 1000 --vocab_size 100 --max_len 10 \
  --batch_size 32 --sender_lr 1e-1 --receiver_lr 1e-1 --lr_decay 0.05 \
  --sender_hidden 50 --receiver_hidden 50 --evaluate --output_json --seed 42 --mode gs 
