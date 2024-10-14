python3 train.py --n_distractors 4 --n_samples 30 --n_epochs 30000 --vocab_size 100 --max_len 10 \
  --batch_size 32 --sender_lr 1e-2 --receiver_lr 1e-2 --lr_scheduler linear \
  --sender_hidden 25 --receiver_hidden 25 \ 
  --evaluate --output_json --mode rf 
