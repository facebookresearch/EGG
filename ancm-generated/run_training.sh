python3 train.py --n_distractors 4 --sender_cell lstm --receiver_cell lstm --train_samples 1e5 \
  --n_epochs 10 --max_len 5 \
  --sender_hidden 50 --receiver_hidden 50 --sender_embedding 10 --receiver_embedding 10 \
  --sender_lr 1e-1 --receiver_lr 2e-2 --sender_entropy_coeff 0.01 --receiver_entropy_coeff 0.001 \
  --mode rf --evaluate --output_json --dump_data_folder data/ --dump_msg_folder runs/ --validation_freq 1
