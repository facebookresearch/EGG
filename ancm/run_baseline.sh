python3 train.py --load_data_path "data/input_data/[4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]_4_distractors.npz" \
  --vocab_size 16 --n_epochs 10 --max_len 5 --length_cost 0.01 --erasure_pr 0. \
  --sender_cell lstm --receiver_cell lstm \
  --sender_hidden 50 --receiver_hidden 50 --sender_embedding 10 --receiver_embedding 10 \
  --sender_lr 1e-3 --receiver_lr 2e-4 --sender_entropy_coeff 0.01 --receiver_entropy_coeff 0.001 \
  --mode rf --evaluate --output_json --dump_data_folder data/input_data/ --dump_results_folder runs/ --filename baseline
