from train import main
from egg.core import init
import time
from datetime import timedelta
import subprocess

seeds = [42, 111, 123, 44, 100]
# erasure_probs = [0.00, 0.10, 0.15, 0.20, 0.25]
# max_lengths = [2, 5, 10]
erasure_probs = [0.15, 0.20, 0.25]
max_lengths = [5]

slr = round(0.0007491411007735139, 5)
rlr = round(0.0014182508770318705, 5)
length_cost = round(0.0007661277251354441, 5)
vocab_size = 51
hidden_units = 58


run_count = 1
t_start = time.monotonic()
for pr in erasure_probs:
    for seed in seeds:
        for max_len in max_lengths:
            print("="*3, f"{run_count} / {len(max_lengths) * len(erasure_probs) * len(seeds)}".center(10), "="*3)
            print(f"erasure_pr: {pr}")
            print(f"seed: {seed}")
            print(f"max_len: {max_len}")

            results_dir = f'runs/erasure_pr_{pr}/'
            filename = f'{max_len}_{seed}'
            opts = [
                f'--erasure_pr {pr}',
                f'--max_len {max_len}',
                f'--random_seed {seed}',
                f'--filename {filename}',
                f'--dump_results_folder {results_dir}',
                f'--vocab_size {vocab_size}',
                f'--sender_lr {slr}',
                f'--receiver_lr {rlr}',
                f'--length_cost {length_cost}',
                f'--sender_hidden {hidden_units}',
                f'--receiver_hidden {hidden_units}',
                '--sender_embedding 10',
                '--receiver_embedding 10',
                '--n_epochs 10',
                '--sender_entropy_coeff 0.01',
                '--receiver_entropy_coeff 0.001',
                '--sender_cell lstm',
                '--receiver_cell lstm',
                '--mode rf',
                '--evaluate',
                '--validation_freq 1',
                '--load_data_path ' \
                    '"data/input_data/[4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]_4_' \
                    'distractors.npz"']

            process = subprocess.Popen(
                ['python3', 'train.py']
                + [o for opt in opts[:-1] for o in opt.split()]
                + ['--load_data_path', 
                   'data/input_data/[4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]_4_' \
                   'distractors.npz'])
            exitcode = process.wait()

            elapsed = timedelta(seconds=time.monotonic()-t_start)
            elapsed_per_run = elapsed.seconds / run_count
            minutes, seconds = divmod(elapsed_per_run, 60)
            elapsed = str(elapsed).split('.', maxsplit=1)[0]
            elapsed_per_run = f'{int(minutes):02}:{int(seconds):02}'
            print(f"elapsed time: {elapsed} ({elapsed_per_run} per run)")
            print('')
            run_count += 1
 
training_time = timedelta(seconds=time.monotonic()-t_start)
sec_per_run = training_time.seconds / (len(seeds) * len(max_lengths) * len(erasure_probs))
minutes, seconds = divmod(sec_per_run, 60)

time_total = str(training_time).split('.', maxsplit=1)[0]
time_per_run = f'{int(minutes):02}:{int(seconds):02}'

print("Total training time:", time_total)
print("       Time per run:", time_per_run)

