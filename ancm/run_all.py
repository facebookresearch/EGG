from train import main
from egg.core import init
import time
from datetime import timedelta
import subprocess

seeds = [42, 111, 123, 44, 100]
erasure_probs = [0. + 0.05 * i for i in range(6)]
max_lengths = [2, 5, 10]

slr = round(0.0007491411007735139, 5)
rlr = round(0.0014182508770318705, 5)
length_cost = round(0.0007661277251354441, 5)
vocab_size = 51
hidden_units = 58


t_start = time.monotonic()
for pr in erasure_probs:
    print("="*14, f"erasure_pr = {pr}".center(18), "="*14)
    for seed in seeds:
        print("–"*12, f"seed = {seed}".center(22), "–"*12)
        for max_len in max_lengths:
            print("-"*10, f"max_len = {max_len}".center(26), "-"*10)
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
 
training_time = timedelta(seconds=time.monotonic()-t_start)
sec_per_run = training_time.seconds / (len(seeds) * len(max_lengths) * len(erasure_probs))
minutes, seconds = divmod(sec_per_run, 60)

time_total = str(training_time).split('.', maxsplit=1)[0]
time_per_run = f'{int(minutes):02}:{int(seconds):02}'

print("Total training time:", time_total)
print("       Time per run:", time_per_run)

