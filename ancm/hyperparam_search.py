import os
import json
import random
import subprocess

from train import main
import argparse

from tqdm import tqdm
from rich.progress import (
    Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
)
from bayes_opt import BayesianOptimization
from bayes_opt.event import DEFAULT_EVENTS, Events
from bayes_opt.logger import JSONLogger, ScreenLogger


pbounds = {
    'slr': (1e-4, 1e-2),
    'rlr_multiplier': (1e-1, 1e1),
    'vocab_size': (10, 50),
    'hidden_units': (10, 100),
    'length_cost': (0, 0.05),
}

init_points = 8
n_iter = 32

run_count = 1
seed = 42

ScreenLogger._default_cell_size = 16


class Observer:
    def __init__(self, total, loggers=None):
        self.loggers = loggers if isinstance(loggers, list) else list()
        self.progress = Progress(
            "[bold]{task.completed}/{task.total}",
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%", "•",
            TimeRemainingColumn(), "•",
            TimeElapsedColumn())
        self.p = self.progress.add_task("Searching", total=total)
        self.progress.start()

    def update(self, event, instance):
        if event == Events.OPTIMIZATION_STEP:
            self.progress.update(self.p, advance=1)

        for logger in self.loggers:
            logger.update(event, instance)


def game(sender_lr, receiver_lr_multiplier, vocab_size,
         hidden_units, length_cost, run_id):

    # global seed

    assert type(vocab_size) == int
    assert type(hidden_units) == int

    # random_number = random.randint(0, 1000)

    params = [
        f'--vocab_size {vocab_size}',
        f'--sender_lr {slr}',
        f'--receiver_lr {rlr_multiplier * sender_lr}',
        f'--erasure_pr 0.0',
        f'--length_cost {length_cost}',
        f'--sender_hidden {hidden_units}',
        f'--receiver_hidden {hidden_units}',
        f'--random_seed 42',  # {random_number}',
        f'--filename {run_id}',
        '--sender_embedding 10',
        '--receiver_embedding 10',
        '--dump_results_folder search/',
        '--n_epochs 10',
        '--max_len 5',
        '--sender_entropy_coeff 0.01',
        '--receiver_entropy_coeff 0.001',
        '--sender_cell lstm',
        '--receiver_cell lstm',
        '--mode rf',
        '--evaluate',
        '--simple_logging',
        '--silent',
        '--validation_freq -1',
        '--load_data_path ' \
            '"data/input_data/[4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]_4_' \
            'distractors.npz"']

    process = subprocess.Popen(
        ['python3', 'train.py']
        + [p for param in params[:-1] for p in param.split()]
        + ['--load_data_path', 
           'data/input_data/[4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]_4_' \
           'distractors.npz'])
    exitcode = process.wait()

    with open(f'search/{run_id}-results.json') as fp:
        results = json.load(fp)

    return results['results']['accuracy']


def func(sender_lr, receiver_lr_multiplier, vocab_size,
         hidden_units, length_cost):
    global run_count
    acc =  game(sender_lr, receiver_lr_multiplier, int(vocab_size),
                int(hidden_units), length_cost, run_count)
    run_count += 1
    return acc


def main():
    os.makedirs('search', exist_ok=True)
    # random.seed(seed)

    optimizer = BayesianOptimization(
        f=func,
        pbounds=pbounds,
        random_state=seed,
        allow_duplicate_points=False,
        verbose=2)

    optimizer.set_gp_params(alpha=1e-3, n_restarts_optimizer=5)

    logger_screen = ScreenLogger(verbose=2, is_constrained=False) 
    logger_json = JSONLogger(path='search/search_results.json') 
    observer = Observer(total=init_points+n_iter, loggers=[logger_screen, logger_json])

    for event in DEFAULT_EVENTS:
        optimizer.subscribe(
            event=event,
            subscriber=observer,
            callback=None)

    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    observer.progress.stop()


if __name__ == '__main__':
    main()
