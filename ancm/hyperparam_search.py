import os
import json
import math
import random
import subprocess
import types
from colorama import Fore
from colorama import init as colorama_init

from train import main
import argparse

from rich.progress import (
    Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
)
from bayes_opt import BayesianOptimization
from bayes_opt.event import DEFAULT_EVENTS, Events
from bayes_opt.logger import JSONLogger, ScreenLogger


pbounds = {
    'slr': (2, 4),              # transformed to (10^-2, 10^-4)
    'rlr_multiplier': (-1, 1),  # transformed to (0.1, 10)
    'vocab_size': (1, 2),       # transformed to (10, 100)
    'hidden_units': (3, 6),     # transformed to (8, 64)
    'length_cost': (1, 6),      # transformed to (10^-1, 10^-6)
}

init_points = 8
n_iter = 32

run_count = 1
seed = 42


ScreenLogger._default_cell_size = 16


def transform(params):  
    output_dict = dict(params)

    # maps  (2, 4) to (10^-2, 10^-4)
    output_dict['slr'] = math.exp(-params['slr'] * math.log(10)) 

    # maps (-1, 1) to (0.1, 10)
    output_dict['rlr_multiplier'] = math.exp(params['rlr_multiplier'] * math.log(10))

    # maps (1, 2) to (10, 100), returns an integer
    output_dict['vocab_size'] = int(math.exp(params['vocab_size'] * math.log(10)))

    # maps (3, 6) to (8, 64), returns an integer
    output_dict['hidden_units'] = int(math.exp(params['hidden_units'] * math.log(2)))

    # maps (1, 6) to (10^-1, 10^-6)
    output_dict['length_cost'] = math.exp(-params['length_cost'] * math.log(10))

    return output_dict


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


def transform_update_json(self, event, instance):
    if event == Events.OPTIMIZATION_STEP:
        data = dict(instance.res[-1])

        now, time_elapsed, time_delta = self._time_metrics()
        data["datetime"] = {"datetime": now, "elapsed": time_elapsed, "delta": time_delta}

        if "allowed" in data:  # fix: github.com/fmfn/BayesianOptimization/issues/361
            data["allowed"] = bool(data["allowed"])

        if "constraint" in data and isinstance(data["constraint"], np.ndarray):
            data["constraint"] = data["constraint"].tolist()

        data['params'] = transform(data['params'])
        data['params']['rlr'] = data['params']['slr'] * data['params']['rlr_multiplier']

        with open(self._path, 'a') as f:
            f.write(json.dumps(data) + "\n")


def transform_step_screen(self, instance, colour = Fore.RESET):
    res: dict[str, Any] = instance.res[-1]
    keys: list[str] = instance.space.keys
    cells: list[str | None] = [None] * (3 + len(keys))

    cells[:2] = \
        ScreenLogger._format_number(self, self._iterations + 1), \
        ScreenLogger._format_number(self, res["target"])
    if self._is_constrained:
        cells[2] = self._format_bool(res["allowed"])
    params = res.get("params", {})
    params = transform(params)
    cells[3:] = [ScreenLogger._format_number(self, params.get(key, float("nan"))) for key in keys]

    return "| " + " | ".join([x for x in cells if x is not None]) + " |"


def game(slr, rlr, vocab_size, hidden_units, length_cost, run_id):

    global seed

    assert type(vocab_size) == int
    assert type(hidden_units) == int

    random_number = random.randint(0, 1000)

    opts = [
        f'--vocab_size {vocab_size}',
        f'--sender_lr {slr}',
        f'--receiver_lr {rlr}',
        f'--erasure_pr 0.0',
        f'--length_cost {length_cost}',
        f'--sender_hidden {hidden_units}',
        f'--receiver_hidden {hidden_units}',
        f'--random_seed {random_number}',
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
        + [o for opt in opts[:-1] for o in opt.split()]
        + ['--load_data_path', 
           'data/input_data/[4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]_4_' \
           'distractors.npz'])
    exitcode = process.wait()

    with open(f'search/{run_id}-results.json') as fp:
        results = json.load(fp)

    return results['results']['accuracy']


def func(slr, rlr_multiplier, vocab_size, hidden_units, length_cost):
    global run_count
    params = {
        'slr': slr,
        'rlr_multiplier': rlr_multiplier,
        'vocab_size': vocab_size,
        'hidden_units': hidden_units,
        'length_cost': length_cost}
    params = transform(params)

    acc =  game(
        params['slr'],
        params['slr'] * params['rlr_multiplier'],
        params['vocab_size'],
        params['hidden_units'],
        params['length_cost'],
        run_count)

    run_count += 1
    return acc


def main():
    os.makedirs('search', exist_ok=True)
    random.seed(seed)

    optimizer = BayesianOptimization(
        f=func,
        pbounds=pbounds,
        random_state=10,
        allow_duplicate_points=True,
        verbose=2)

    optimizer.set_gp_params(alpha=1e-3, n_restarts_optimizer=5)

    # override _step/update methods for both logger instances
    logger_screen = ScreenLogger(verbose=2, is_constrained=False) 
    logger_screen._step = types.MethodType(transform_step_screen, logger_screen)

    results_json = JSONLogger(path='search/search_results.json') 
    results_json.update = types.MethodType(transform_update_json, results_json)

    # logger for saving values without transformation (for resuming the search)
    logger_json = JSONLogger(path='search/search_log.json')

    # create an Observer instance with both loggers
    observer = Observer(total=init_points+n_iter, loggers=[logger_json, results_json, logger_screen])

    for event in DEFAULT_EVENTS:
        optimizer.subscribe(
            event=event,
            subscriber=observer,
            callback=None)

    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    observer.progress.stop()


if __name__ == '__main__':
    main()
