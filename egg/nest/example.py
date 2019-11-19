# not necessary, but could be useful
def dict2string(d):
    s = []

    for k, v in d.items():
        if type(v) in (int, float):
            s.append(f"--{k}={v}")
        elif type(v) is bool and v:
            s.append(f"--{k}")
        elif type(v) is str:
            assert '"' not in v, f"Key {k} has string value {v} which contains forbidden quotes."
            s.append(f'--{k}={v}')
        else:
            raise Exception(
                f"Key {k} has value {v} of unsupported type {type(v)}.")
    return s

# only grid() is called from nest


def grid():
    """
    Should return an iterable of the parameter strings, e.g.
    `--param1=value1 --param2`
    """
    for random_seed in range(4):
      for vocab_size in [10, 15]:
            params = dict(vocab_size=vocab_size, random_seed=random_seed, n_epoch=15, batch_size=256)
            yield dict2string(params)

      for vocab_size in [16, 32]:
            params = dict(vocab_size=vocab_size, random_seed=random_seed, n_epoch=150, batch_size=128)
            yield dict2string(params)
