import json
from hyperparam_search import transform

with open('search/search_log.json') as fp:
    lines = fp.read().split('\n')

    for i, line in enumerate(lines[:-1]):
        output = '    ' + str(i+1) + ' & '
        data = json.loads(line)
        output += f'{data["target"]:.2f} & '
        params = transform(data['params'])
        params = (
            f'\\lrate{{{params["slr"]:.5f}}}',
            f'\\lrate{{{params["slr"] * params["rlr_multiplier"]:.5f}}}',
            str(int(params['vocab_size'])),
            str(int(params['hidden_units'])),
            f'\\lrate{{{params["length_cost"] * params["lc_multiplier"]:.7f}}}' if params["lc_multiplier"] != 0 else '0',
        )
        output += ' & '.join(params)
        output += ' \\\\'
        print(output)


