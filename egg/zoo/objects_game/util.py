# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

def compute_binomial(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke.
    See http://stackoverflow.com/questions/3025162/statistics-combinations-in-python
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


def compute_baseline_accuracy(num_dist, symbols, *dims):
    final = []
    for num_dim in dims:
        result  = 0
        for j in range(num_dist+1):
            probability = 1 / (j+1)
            number_of_equal_dist = compute_binomial(num_dist, j)
            equal_dist = (1 / num_dim) ** (symbols * j)
            diff_dist = ((num_dim - 1) / num_dim) ** (symbols * (num_dist - j))
            result += probability * number_of_equal_dist * equal_dist * diff_dist
        final.append(result)

    return [round(elem, 4) * 100 for elem in final]


def entropy_dict(freq_table):
    H = 0
    n = sum(v for v in freq_table.values())

    for m, freq in freq_table.items():
        p = freq_table[m] / n
        H += -p * np.log(p)
    return H / np.log(2)


def entropy(elems):
    from collections import defaultdict

    freq_table = defaultdict(float)

    for m in elems:
        m = _hashable_tensor(m)
        freq_table[m] += 1.0

    return entropy_dict(freq_table)


def _hashable_tensor(t):
    if isinstance(t, tuple):
        return t
    if isinstance(t, int):
        return t

    try:
        t = t.item()
    except:
        t = tuple(t.view(-1).tolist())
    return t


def mutual_info(xs, ys):
    e_x = entropy(xs)
    e_y = entropy(ys)

    xys = []

    for x, y in zip(xs, ys):
        xy = (_hashable_tensor(x), _hashable_tensor(y))
        xys.append(xy)

    e_xy = entropy(xys)

    return e_x + e_y - e_xy


def compute_mi_input_msgs(sender_inputs, messages):
    num_dimensions = len(sender_inputs[0])
    each_dim = [[] for _ in range(num_dimensions)]
    result = []
    for i, _ in enumerate(each_dim):
        for vector in sender_inputs:
            each_dim[i].append(vector[i]) # only works for 1-D sender inputs

    for i, dim_list in enumerate(each_dim):
        result.append(round(mutual_info(messages,  dim_list), 4))

    print(f'| Entropy for each dimension of the input vectors = {[entropy(elem) for elem in each_dim]}')
    print(f'| H(msg) = {entropy(messages)}')
    print(f'| MI = {result}')
