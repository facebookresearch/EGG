# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import sys


def process_file(filename):
    with open(filename, 'r') as f:
        file_contents = ''.join(f.readlines())
    new_entry = {}
    z_dim = re.findall('z_dim=[0-9]*', file_contents)[0].split('=')[1]
    vocab_size = re.findall('vocab_size=[0-9]*', file_contents)[0].split('=')[1]
    beta = re.findall('beta=[0-9]*', file_contents)[0].split('=')[1]
    random_seed = re.findall('random_seed=[0-9]*', file_contents)[0].split('=')[1]
    new_entry['z_dim'] = int(z_dim)
    new_entry['vocab_size'] = int(vocab_size)
    new_entry['beta'] = int(beta)
    new_entry['random_seed'] = int(random_seed)
    topsim = []
    posdis = []
    for entry in re.findall('{"topsim": .*}', file_contents):
        topsim.append(eval(entry)['topsim'])
    for entry in re.findall('{"posdis": .*}', file_contents):
        posdis.append(eval(entry)['posdis'])
    new_entry['topsim'] = topsim
    new_entry['posdis'] = posdis
    return new_entry

def main(filepath):
    computed_result = []
    print(filepath)
    for filename in os.listdir(filepath):
        print('Processing file: {}'.format(filename))
        new_entry = process_file(os.path.join(filepath, filename))
        computed_result.append(new_entry)
    print(computed_result)

if __name__ == '__main__':
    main(sys.argv[1])