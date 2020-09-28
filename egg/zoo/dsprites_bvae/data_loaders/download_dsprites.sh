#! /bin/sh
 
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

mkdir data
cd data
git clone https://github.com/deepmind/dsprites-dataset.git
cd dsprites-dataset
rm -rf .git* *.md LICENSE *.ipynb *.gif *.hdf5
