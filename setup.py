# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="EGG",
    version="0.1.0",
    url="https://github.com/facebookresearch/EGG",
    author="Eugene Kharitonov",
    author_email="kharitonov@fb.com",
    description="Emergence of lanGuage in Games (EGG): A toolbox for language games research",
    packages=find_packages(),
    install_requires=requirements,
)
