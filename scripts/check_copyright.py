# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from https://github.com/marcofavorito/python-project-template/blob/main/scripts/check_copyright.py

"""
This script checks that all the Python files of the repository have the copyright notice.

In particular:
- (optional) the Python shebang
- (optional) the encoding header;
- the copyright and license notices;

It is assumed the script is run from the repository root.
"""

import itertools
import re
import sys
from pathlib import Path

HEADER_REGEX = r"""(#!/usr/bin/env python3
# -\*- coding: utf-8 -\*-
)?# Copyright \(c\) Facebook, Inc\. and its affiliates\.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree\.
"""
COMPILED_HEADER_REGEX = re.compile(HEADER_REGEX)


def check_copyright(file: Path) -> bool:
    """
    Given a file, check if the header is present.

    Return True if the files has the encoding header and the copyright notice,
    optionally prefixed by the shebang. Return False otherwise.

    :param file: file to inspect.
    :return True if the file is compliant with the checks, False otherwise.
    """
    return COMPILED_HEADER_REGEX.search(file.read_text()) is not None


if __name__ == "__main__":
    exclude_files = {
        Path("egg/zoo/emcom_as_ssl/LARC.py")
    }  # add any file (as a Path object) to exclude in this set
    python_files = filter(
        lambda x: x not in exclude_files,
        itertools.chain(
            Path("egg").glob("**/*.py"),
            Path("tests").glob("**/*.py"),
            Path("scripts").glob("**/*.py"),
        ),
    )

    bad_files = [filepath for filepath in python_files if not check_copyright(filepath)]

    if len(bad_files) > 0:
        print("The following files are not well formatted:")
        print(*bad_files, sep="\n")
        sys.exit(1)
    else:
        print("OK")
