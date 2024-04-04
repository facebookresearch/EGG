import json
import sys

import pytest

from egg.nest.wrappers import ConcurrentWrapper


def dummy_runnable(args):
    print("Running dummy_runnable")
    print(json.dumps(args), file=sys.stderr)


def test_file_descriptor_closure(tmp_path):
    """
    Test to check if file descriptors are closed.
    Attempting to write to a closed file should raise a ValueError
    """
    runnable = dummy_runnable
    log_dir = tmp_path
    job_id = 1

    wrapper = ConcurrentWrapper(runnable, log_dir, job_id)
    wrapper({"key": "value"})

    with pytest.raises(ValueError):
        wrapper.stdout.write("This should fail if the file is closed.")

    with pytest.raises(ValueError):
        wrapper.stderr.write("This should fail if the file is closed.")


def test_stdout_stderr_restoration(tmp_path):
    """Test to ensure sys.stdout and sys.stderr are restored"""
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    runnable = dummy_runnable
    log_dir = tmp_path
    job_id = 2

    wrapper = ConcurrentWrapper(runnable, log_dir, job_id)
    wrapper({"another_key": "another_value"})

    assert sys.stdout == original_stdout
    assert sys.stderr == original_stderr
