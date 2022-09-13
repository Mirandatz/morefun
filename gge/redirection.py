import contextlib
import os
import sys
import typing


@contextlib.contextmanager
def discard_stderr_and_stdout() -> typing.Iterator[None]:
    copy_of_stdout = os.dup(sys.stdout.fileno())
    copy_of_stderr = os.dup(sys.stderr.fileno())
    with open(os.devnull, "w") as dev_null_file:
        sys.stdout.flush()
        os.dup2(dev_null_file.fileno(), sys.stdout.fileno())

        sys.stderr.flush()
        os.dup2(dev_null_file.fileno(), sys.stderr.fileno())
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(copy_of_stdout, sys.stdout.fileno())
            os.close(copy_of_stdout)

            sys.stderr.flush()
            os.dup2(copy_of_stderr, sys.stderr.fileno())
            os.close(copy_of_stderr)
