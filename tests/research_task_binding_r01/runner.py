"""Return original pytest exit; capture cleanup after pytest config cleanups finish."""

import sys

import pytest

from tests.research_task_binding_r01.observer import Observer, export


def main():
    observer = Observer()
    status = int(pytest.main(sys.argv[1:], plugins=[observer]))
    export("pytest-return", dict(direct_exit=status))
    observer.finish()
    return status


if __name__ == "__main__":
    sys.exit(main())
