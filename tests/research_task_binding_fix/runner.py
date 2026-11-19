"""Original pytest exit, independent of collector completeness."""

import sys

import pytest

from tests.research_task_binding_fix.observer import Observer
from tests.research_task_binding_r01.observer import export


def main():
    observer = Observer()
    status = int(pytest.main(sys.argv[1:], plugins=[observer]))
    export("pytest-return", dict(direct_exit=status))
    observer.finish()
    return status


if __name__ == "__main__":
    sys.exit(main())
