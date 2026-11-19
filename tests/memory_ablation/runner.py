"""New entry point with explicit deterministic inputs and original owned lease observer."""

import sys

import pytest

from tests.research_task_binding_increment.observer import Observer as OriginalObserver
from tests.research_task_binding_r01.diagnostics import png
from tests.research_task_binding_r01.observer import export
from tests.research_task_completion import support

INPUTS = {
    "independent-0": [1, 2, 3, 4, 5, 6],
    "independent-1": [0, 6, 5, 4, 3, 2],
    "query-held-out": [1, 9, 8, 7, 6, 5],
}


def fixed_input(identity):
    label = identity.split(":ASSET:", 1)[0]
    return png(bytes(INPUTS[label]))


class Observer(OriginalObserver):
    def pytest_sessionstart(self, session):
        super().pytest_sessionstart(session)
        patch = pytest.MonkeyPatch()
        session.config.add_cleanup(patch.undo)
        patch.setattr(support, "synthetic_input", fixed_input)
        export("input-recipe", INPUTS)


def main():
    observer = Observer()
    status = int(pytest.main(sys.argv[1:], plugins=[observer]))
    export("pytest-return", dict(direct_exit=status))
    observer.finish()
    return status


if __name__ == "__main__":
    sys.exit(main())
