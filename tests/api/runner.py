"""12A direct-exit runner; the original SessionDatabaseLease is the sole cleanup owner."""

import sys
from pathlib import Path

import pytest

from tests.tool_gap_queue.runner import Observer


def main():
    out = Path(sys.argv[1]).resolve()
    if not (out / "collector-owner.json").is_file():
        raise SystemExit("12A collector-owned output required")
    observer = Observer(out)
    status = int(pytest.main(sys.argv[2:], plugins=[observer]))
    cleanup = observer.finish(status)
    (out / "pytest.exit").write_text(str(status) + "\n")
    (out / "cleanup.exit").write_text(str(cleanup) + "\n")
    return status or cleanup


if __name__ == "__main__":
    sys.exit(main())
