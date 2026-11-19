"""Fixed 06 entrypoint; single SLO input, two fixed texts, one GPU, no dataset loader."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.fairclip_research.backend import main  # noqa: E402

if __name__ == "__main__":
    main()
