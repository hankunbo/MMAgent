"""Versioned absolute-script launcher; fixed model semantics, no downloads."""

import sys
from pathlib import Path

_apps = Path(__file__).resolve().parent
# Absolute scripts expose sibling modules at top level. Remove only that path,
# preserving the interpreter's own stdlib/site-packages and qualified apps.*.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _apps]
sys.path.insert(0, str(_apps.parent))

from packages.fairclip_research.backend import main  # noqa: E402


def check_runtime():
    import importlib.metadata
    import json
    import platform
    import sysconfig

    import torch

    if (
        not Path(platform.__file__)
        .resolve()
        .is_relative_to(Path(sysconfig.get_path("stdlib")).resolve())
    ):
        raise RuntimeError("fairclip_stdlib_shadowing")
    print(
        json.dumps(
            dict(
                versions={n: importlib.metadata.version(n) for n in ("torch", "numpy", "pillow")},
                executable=sys.executable,
                platform_origin=platform.__file__,
                system=platform.system(),
                torch_imported=bool(torch.__version__),
                forward_count=0,
            )
        )
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--check-runtime"]:
        check_runtime()
    else:
        main()
