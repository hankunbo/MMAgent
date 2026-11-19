"""Fork-only synthetic installation; never a deployment recipe for real cases."""

factory = None


def build():
    if factory is None:
        raise ValueError("synthetic_fixture_not_installed")
    return factory()
