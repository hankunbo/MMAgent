"""Load shared database ownership before collecting either database or graph tests."""

pytest_plugins = ("tests.session_temp", "tests.database.pytest_plugin")
