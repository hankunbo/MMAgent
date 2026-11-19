"""Foreground local API or bounded original Worker. Stop service with Ctrl+C/SIGTERM."""

import argparse
import fcntl
import importlib
import json
import os
import signal
import socket
import stat
from contextlib import contextmanager
from pathlib import Path
from threading import Event

from apps.worker.service import Worker

from .runtime import Installation


def load_factory(name):
    # Only a trusted CLI operator can select Python assembly; HTTP cannot provide this.
    module, separator, member = name.partition(":")
    if (
        not separator
        or not member.isidentifier()
        or not module.startswith(("apps.", "tests.runtime_platform."))
    ):
        raise ValueError("platform_factory_invalid")
    result = getattr(importlib.import_module(module), member)()
    if not isinstance(result, Installation):
        raise ValueError("platform_installation_required")
    return result


def bind_local(path):
    path = Path(path).absolute()
    parent = path.parent
    if parent.is_symlink() or parent.resolve() != parent:
        raise ValueError("platform_socket_directory")
    info = parent.stat()
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("platform_private_directory_required")
    # A held lock fences startup and shutdown without trusting a stale PID.
    fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        lock_info = os.fstat(fd)
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or lock_info.st_uid != os.geteuid()
            or stat.S_IMODE(lock_info.st_mode) != 0o600
        ):
            raise ValueError("platform_private_lock_required")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists() or path.is_symlink():
            raise ValueError("platform_socket_exists_inspect_before_recovery")
        sock.bind(str(path))
        os.chmod(path, 0o600)
        sock.listen(128)
        return sock, fd, path
    except BaseException:
        sock.close()
        os.close(fd)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("serve", "worker", "inspect"))
    parser.add_argument("--factory", default="apps.platform.local:build")
    parser.add_argument("--socket")
    parser.add_argument("--run-id")
    parser.add_argument("--batch-key")
    parser.add_argument("--worker-id", default="local-worker")
    args = parser.parse_args()
    installation = load_factory(args.factory)
    if args.mode == "inspect":
        print(
            json.dumps(
                dict(
                    schema_version="platform-installation/1.0.0",
                    profile=installation.profile.model_dump(mode="json"),
                    installed_runs=list(installation.dependencies.planners),
                    dispatch_installed=installation.dependencies.dispatch is not None,
                    live_enabled=False,
                )
            )
        )
        return 0
    if args.mode == "worker":
        if not args.run_id or not args.batch_key or installation.dependencies.dispatch is None:
            raise ValueError("platform_worker_owner_required")
        stop = Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        value = Worker(installation.dependencies.dispatch, worker_id=args.worker_id).run_bounded(
            args.run_id,
            batch_key=args.batch_key,
            max_ticks=installation.profile.max_ticks,
            shutdown=stop,
        )
        print(
            json.dumps(
                dict(
                    schema_version="platform-worker-result/1.0.0",
                    run_id=args.run_id,
                    state=value.state if value else "not_found",
                    shutdown_requested=stop.is_set(),
                )
            )
        )
        return 0
    if not args.socket:
        raise ValueError("platform_socket_required")
    import uvicorn

    sock, fd, path = bind_local(args.socket)
    inode = path.stat().st_ino
    try:
        cfg = uvicorn.Config(
            installation.app(),
            workers=1,
            loop="asyncio",
            http="h11",
            ws="none",
            lifespan="off",
            access_log=False,
            log_config=None,
            log_level="critical",
            proxy_headers=False,
            timeout_keep_alive=installation.profile.keep_alive_seconds,
            timeout_graceful_shutdown=installation.profile.shutdown_seconds,
            limit_concurrency=installation.profile.concurrency,
        )

        class LocalServer(uvicorn.Server):
            @contextmanager
            def capture_signals(self):
                # Finish outer socket cleanup after graceful shutdown, without re-raising
                # SIGTERM under its restored default handler before the finally block.
                previous = {
                    sig: signal.signal(sig, self.handle_exit)
                    for sig in (signal.SIGINT, signal.SIGTERM)
                }
                try:
                    yield
                finally:
                    for sig, handler in previous.items():
                        signal.signal(sig, handler)

        LocalServer(cfg).run(sockets=[sock])
    finally:
        sock.close()
        if path.exists() and not path.is_symlink() and path.stat().st_ino == inode:
            path.unlink()
        os.close(fd)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        # Owner exceptions can include DSN or input; details stay with original controlled audit.
        print(json.dumps({"error": "platform_command_failed", "automatic_retry": False}))
        raise SystemExit(1) from None
