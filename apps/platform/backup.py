"""Quiesced local backup bundle; restoring never overwrites the source database."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from db import manage
from packages.object_store import checksum


def client(binary, target, *args):
    env = dict(os.environ)
    if target.password:
        env["PGPASSWORD"] = target.password
    result = subprocess.run(
        [
            manage._binary(binary),
            "-h",
            target.host,
            "-p",
            str(target.port),
            "-U",
            target.user,
            *args,
        ],
        env=env,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError("platform_backup_client_failed")


def inventory(root):
    root = Path(root)
    values = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError("backup_symlink_forbidden")
        if p.is_file():
            values[str(p.relative_to(root))] = checksum(p.read_bytes())
    return values


def backup(target, objects, output):
    """Operator must stop writers first; pg_dump does not atomically snapshot files."""
    objects, output = Path(objects), Path(output)
    if (
        objects.is_symlink()
        or not objects.is_dir()
        or output.exists()
        or output.resolve().is_relative_to(objects.resolve())
    ):
        raise ValueError("backup_paths")
    output.mkdir(mode=0o700)
    first = inventory(objects)
    client(
        "pg_dump",
        target,
        "-d",
        target.database,
        "--format=custom",
        "--file",
        str(output / "database.dump"),
    )
    shutil.copytree(objects, output / "objects")
    if inventory(objects) != first or inventory(output / "objects") != first:
        raise ValueError("backup_objects_changed")
    manifest = dict(
        schema_version="platform-backup/1.0.0",
        objects=first,
        database_checksum=checksum((output / "database.dump").read_bytes()),
        requires_quiesced_writers=True,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def restore_to_empty(target, bundle, object_destination):
    import psycopg

    bundle, object_destination = Path(bundle), Path(object_destination)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["schema_version"] != "platform-backup/1.0.0" or object_destination.exists():
        raise ValueError("restore_destination")
    if (
        checksum((bundle / "database.dump").read_bytes()) != manifest["database_checksum"]
        or inventory(bundle / "objects") != manifest["objects"]
    ):
        raise ValueError("backup_checksum")
    with psycopg.connect(
        host=target.host,
        port=target.port,
        dbname=target.database,
        user=target.user,
        password=target.password,
    ) as connection:
        count = connection.execute(
            "SELECT (SELECT count(*) FROM pg_namespace "
            "WHERE nspname NOT IN ('public','information_schema') AND nspname NOT LIKE 'pg_%') "
            "+ (SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname='public')"
        ).fetchone()[0]
        if count:
            raise ValueError("restore_requires_empty_database")
    # Roles must already exist in the isolated cluster; no --clean or DROP commands.
    client(
        "pg_restore",
        target,
        "-d",
        target.database,
        "--exit-on-error",
        "--single-transaction",
        str(bundle / "database.dump"),
    )
    shutil.copytree(bundle / "objects", object_destination)
    if inventory(object_destination) != manifest["objects"]:
        raise ValueError("restore_object_checksum")
    return manifest
