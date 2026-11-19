"""07 P2 composition on the explicit 09A baseline; no deployment or patient execution."""

import json
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psycopg

from apps.deterministic_case_pipeline import assemble_visible_pipeline
from db.knowledge.repository import KnowledgeRepository
from db.manage import ConnectionTarget
from packages.deterministic_pipeline.models import PipelineConfig
from packages.disease_packs import DiseasePackCatalog
from packages.object_store import LocalObjectStore

CHANNELS = ("runtime", "audit_writer", "audit_reader", "knowledge_reader")
ROOT = Path("/opt/data/test/MMAgent")
OBJECT_ROOTS = {
    "visible": ROOT / ".local/07-patient-v1/visible",
    "knowledge": ROOT / ".local/07-knowledge-v1",
}
OWNER = "07-p2-services-v1"


class AssemblyUnavailable(RuntimeError):
    """A bounded reason without connection credentials or patient content."""


def check_runtime_bridge(target: ConnectionTarget) -> None:
    """Verify the deployed 0010 port using a read-only Runtime connection."""
    with psycopg.connect(
        host=target.host,
        port=target.port,
        dbname=target.database,
        user=target.user,
        password=target.password,
        connect_timeout=3,
        options="-c default_transaction_read_only=on -c statement_timeout=5000",
    ) as connection:
        connection.execute("SET LOCAL ROLE mmagent_runtime_app")
        valid = connection.execute(
            "SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
            "WHERE n.nspname='runtime' AND p.proname IN "
            "('append_case_pipeline_event','transition_case_pipeline_status') "
            "AND pg_get_function_identity_arguments(p.oid)='document jsonb' "
            "AND pg_get_userbyid(p.proowner)='mmagent_runtime_owner' AND p.prosecdef "
            "AND has_function_privilege(current_user,p.oid,'EXECUTE')"
        ).fetchone()[0]
        if valid != 2:
            raise AssemblyUnavailable("runtime_0010_ports_unavailable")


def check_object_root(root: Path, purpose: str) -> None:
    """Never adopt old evidence, review packages, or symlinked object trees."""
    if root != OBJECT_ROOTS[purpose]:
        raise AssemblyUnavailable("object_root_not_authorized")
    for path in (root, *root.parents):
        if path.is_symlink():
            raise AssemblyUnavailable("object_root_symlink")
    for path in (root, root / "objects", root / "staging"):
        info = path.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise AssemblyUnavailable("object_root_ownership_or_mode")
    marker = root / "owner.json"
    info = marker.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o600
        or json.loads(marker.read_text()) != {"owner": OWNER, "purpose": purpose}
    ):
        raise AssemblyUnavailable("object_root_owner_marker")


class KnowledgeObjectReader:
    """Published services receive bounded reads, with no object mutation methods."""

    __slots__ = ("__store",)

    def __init__(self, store):
        self.__store = store

    def read(self, uri: str) -> bytes:
        return self.read_bounded(uri, 4 * 1024 * 1024)

    def read_bounded(self, uri: str, max_bytes: int) -> bytes:
        if not uri.startswith("object://knowledge/objects/"):
            raise ValueError("published_object_required")
        return self.__store.read_bounded(uri, max_bytes)


@dataclass(frozen=True)
class PatientServices:
    pipeline: object
    config: PipelineConfig

    def readiness(self) -> dict:
        """P2 cannot attest P1-B content or implement the missing P3 branch."""
        return {
            "service_assembly_available": True,
            "content_publication_ready": False,
            "patient_run_ready": False,
            "reasons": [
                "published_content_and_review_not_handed_off",
                "mapping_not_bound",
                "p3_demand_gap_ports_not_implemented",
            ],
        }


def assemble_patient_services(
    resolve_target: Callable[[str], ConnectionTarget],
) -> PatientServices:
    """Resolve only four service channels at the trusted connection boundary.

    The resolver is not retained. No evaluator path, preparer, migration identity,
    editor/reviewer/publisher, credential file, or whole P1-A package is injected.
    09E-R01 hands this consumer over to the exact 09A validator/configuration.
    Future content binding still needs a separate handoff.
    """
    try:
        for purpose, root in OBJECT_ROOTS.items():
            check_object_root(root, purpose)
        targets = {channel: resolve_target(channel) for channel in CHANNELS}
        if any(
            type(t) is not ConnectionTarget
            or (t.host, t.port, t.database) != ("127.0.0.1", 5432, "mmagent_06e_benchmark_v1")
            for t in targets.values()
        ) or len({t.user for t in targets.values()}) != len(CHANNELS):
            raise AssemblyUnavailable("service_target_or_identity_conflict")
        catalog = DiseasePackCatalog.from_directory(ROOT / "configs/disease_packs")
        config = PipelineConfig.model_validate_json(
            (ROOT / "configs/deterministic_pipeline/fairvlmed_gap.json").read_bytes()
        )
        # 09E-R01: only 09A's validator 1.0 -> 1.1 and config checksum changed.
        # Keep the complete Draft/empty-Mapping lock; this is no publication approval.
        if (
            config.checksum
            != "sha256:fe15b81fa397960d8a11999bbf23a90f8ea3ab986d0f136cf5772a9985cd66bd"
            or catalog.pack(config.pack_ref).status != "Draft"
            or config.mapping_refs
        ):
            raise AssemblyUnavailable("p2_baseline_changed_requires_new_handoff")
        check_runtime_bridge(targets["runtime"])
        reader = KnowledgeRepository(targets["knowledge_reader"], capability="reader")
        pipeline = assemble_visible_pipeline(
            runtime_target=targets["runtime"],
            audit_writer_target=targets["audit_writer"],
            audit_reader_target=targets["audit_reader"],
            knowledge_reader=reader,
            knowledge_store=KnowledgeObjectReader(LocalObjectStore(OBJECT_ROOTS["knowledge"])),
            visible_store=LocalObjectStore(OBJECT_ROOTS["visible"]),
            catalog=catalog,
            configs=(config,),
        )
        return PatientServices(pipeline, config)
    except AssemblyUnavailable:
        raise
    except Exception:
        raise AssemblyUnavailable("service_assembly_unavailable") from None
