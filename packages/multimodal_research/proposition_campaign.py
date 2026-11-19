"""One shared acquisition and separate original Runtime/Review/score per proposition.

This is a bounded orchestration manifest, not a new executor or graph writer.
All requests consume the original journals and one case-wide 40-request budget.
"""

import html
import importlib.util
import json
import os
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.grounding.models import digest
from packages.llm_provider.provider import ProviderError

from .campaign import MultimodalCampaign
from .hypotheses import LABELS
from .single_proposition import REGISTRY


class PropositionCampaign(MultimodalCampaign):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.propositions = {label: load_intent(label, REGISTRY)[0] for label, _, _ in LABELS}
        self.active = next(iter(self.propositions))

    def paths(self):
        return [self.phase_path(t) for t in self.targets] + [
            self.output / ("review-" + p) for p in self.propositions
        ]

    def phase_path(self, target):
        return (
            super().phase_path(target)
            if target is not None
            else self.output / ("review-" + self.active)
        )

    def request_key(self, target):
        return (
            super().request_key(target)
            if target is not None
            else "mm-proposition:" + digest((self.campaign_id, self.active))[7:55]
        )

    def register_phase(self, evidence, key):
        if evidence.target is not None:
            return super().register_phase(evidence, key)
        self.base.check()
        task = self.propositions[self.active]
        ctx = self.runs.create(
            case_code=self.base.binding["case_code"],
            key=key,
            task=task,
            policy_ref=dict(
                kind="policy",
                code="POLICY:SINGLE_PROPOSITION",
                version="1.0.0",
                checksum=digest((self.base.ledger, self.config, task)),
            ),
            catalog_ref=dict(
                kind="catalog", code="CATALOG:NO_NATIVE_TOOL", version="1.0.0", checksum=digest([])
            ),
            synthetic=self.synthetic,
        )
        self.base.contexts[ctx.run_id] = ctx
        state, _ = evidence.project(run_id=ctx.run_id, config=self.config)
        return ctx, state

    def save_tree(self, acquired, review):
        from .memory import archive
        from .report import export, restore_export

        self.report_name = self.output.name + "-" + self.active
        result = dict(
            schema_version="multimodal-case-result/1.0.0",
            sample=self.sample,
            dataset=self.base.manifest.dataset,
            case_code=self.base.binding["case_code"],
            task=self.propositions[self.active].model_dump(mode="json"),
            phases=[*acquired, review],
            complete=review["accepted"] or review["evidence_limited"],
            new_requests=self.usage(),
            hidden_labels_used=False,
            clinical_acceptance=False,
        )
        if hasattr(self, "continuation"):
            result["acquisition_continuation"] = self.continuation.report()
        if hasattr(self, "remaining_review"):
            result["remaining_review"] = self.remaining_review.report()
        docs = self.documents(review["run_id"])
        with docs.session():
            existing = docs.read("multimodal-case-result")
            if existing:
                if {k: v for k, v in existing[0].items() if k != "new_requests"} != {
                    k: v for k, v in result.items() if k != "new_requests"
                } or existing[0]["new_requests"] > result["new_requests"]:
                    raise ValueError("proposition_tree_recovery_binding")
                result = existing[0]
            else:
                docs.write("multimodal-case-result", (result,))
        write_once(self.phase_path(None) / "case-result.json", result)
        exported = (
            restore_export(self, result)
            if docs.read("dynamic-case-report")
            else export(self, result)
        )
        memory = archive(self, result)
        return dict(proposition=self.active, run_id=review["run_id"], **exported, memory=memory)

    def execute(self):
        import fcntl

        with (self.output / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.freeze()
            path = self.output / "suite-result.json"
            if path.exists():
                # Completed or stopped suites are never re-dispatched.
                saved = json.loads(path.read_bytes())
                if (
                    digest(saved)
                    != json.loads((self.output / "suite-checksum.json").read_bytes())["checksum"]
                ):
                    raise ValueError("proposition_suite_result_changed")
                return saved
            self.guard()
            spec = importlib.util.spec_from_file_location(
                "private_launcher", ".local/vectorengine/run.py"
            )
            launcher = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(launcher)
            name = self.config.transport.api_key_env
            old = os.environ.get(name)
            acquired, trees, error = [], [], None
            if hasattr(self, "remaining_review"):
                trees.extend(self.remaining_review.trees)
            try:
                os.environ[name] = launcher.load_secret()
                for target in self.targets:
                    phase = self.execute_phase(target)
                    acquired.append(phase)
                    print(
                        json.dumps(
                            dict(
                                stage="shared_acquisition",
                                target=target,
                                accepted=phase["accepted"],
                                error=phase["error"],
                            )
                        ),
                        flush=True,
                    )
                    if (
                        phase["error"]
                        and not phase["accepted"]
                        and not self.local_acquisition_gap(phase)
                    ):
                        error = phase["error"]
                        break
                    if self.acquisition_exhausted():
                        break  # Review confirmed observations; skipped targets remain gaps.
                if error is None:
                    for label in self.propositions:
                        if (
                            hasattr(self, "remaining_review")
                            and label in self.remaining_review.labels
                        ):
                            continue
                        self.active = label
                        review = self.execute_phase()
                        trees.append(self.save_tree(acquired, review))
                        print(
                            json.dumps(
                                dict(
                                    stage="independent_tree",
                                    target=label,
                                    accepted=review["accepted"],
                                    error=review["error"],
                                )
                            ),
                            flush=True,
                        )
                        if review["error"] and not review["accepted"]:
                            error = review["error"]
                            break
            except (ProviderError, ValueError, OSError) as exc:
                # Safe code only; arbitrary exception payloads can contain private data.
                error = exc.code if isinstance(exc, ProviderError) else type(exc).__name__
            finally:
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old
            result = dict(
                schema_version="independent-proposition-campaign/1.0.0",
                case_code=self.base.binding["case_code"],
                trees=trees,
                shared_acquisition=acquired,
                new_requests=self.usage(),
                error=error,
                complete=error is None
                and len(trees) == len(self.propositions)
                and all(t["complete"] for t in trees),
                clinical_diagnosis=False,
                calibrated_probabilities=None,
                class_label=None,
                hidden_labels_used=False,
            )
            if hasattr(self, "continuation"):
                result["acquisition_continuation"] = self.continuation.report()
            if hasattr(self, "remaining_review"):
                result["remaining_review"] = self.remaining_review.report()
            if getattr(self.base, "resilient_vision", False):
                result["acquisition_circuit_open"] = self.acquisition_exhausted()
            write_once(path, result)
            write_once(self.output / "suite-checksum.json", dict(checksum=digest(result)))
            self.comparison(result)
            return result

    def comparison(self, result):
        from .report_storage import save

        root = Path("docs/exports")
        name = "Multimodal_" + self.output.name + "-comparison"
        links = "".join(
            '<li><a href="'
            + html.escape(Path(t["html"]).name, quote=True)
            + '">'
            + html.escape(t["proposition"])
            + "</a></li>"
            for t in result["trees"]
        )
        page = (
            '<!doctype html><meta charset="utf-8"><h1>独立命题树比较</h1>'
            "<p>共用取证，每个命题独立评阅、接纳、评分和报告。"
            "分数不是分类概率；本页不是总根节点。</p><ul>"
            + links
            + "</ul><p>完成："
            + str(result["complete"])
            + "；新请求："
            + str(result["new_requests"])
            + "；停止原因："
            + html.escape(str(result["error"]))
            + "</p>"
        )
        write_once(root / (name + ".json"), result)
        path = root / (name + ".html")
        if path.exists() and path.read_text() != page:
            raise ValueError("proposition_comparison_conflict")
        if not path.exists():
            with path.open("x") as stream:
                stream.write(page)
        if result["trees"]:
            save(
                self.documents(result["trees"][-1]["run_id"]),
                result,
                page,
                key="independent-proposition-comparison",
            )
