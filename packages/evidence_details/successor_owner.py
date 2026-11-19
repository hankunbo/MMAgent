"""Explicit installation of accepted research Material using the existing document port.

The mandatory dependency verifier is installed by the original source/input owner;
checksums and historical receipts alone never grant current permission. No new run,
SQL table, medical rule or interpretation is introduced.
"""

from packages.grounding.models import digest, validate
from packages.llm_provider.models import Material

from .acceptance import ResearchAcceptance
from .acceptance_models import DomainState


class ResearchSuccessorOwner:
    def __init__(
        self,
        *,
        initial,
        documents,
        verify_dependencies,
        read_image,
        config,
        role,
        live_permission=None,
        source_quote_policy="research-quote-policy/literal/1.0.0",
    ):
        from packages.knowledge_claims.research_quote import check_quote_policy

        self.source_quote_policy = check_quote_policy(source_quote_policy)
        if role not in {"source", "vision", "review"}:
            raise ValueError("domain_role_required")
        self.initial = validate(DomainState, initial)
        if self.initial.config_checksum != digest(config):
            raise ValueError("domain_config_binding")
        self.documents, self.verify_dependencies, self.read_image = (
            documents,
            verify_dependencies,
            read_image,
        )
        self.config, self.role = config, role
        self.live_permission = live_permission

    def _configured(self, state):
        prompt = next(r.prompt_checksum for r in self.config.roles if r.role == self.role)
        return validate(DomainState, state.model_copy(update={"prompt_checksum": prompt}))

    def _entries(self):
        for n in range(256):
            entry = self.documents.read("research-owner-install:" + str(n))
            if entry is None:
                return
            yield n, entry[0]
        raise ValueError("research_owner_install_limit")

    def current(self):
        state = self.initial
        for _, entry in self._entries():
            if entry["initial_checksum"] != digest(self.initial):
                raise ValueError("research_owner_origin_changed")
            state = validate(DomainState, entry["successor"])
        state = self._configured(state)
        self.documents.validate_scope(state.material.scope)
        self.verify_dependencies(state)
        if self.config.mode == "live":
            from .live_permission import require_live_permission

            require_live_permission(self.live_permission, state, role=self.role)
        return state

    def read(self, scope):
        material = self.current().material
        if (scope.case_code, scope.run_id, scope.input_checksum, scope.visible_checksum) != (
            material.scope.case_code,
            material.scope.run_id,
            material.scope.input_checksum,
            material.scope.visible_checksum,
        ):
            raise ValueError("research_owner_scope")
        return validate(Material, material)

    def image(self, binding):
        state = self.current()
        if binding not in [o.image for o in state.material.observations]:
            raise ValueError("research_image_scope")
        data = self.read_image(binding.reference)
        from packages.llm_provider.images import validate_image

        validate_image(binding, data)
        return data

    def verify_image(self, binding):
        self.image(binding)

    def review_template(self, scope):
        from .review_tasks import project_review_tasks

        state = self.current()
        if scope != state.material.scope:
            raise ValueError("research_task_scope")
        return project_review_tasks(state)

    def install(self, event, *, parent, successor):
        parent, successor = validate(DomainState, parent), validate(DomainState, successor)
        with self.documents.session():
            entries = list(self._entries())
            for _, entry in entries:
                if entry["event"] == event.model_dump(mode="json"):
                    if entry["successor"] != successor.model_dump(mode="json"):
                        raise ValueError("research_install_conflict")
                    if self.current() != self._configured(successor):
                        raise ValueError("historical_install_not_current")
                    return
            if self.current() != parent:
                raise ValueError("research_install_parent_changed")
        # successor_material owns saved-object/graph validation; avoid nested Runtime locks.
        material = ResearchAcceptance(
            owner=self,
            documents=self.documents,
            requests=self.live_permission.budget if self.live_permission else None,
            live_permission=self.live_permission,
        ).successor_material(key=event.code + ":domain")
        expected = validate(DomainState, parent.model_copy(update={"material": material}))
        if expected != successor:
            raise ValueError("research_install_successor_binding")
        self.verify_dependencies(successor)
        with self.documents.session():
            if self.current() != parent or len(list(self._entries())) != len(entries):
                raise ValueError("research_install_race")
            if self.live_permission is not None:
                self.live_permission.budget.check_control()
            self.documents.write(
                "research-owner-install:" + str(len(entries)),
                (
                    dict(
                        schema_version="research-owner-install/1.0.0",
                        initial_checksum=digest(self.initial),
                        event=event.model_dump(mode="json"),
                        parent_checksum=digest(parent),
                        successor=successor.model_dump(mode="json"),
                    ),
                ),
            )
