"""Installed visibility/source owner bridge; no caller supplied release flag grants access."""

from packages.grounding.models import validate

from .models import Material


class BoundMaterials:
    """Pin an approved projection and re-read EVERY owner dependency before each use.

    read_source must return a fresh approved Source through 04: formal installations
    use original Published/release/applicability reads; research uses source/use locks.
    There are no permissive defaults for any reader. Production assembly supplies
    owners, never text returned by a model. Images are fetched by exact approved ref.
    """

    def __init__(
        self,
        *,
        material,
        read_graph,
        read_visible,
        read_source,
        read_observation,
        read_image,
        review_template=None,
    ):
        self.material = validate(Material, material)
        self.read_graph, self.read_visible = read_graph, read_visible
        self.read_source, self.read_observation = read_source, read_observation
        self.read_image, self.template = read_image, review_template

    def read(self, scope):
        m = self.material
        if (scope.case_code, scope.run_id, scope.input_checksum, scope.visible_checksum) != (
            m.scope.case_code,
            m.scope.run_id,
            m.scope.input_checksum,
            m.scope.visible_checksum,
        ):
            raise ValueError("material_scope")
        if (
            self.read_graph(scope) != m.scope.graph
            or self.read_visible(scope) != m.scope.visible_checksum
        ):
            raise ValueError("material_graph_or_visibility_changed")
        for source in m.sources:
            if self.read_source(source.reference, source.purpose, source.release) != source:
                raise ValueError("material_source_changed")
        for observation in m.observations:
            if self.read_observation(scope, observation.reference) != observation:
                raise ValueError("material_observation_changed")
        return validate(Material, m)

    def image(self, binding):
        if binding not in [o.image for o in self.material.observations]:
            raise ValueError("material_image_not_visible")
        raw = self.read_image(binding.reference)
        from .images import validate_image

        validate_image(binding, raw)
        return raw

    def review_template(self, scope):
        if self.template is None:
            raise ValueError("review_template_not_installed")
        self.read(scope)
        from packages.evidence_details.review_tasks import project_review_tasks

        # Installed owner callable, not a caller-approved template snapshot.
        state = self.template()
        if state.material != self.material:
            raise ValueError("review_owner_material_changed")
        return project_review_tasks(state)
