"""B03 separate request type. The frozen v1 context and 05/06 schemas stay unchanged."""

from packages.knowledge_review.research import KnowledgeUseRequest

from .models import RetrievalContext


class ResearchRetrievalContext(RetrievalContext):
    # Explicit independently versioned use envelope. No default or implicit authorization.
    use: KnowledgeUseRequest


def parse_context(value):
    data = dict(vars(value))
    model = ResearchRetrievalContext if "use" in data else RetrievalContext
    return model.model_validate(data, strict=True)
