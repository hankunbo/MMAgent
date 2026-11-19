"""Explicit read dispatch: no inferred upgrades or rehashing of legacy documents."""

from packages.grounding.models import validate

from . import models as v10
from . import models_v11 as v11


def parse(kind, value):
    data = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    version = data.get("schema_version")
    for module in (v10, v11):
        model = getattr(module, kind)
        if version == model.model_fields["schema_version"].default:
            return validate(model, value)
    raise ValueError("discovery_unsupported_" + kind + "_version")


def contract_status(record):
    if record is None:
        return "no_record"
    plan = parse("DiscoveryPlan", record["plan"])
    return (
        "current_input_contract"
        if isinstance(plan, v11.DiscoveryPlan)
        else "legacy_missing_input_contract"
    )
