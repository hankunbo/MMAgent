"""Original definition receipt controls which old judgment is superseded."""

from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.generic_definition_projection import project_interpretations
from packages.evaluation_preparation.generic_definition_selection import (
    DefinitionRevisionSelection,
    RetainedDefinitionOriginReader,
)
from packages.evidence_details.accepted_aggregation import AcceptedOriginReader
from tests.evaluation_preparation.test_generic_definition_projection import fixture


def setup():
    f, _, seed, _ = fixture()
    current, manifest = project_interpretations(
        seed, f.domain, ["new-definition"], selected_only=False
    )
    old = seed.model_copy(
        update={
            "material": seed.material.model_copy(
                update={
                    "scope": seed.material.scope.model_copy(update={"run_id": f.current["run_id"]})
                }
            )
        }
    )
    candidate = SimpleNamespace(
        items=tuple(SimpleNamespace(assessment_id=i.id) for i in old.policy.items)
    )
    owner = DefinitionRevisionSelection(
        owner=f.domain, keys=["new-definition"], seed_reader=lambda _: seed
    )
    return f, old, current, candidate, owner, manifest


def test_only_owner_revised_definition_is_removed_not_a_second_vote():
    f, old, current, candidate, owner, manifest = setup()
    retained = owner.retained_ids(old, current, candidate)
    assert retained == {i.assessment_id for i in candidate.items} - {f.target}
    assert len(retained) == len(candidate.items) - 1
    assert manifest["added_requirements"] == []
    assert old.policy.items[0].definition != current.policy.items[0].definition
    reader = RetainedDefinitionOriginReader(
        definition_selection=owner,
        documents=None,
        requests=None,
        state_reader=None,
        acceptance_key="old",
    )
    reader.check_compatible_ids(old, current, candidate, retained)
    with pytest.raises(ValueError, match="unexpected_changed_item"):
        reader.check_compatible_ids(old, current, candidate, retained - {next(iter(retained))})
    with pytest.raises(ValueError, match="unexpected_changed_item"):
        reader.check_compatible_ids(old, current, candidate, retained | {f.target})
    ordinary = AcceptedOriginReader(
        documents=None, requests=None, state_reader=None, acceptance_key="old"
    )
    with pytest.raises(ValueError, match="changed_item"):
        ordinary.check_compatible_ids(old, current, candidate, retained)


@pytest.mark.parametrize(
    "fault", ["origin", "definition", "observation", "current", "receipt", "missing_item"]
)
def test_changed_binding_cannot_silently_discard_old_judgment(fault):
    f, old, current, candidate, owner, _ = setup()
    if fault == "origin":
        old = old.model_copy(
            update={
                "material": old.material.model_copy(
                    update={"scope": old.material.scope.model_copy(update={"run_id": "RUN:OTHER"})}
                )
            }
        )
    elif fault == "definition":
        old = old.model_copy(
            update={
                "policy": old.policy.model_copy(
                    update={"requirements": current.policy.requirements}
                )
            }
        )
    elif fault == "observation":
        old = old.model_copy(
            update={
                "material": old.material.model_copy(
                    update={
                        "observations": tuple(
                            o.model_copy(update={"value": "changed"})
                            for o in old.material.observations
                        )
                    }
                )
            }
        )
    elif fault == "current":
        current = current.model_copy(update={"laterality": "changed"})
    elif fault == "receipt":
        owner.keys = ("unknown",)
    else:
        candidate.items = tuple(i for i in candidate.items if i.assessment_id != f.target)
    with pytest.raises(ValueError):
        owner.retained_ids(old, current, candidate)
