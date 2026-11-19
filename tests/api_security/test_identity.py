import pytest

from apps.api.auth import LocalIdentities, Principal, require_role
from apps.api.errors import Fault


@pytest.mark.parametrize(
    "updates",
    [
        {"role": "admin"},
        {"synthetic": "false"},
        {"actor_code": "raw patient name"},
        {"case_codes": {"CASE:ONE"}},
        {"case_codes": frozenset({"/tmp/private"})},
    ],
)
def test_invalid_identity_rejected(updates):
    values = dict(actor_code="SOURCE:RESEARCHER", role="researcher")
    values.update(updates)
    with pytest.raises(ValueError, match="local_identity"):
        LocalIdentities({"synthetic-token": Principal(**values)})


def test_clinical_alias_has_only_existing_review_permission():
    actor = Principal("SOURCE:REVIEWER", "clinical_reviewer")
    require_role(actor, "offline_reviewer")
    with pytest.raises(Fault, match="SCOPE_FORBIDDEN"):
        require_role(actor, "researcher")
