"""Current project withdrawals, separate from immutable historical catalog facts.

User decision 2026-09-15: remove MedAgent-Pro optic-cup implementations from
new use while SAMed is evaluated independently. No automatic replacement.
"""

RETIREMENT_VERSION = "project-tool-withdrawal/1.0.0"
RETIRED_IMPLEMENTATIONS = frozenset(
    {
        ("medical_sam_optic_cup_segmenter", "1.0.0", "local_medagent_pro"),
        ("medical_sam_optic_cup_segmenter", "1.0.1", "local_medagent_pro_complete_v2"),
    }
)


def retired(tool_code: str, version: str, implementation_key: str) -> bool:
    return (tool_code, version, implementation_key) in RETIRED_IMPLEMENTATIONS
