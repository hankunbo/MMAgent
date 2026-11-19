"""03 integration for conservative mapped input initialization; no clinical fixtures."""

from dataclasses import replace

import pytest

from packages.contracts import DataAvailability, NodeType
from packages.deterministic_pipeline.demands import mapped_node_resolution
from packages.graph_engine import Applicability, NodeResolution, instantiate_template
from tests.graph.test_graph_engine_03a import _case, _context, _template


def test_real_mapping_does_not_make_unknown_evidence_derivable():
    template = _template()
    evidence = tuple(n.node_code for n in template.nodes if n.node_type is NodeType.EVIDENCE)
    result = instantiate_template(
        template,
        _case(),
        replace(
            _context(template),
            resolutions=tuple(mapped_node_resolution(n, synthetic=False) for n in evidence),
        ),
    )
    traces = [t for t in result.trace.node_results if t.node_code in evidence]
    assert all(t.applicability is Applicability.UNKNOWN for t in traces)
    assert all(t.availability is DataAvailability.UNAVAILABLE for t in traces)
    assert all(not t.evidence_binding_codes for t in traces)
    assert all(
        n.status.value == "PENDING" for n in result.instance.nodes if n.node_code in evidence
    )


def test_synthetic_mapping_keeps_original_graph_bytes():
    template = _template()
    evidence = tuple(n.node_code for n in template.nodes if n.node_type is NodeType.EVIDENCE)
    old = replace(
        _context(template),
        resolutions=tuple(
            NodeResolution(n, Applicability.APPLICABLE, DataAvailability.DERIVABLE)
            for n in evidence
        ),
    )
    current = replace(
        old, resolutions=tuple(mapped_node_resolution(n, synthetic=True) for n in evidence)
    )
    assert instantiate_template(template, _case(), old) == instantiate_template(
        template, _case(), current
    )


def test_missing_scope_is_not_a_synthetic_or_clinical_permission():
    with pytest.raises(ValueError, match="explicit_input_scope_required"):
        mapped_node_resolution("EVIDENCE:UNKNOWN", synthetic=None)
