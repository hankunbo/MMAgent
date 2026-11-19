"""05C seeded small graphs with separate mappings for intermediate products."""

import random

from packages.grounding import CatalogSnapshot, Mapping, digest, validate
from packages.grounding.locks import GroundingLock
from packages.grounding.ports import KnowledgeProjection
from packages.grounding.synthetic import SyntheticCatalogReader, SyntheticKnowledgeReader, ref
from tests.grounding.query_support import metadata, paths, ready, request_for, updated
from tests.grounding.test_models import reseal


def small_graph(seed):
    rng = random.Random(seed)
    _, base, policy, catalog, knowledge, _ = ready()
    depth = rng.randint(2, 4)
    widths = [rng.randint(2, 3) for _ in range(depth)]
    products = [base.accepted_artifacts[0]] + [
        validate(
            type(base.accepted_artifacts[0]),
            {
                **base.accepted_artifacts[0].model_dump(mode="json"),
                "artifact_ref": ref("artifact", f"synthetic:Product{seed}_{level}"),
            },
        )
        for level in range(1, depth)
    ]
    image = base.candidates[0].inputs[2].constraint
    cup = base.candidates[0].inputs[1].constraint
    tools, groups = [], []
    for level, width in enumerate(widths):
        candidates = []
        for branch in range(width):
            tool_ref = ref("tool", f"synthetic_chain_{seed}_{level}_{branch}")
            implementation = ref("implementation", f"synthetic_impl_{seed}_{level}_{branch}")
            constraints = [products[level + 1] if level + 1 < depth else image, cup]
            inputs = [
                {
                    **base.candidates[0].inputs[n].model_dump(mode="json"),
                    "port": dict(tool_ref=tool_ref, direction="input", port_name=f"input_{n}"),
                    "constraint": constraint,
                }
                for n, constraint in enumerate(constraints)
            ]
            output_port = dict(tool_ref=tool_ref, direction="output", port_name="product")
            tool = {
                **catalog.tools[0].model_dump(mode="json"),
                "tool_ref": tool_ref,
                "implementation_ref": implementation,
                "inputs": inputs,
                "outputs": [products[level]],
                "output_ports": [output_port],
                "workflows": [],
            }
            tools.append(tool)
            candidates.append(
                {
                    **base.candidates[0].model_dump(mode="json"),
                    "code": f"CAPABILITY:SYNTHETIC:L{level}B{branch}",
                    "tool_ref": tool_ref,
                    "implementation_ref": implementation,
                    "inputs": inputs,
                    "output": products[level],
                    "output_port": output_port,
                    "workflow_ref": None,
                }
            )
        groups.append(candidates)
    catalog = reseal(
        CatalogSnapshot,
        {
            **catalog.model_dump(mode="json"),
            "artifacts": [
                *(c.artifact_ref for c in products),
                image.artifact_ref,
                cup.artifact_ref,
            ],
            "tools": tools,
            "manifest_checksum": digest({"seed": seed, "synthetic_graph": tools}),
        },
    )
    mappings, readers = [], []
    for level, candidates in enumerate(groups):
        requirement = (
            base.requirement_ref
            if level == 0
            else ref("evidence_requirement", f"SOURCE:SYNTHETIC:CHAIN{seed}_{level}")
        )
        mapping = reseal(
            Mapping,
            {
                **base.model_dump(mode="json"),
                "code": f"BINDING:OID_CHAIN{seed:04d}_{level}",
                "catalog_ref": catalog.reference(),
                "requirement_ref": requirement,
                "accepted_artifacts": [products[level]],
                "candidates": candidates,
            },
        )
        mappings.append(mapping)
        readers.append(
            SyntheticKnowledgeReader(
                validate(
                    KnowledgeProjection,
                    {
                        **knowledge.projection.model_dump(mode="json"),
                        "requirement_ref": requirement,
                        "artifact_type_code": products[level].artifact_ref.code,
                    },
                )
            )
        )
    request = request_for(mappings[0], policy)
    lock = reseal(
        GroundingLock,
        {
            **request.lock.model_dump(mode="json"),
            "mapping_refs": [m.reference() for m in mappings],
            "knowledge_refs": [
                *request.lock.knowledge_refs,
                *(m.requirement_ref for m in mappings[1:]),
            ],
        },
    )
    request = updated(
        request,
        lock=lock,
        artifacts=[metadata(request, image, "CHAINIMAGE"), metadata(request, cup, "CHAINCUP")],
        environment=[
            dict(
                implementation_ref=t.implementation_ref,
                availability="available",
                preconditions="satisfied",
                source_ref=ref("source_version", "SOURCE:SYNTHETIC:ENV"),
            )
            for t in catalog.tools
        ],
    )
    reader = SyntheticCatalogReader([catalog])

    def views():
        return tuple(paths(m, policy, k, reader)[0] for m, k in zip(mappings, readers, strict=True))

    return request, views, reader, widths
