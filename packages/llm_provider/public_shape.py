"""45 synthetic review tasks with real task projection and admission contracts."""

from packages.evidence_details.acceptance_models import DomainState, TaskPolicyV2
from packages.grounding.models import digest, validate
from packages.llm_provider.live_probes import PublicProbeMaterials
from packages.llm_provider.models import Material


class PublicReviewMaterials(PublicProbeMaterials):
    def __init__(self, run_id, config, *, task_count=45):
        if type(task_count) is not int or not 1 <= task_count <= 45:
            raise ValueError("public_task_count")
        super().__init__(run_id, config)
        raw = self.state.model_dump(mode="json")
        template = raw["policy"]["items"][0]
        observation = raw["material"]["observations"][0]
        items, observations = [], []
        import copy

        for index in range(task_count):
            item, obs = copy.deepcopy(template), copy.deepcopy(observation)
            item["id"] = f"probe:task{index:02d}"
            obs["reference"]["code"] = f"OBS:PUBLIC_{index:02d}"
            obs["status"] = ("obtained", "unknown", "missing", "ungradable", "obtained")[index % 5]
            obs["value"] = "2 + 2 = 4" if obs["status"] == "obtained" else None
            obs["reference"]["checksum"] = digest(
                [obs["reference"]["code"], obs["status"], obs["value"]]
            )
            item["observation_code"] = obs["reference"]["code"]
            definition = item["definition"]
            if index % 5 == 4:
                definition["interpretation"] = None
                definition["interpretation_status"] = "missing"
            definition["reference"]["code"] = f"POLICY:PUBLIC_DEF_{index:02d}"
            definition["reference"]["checksum"] = digest(
                {k: v for k, v in definition.items() if k != "reference"}
            )
            items.append(item)
            observations.append(obs)
        raw["policy"]["items"] = items
        raw["policy"]["groups"][0]["slots"] = [[i["id"]] for i in items]
        policy = validate(TaskPolicyV2, raw["policy"])
        material = raw["material"]
        material["observations"] = observations
        material["scope"]["policy_checksum"] = digest(policy)
        material["policy_ref"]["checksum"] = digest(policy)
        material["scope"].pop("material_checksum")
        material["scope"]["material_checksum"] = digest(material)
        raw.update(material=validate(Material, material), policy=policy)
        self.state = validate(DomainState, raw)

    def current(self):
        return validate(DomainState, self.state)
