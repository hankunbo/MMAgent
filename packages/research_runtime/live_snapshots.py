"""Original live saved-fact snapshot adapter; no cross-call authority cache."""

from packages.planner_control.models import Outcome, Reference


class LiveSnapshots:
    def __init__(self, owner):
        self.owner = owner

    def current_with_domain(self, binding):
        state = self.owner.current()
        self._check_binding(binding, state.material)
        return state.material.scope.graph, state

    def current(self, binding):
        return self.current_with_domain(binding)[0]

    @staticmethod
    def _check_binding(binding, material):
        if (binding.run_id, binding.input_checksum, binding.visible_checksum) != (
            material.scope.run_id,
            material.scope.input_checksum,
            material.scope.visible_checksum,
        ):
            raise ValueError("live_snapshot_binding")

    def step(self, binding, action):
        graph, state = self.current_with_domain(binding)
        material = state.material
        if action == "load":
            refs = (
                Reference(
                    role="visible",
                    code=binding.case_code,
                    checksum=binding.visible_checksum,
                    version="1.0.0",
                ),
            )
        elif action == "retrieve":
            refs = tuple(s.reference for s in material.sources)
        elif action == "instantiate":
            refs = (material.policy_ref,)
        else:
            refs = (
                Reference(
                    role="graph", code=graph.code, checksum=graph.checksum, version=graph.version
                ),
            )
        return Outcome(reason="reuse_saved_observations_no_new_tool_execution", refs=refs)
