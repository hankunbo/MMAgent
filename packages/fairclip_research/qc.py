"""Exact JSON, input, model and class mapping QC within the original 06 port."""

from typing import Literal

from packages.contracts import QCStatus
from packages.grounding.models import Ref, Term, Value, digest
from packages.object_store import checksum
from packages.tool_execution.models import ExecutionError
from packages.tool_execution.qc import QCObservation, TechnicalQCRegistry

from .models import read_output


class FairCLIPQCPolicy(Value):
    implementation: Literal["fairclip_json/1.0.0"] = "fairclip_json/1.0.0"
    artifact_ref: Ref
    artifact_type_code: str
    unit: Literal["logit"] = "logit"
    unit_term: Term
    role: Literal["model_prediction"] = "model_prediction"


class FairCLIPQCRegistry(TechnicalQCRegistry):
    def evaluate(self, output, rule, data, inputs):
        policy = self.policy_for(output)
        if not isinstance(policy, FairCLIPQCPolicy):
            return super().evaluate(output, rule, data, inputs)
        if (
            len(data) > 65536
            or len(inputs) != 1
            or "application/json" not in rule.media_types
            or rule.dimensions_min != (2,)
            or rule.dimensions_max != (2,)
        ):
            raise ExecutionError("fairclip_output_shape")
        value = read_output(data)
        if value.input_checksum != inputs[0].checksum:
            raise ExecutionError("fairclip_input_drift")
        return QCObservation(
            status=QCStatus.PASS,
            dimensions=(2,),
            media_type="application/json",
            policy_checksum=digest(policy),
            content_checksum=checksum(data),
            implementation=policy.implementation,
        )
