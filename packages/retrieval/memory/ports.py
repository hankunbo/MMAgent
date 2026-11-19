"""Installed query owner; optional 06 readmission never launches a tool."""

from packages.memory_review.models import Query
from packages.memory_review.task_models import qualification


class BoundQueryReader:
    def __init__(self, *, read_query, read_rules, tool_requests, admission):
        self.read_query, self.read_rules = read_query, read_rules
        self.tool_requests, self.admission = tool_requests, admission

    def current(self, run_id):
        return qualification(Query, self.read_query(run_id))

    def check_versions(self, query):
        if self.read_rules(query.run_id) != query.rule_versions:
            raise ValueError("memory_rule_version_changed")
        requests = self.tool_requests(query.run_id)
        versions = {
            r.candidate.implementation_ref.code: r.candidate.implementation_ref.version
            for r in requests
        }
        if versions != query.tool_versions:
            raise ValueError("memory_tool_version_changed")
        for request in requests:
            if request.lock.run_id != query.run_id or request.scope.case_code != query.case_code:
                raise ValueError("memory_tool_scope")
            result = self.admission.admit_research(request)
            if not result.retained:
                raise ValueError("memory_tool_not_currently_admitted")
