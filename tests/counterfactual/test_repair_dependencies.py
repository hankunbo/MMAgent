"""Current dependency boundaries over real FactReader/ResearchPathReader acceptance."""

import pytest

from packages.counterfactual.models import Candidate
from packages.counterfactual.review import review
from packages.counterfactual.service import CounterfactualSimulation, render
from packages.evidence_details.current_rejection import CurrentResearchRejected
from packages.grounding.models import digest
from packages.llm_provider.transport import Reply
from tests.counterfactual.support import candidate, fixture, protection, run
from tests.counterfactual.test_review import reply
from tests.counterfactual.test_types import other_run, replacing


def restart(x):
    x.service = CounterfactualSimulation(
        reader=x.reader,
        documents=x.f.documents,
        policy=x.policy,
        alternative_readers=x.service.alternative_readers,
    )


@pytest.mark.parametrize("boundary", ["first_read", "result", "validation", "lost_result"])
def test_alternative_invalidation_during_compute_and_commit(boundary, monkeypatch):
    x, y = replacing("replace_tool")
    c = candidate("replace_tool", alternative="alt")
    x.service.declare((c,))
    before = protection(x)
    if boundary == "first_read":

        def read_then_withdraw():
            fact = y.reader()
            y.f.reader.withdrawn = True
            return fact

        x.service.alternative_readers["alt"] = read_then_withdraw
    else:
        write = x.f.documents.write

        def changed(key, values):
            write(key, values)
            trigger = "validation" if boundary == "validation" else "result"
            if key == "simulation:" + trigger + ":one":
                y.f.reader.withdrawn = True
                if boundary == "lost_result":
                    raise OSError("confirmation_lost")

        monkeypatch.setattr(x.f.documents, "write", changed)
    if boundary == "lost_result":
        with pytest.raises(OSError, match="confirmation_lost"):
            x.service.run(c)
        restart(x)
    result = x.service.run(c)
    assert result["status"] == "stopped"
    assert x.service.read(candidate_id=c.id, historical=True) == result["report"]
    assert bool(result["report"]) == (boundary != "first_read")
    assert protection(x) == before


@pytest.mark.parametrize(
    "boundary", ["before_claim", "before_request", "retry", "return", "saved", "saved_read"]
)
def test_expert_every_reservation_retry_and_return_revalidates(boundary, monkeypatch):
    x, y = replacing("replace_tool")
    c = candidate("replace_tool", alternative="alt")
    original = run(x, c)["report"]

    def withdraw():
        y.f.reader.withdrawn = True

    if boundary == "before_claim":
        old_read = x.service._read

        def read(key):
            value = old_read(key)
            if key == "review:one":
                withdraw()
            return value

        monkeypatch.setattr(x.service, "_read", read)
    elif boundary == "before_request":
        claim = x.service._claim

        def changed(*args):
            value = claim(*args)
            withdraw()
            return value

        monkeypatch.setattr(x.service, "_claim", changed)
    elif boundary in {"retry", "return"}:

        def response(body):
            withdraw()
            return Reply(error="timeout") if boundary == "retry" else reply()(body)

        x.f.provider.transport.replies.append(response)
    elif boundary == "saved":
        write = x.f.documents.write

        def changed(key, values):
            write(key, values)
            if key == "simulation:review:one":
                withdraw()

        monkeypatch.setattr(x.f.documents, "write", changed)
        x.f.provider.transport.replies.append(reply())
    else:
        x.f.provider.transport.replies.append(reply())
        assert review(x.service, c, ports=x.f.ports)["status"] == "reviewed_candidate"
        old_read = x.service._read
        seen = []

        def read(key):
            value = old_read(key)
            if key == "review:one":
                seen.append(key)
                if len(seen) == 2:  # after run's final current check
                    withdraw()
            return value

        monkeypatch.setattr(x.service, "_read", read)
    result = review(x.service, c, ports=x.f.ports)
    assert result["status"] in {"safe_stop", "stopped"}
    assert x.f.budget.used == (1 if boundary in {"before_claim", "before_request"} else 2)
    assert x.service.read(candidate_id=c.id, historical=True) == original
    assert review(x.service, c, ports=x.f.ports)["status"] == "stopped"
    assert x.f.budget.used <= 2


@pytest.mark.parametrize(
    "failure", ["missing", "digest", "owner_version", "reset", "technical", "value_error"]
)
def test_precise_dependency_errors_and_historical_bytes(failure, monkeypatch):
    x, y = replacing("replace_tool")
    c = candidate("replace_tool", alternative="alt")
    original = run(x, c)["report"]
    if failure == "missing":
        x.service.alternative_readers.clear()
    elif failure == "digest":
        # Another fully accepted chain, not a forged Fact or validity flag.
        z = fixture(directions=(-0.7,), mutate=other_run())
        x.service.alternative_readers["alt"] = z.reader
    elif failure == "owner_version":
        state = y.f.reader.state
        y.f.reader.state = state.model_copy(
            update=dict(
                resources=(
                    state.resources[0].model_copy(
                        update=dict(checksum=digest("new resource content"))
                    ),
                )
            )
        )
    else:
        error = (
            CurrentResearchRejected("graph_reset")
            if failure == "reset"
            else OSError("database_connection_unknown")
            if failure == "technical"
            else ValueError("object_decode_unknown")
        )

        def broken():
            raise error

        monkeypatch.setattr(y.f.reader, "current", broken)
    if failure in {"technical", "value_error"}:
        with pytest.raises((OSError, ValueError), match="unknown"):
            x.service.run(c)
        with pytest.raises((OSError, ValueError), match="unknown"):
            review(x.service, c, ports=x.f.ports)
    else:
        assert x.service.run(c)["status"] == "stopped"
        with pytest.raises(ValueError):
            x.service.read(candidate_id=c.id)
    assert x.service.read(candidate_id=c.id, historical=True) == original
    assert x.f.budget.used == 1


def test_missing_reader_is_not_missing_observation():
    x, _ = replacing("replace_tool")
    x.service.alternative_readers.clear()
    result = run(x, candidate("replace_tool", alternative="alt"))
    assert result["reason"] == "simulation_alternative_reader_missing"
    assert result["report"] is None


@pytest.mark.parametrize("withdraw", ["a", "b"])
def test_all_consumed_dependencies_but_not_unrelated_candidates(withdraw):
    y = fixture(statuses=("obtained",) * 2, mutate=other_run())
    z = fixture(statuses=("obtained",) * 2, directions=(-0.7, 0.3), mutate=other_run())
    x = fixture(
        statuses=("obtained",) * 2,
        allow=dict(select_check=("item:0", "item:1"), remove_evidence=("item:0",)),
        policy_updates=dict(alternatives={"a": y.fact.checksum, "b": z.fact.checksum}),
    )
    x.service.alternative_readers = {"a": y.reader, "b": z.reader}
    c = Candidate(
        id="one",
        interventions=(
            candidate("select_check", ("item:0",), alternative="a").interventions[0],
            candidate("select_check", ("item:1",), alternative="b").interventions[0],
        ),
    )
    independent = candidate(id="independent")
    x.service.declare((c, independent))
    r = x.service.run(c)["report"]
    assert set(r["alternatives"]) == {"a", "b"}
    for name, dependency in r["alternatives"].items():
        assert dependency["after"]["checksum"] == x.policy.alternatives[name]
        assert dependency["after"]["input_checksum"] == x.fact.binding.input_checksum
        assert dependency["after"]["owner"]["reader"] == "ResearchPathReader"
    (y if withdraw == "a" else z).f.reader.withdrawn = True
    assert x.service.run(c)["status"] == "stopped"
    assert x.service.run(independent)["status"] == "candidate"
    assert x.service.read(candidate_id="independent")
    assert x.f.budget.used == 1


def test_old_report_policy_and_search_are_historical_only():
    x = fixture()
    c = candidate()
    r = run(x, c)["report"]
    old = dict(r, schema_version="counterfactual-report/1.0.0")
    old["checksum"] = digest({k: v for k, v in old.items() if k != "checksum"})
    # Synthetic old serialization; actual frozen reports are never rewritten.
    x.f.documents.saved["simulation:result:one"] = (old, render(old))
    assert x.service.read(candidate_id="one", historical=True) == old
    with pytest.raises(ValueError, match="legacy_semantics"):
        x.service.read(candidate_id="one")
    with pytest.raises(ValueError, match="legacy_semantics"):
        x.service.run(c)
    policy = dict(x.f.documents.read("simulation:policy")[0])
    policy.pop("semantics")
    x.f.documents.saved["simulation:policy"] = (policy,)
    x.f.documents.saved["simulation:search"] = ({"status": "candidate", "old": True},)
    before = digest(x.f.documents.saved)
    with pytest.raises(ValueError, match="legacy_semantics"):
        x.service.search()
    assert digest(x.f.documents.saved) == before


@pytest.mark.parametrize("boundary", ["current_read", "expert_return"])
def test_late_dependency_technical_errors_are_not_source_withdrawal(boundary, monkeypatch):
    x, y = replacing("replace_tool")
    c = candidate("replace_tool", alternative="alt")
    original = run(x, c)["report"]

    def broken():
        raise OSError("alternative_object_connection_unknown")

    if boundary == "current_read":
        read = x.service._read

        def fail_after_load(key):
            result = read(key)
            if key == "result:one":
                monkeypatch.setattr(y.f.reader, "current", broken)
            return result

        monkeypatch.setattr(x.service, "_read", fail_after_load)
        with pytest.raises(OSError, match="connection_unknown"):
            x.service.read(candidate_id="one")
        assert x.f.budget.used == 1
    else:

        def hook():
            monkeypatch.setattr(y.f.reader, "current", broken)

        x.f.provider.transport.replies.append(reply(hook=hook))
        with pytest.raises(OSError, match="connection_unknown"):
            review(x.service, c, ports=x.f.ports)
        assert x.f.budget.used == 2
        assert x.f.documents.read("simulation:review-claim:one")
        assert x.f.documents.read("simulation:review:one") is None
    assert x.service.read(candidate_id="one", historical=True) == original
