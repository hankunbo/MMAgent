"""Local report recovery creates its directory without requesting any inference."""

from packages.evaluation_preparation.comprehensive_review import ComprehensiveReview
from packages.evaluation_preparation.confirmed_review import ConfirmedCaseReview


def test_recovery_initializes_report_directory_and_preserves_existing_proof(tmp_path, monkeypatch):
    previous = tmp_path / "13-full-review-r1"
    previous.mkdir()

    def prepared(self, sample):
        self.output = previous

    monkeypatch.setattr(ComprehensiveReview, "__init__", prepared)
    recovered = ConfirmedCaseReview("synthetic")
    assert recovered.previous_output == previous
    assert (recovered.output / "review").is_dir()
    # Exact local recovery does not recreate or overwrite already written evidence.
    proof = recovered.output / "recovery.json"
    proof.write_text('{"new_requests": 0}')
    again = ConfirmedCaseReview("synthetic")
    assert again.recovery == {"new_requests": 0}
    assert proof.read_text() == '{"new_requests": 0}'
