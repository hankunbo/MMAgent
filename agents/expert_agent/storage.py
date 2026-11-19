"""Small immutable records over the existing RuntimeDocuments transaction port."""

from packages.grounding.models import digest


class ReviewStorage:
    def __init__(self, documents):
        self.documents = documents

    def read(self, key):
        with self.documents.session():
            return self.documents.read("review:" + key)

    def save(self, key, value):
        try:
            with self.documents.session():
                self.documents.write("review:" + key, (value,))
        except Exception:
            self.documents.failure(key, digest(value), "object_or_transaction_failure")
            raise

    def claim(self, key, value):
        # Claim before a call; an unconfirmed call is never repeated on recovery.
        with self.documents.session():
            old = self.documents.read("review:" + key)
            if old is not None:
                if old[0] != value:
                    rejection = dict(
                        schema_version="research-review-rejection/1.0.0",
                        reason="idempotency_conflict",
                        key_checksum=digest(key),
                        original_checksum=digest(old[0]),
                        supplied_checksum=digest(value),
                    )
                    self.documents.write("review:conflict:" + digest(rejection)[7:], (rejection,))
                    raise ValueError("review_key_conflict")
                return False
            self.documents.write("review:" + key, (value,))
            return True
