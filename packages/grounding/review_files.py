"""Pure reviewed-file value shared by admission and execution without owner imports."""

from pydantic import Field

from .models import Checksum, Value


class ReviewedFile(Value):
    path: str = Field(min_length=1, max_length=1024)
    checksum: Checksum
    byte_size: int = Field(ge=0, le=8589934592)
