import hashlib
import json

import pytest

from apps.segman_generic_runner import checked_paths


def case(root, key, version="generic-case-binding/1.0.0"):
    directory = root / ".local/fairvlmed-runs" / key
    directory.mkdir(parents=True)
    image = directory / "input.png"
    image.write_bytes(b"synthetic-image-binding-only")
    (directory / "binding.json").write_text(
        json.dumps(
            dict(
                schema_version=version,
                execution_namespace=key,
                image=dict(checksum="sha256:" + hashlib.sha256(image.read_bytes()).hexdigest()),
            )
        )
    )
    return image, directory / "segman-r03/prediction/result.json"


@pytest.mark.parametrize("name", ["case-alpha", "case-beta"])
@pytest.mark.parametrize("version", ["generic-case-binding/1.0.0", "generic-case-binding/1.1.0"])
def test_arbitrary_case_workspace_without_old_two_case_root(tmp_path, name, version):
    image, output = case(tmp_path, name, version)
    result = checked_paths(tmp_path, image, output)
    assert result == (image, output, output.parent.parent)
    assert not result[-1].exists()


@pytest.mark.parametrize("mode", ["cross_case", "image", "existing_job", "output"])
@pytest.mark.parametrize("version", ["generic-case-binding/1.0.0", "generic-case-binding/1.1.0"])
def test_path_and_byte_binding_not_relaxed(tmp_path, mode, version):
    image, output = case(tmp_path, "case-a", version)
    if mode == "cross_case":
        _, output = case(tmp_path, "case-b")
    elif mode == "image":
        image.write_bytes(b"changed")
    elif mode == "existing_job":
        output.parent.parent.mkdir()
    else:
        output = output.parent / "other.json"
    with pytest.raises(ValueError, match="generic_segman"):
        checked_paths(tmp_path, image, output)


def test_unknown_binding_version_still_rejected(tmp_path):
    image, output = case(tmp_path, "case-a", "generic-case-binding/99.0.0")
    with pytest.raises(ValueError, match="generic_segman_image_binding"):
        checked_paths(tmp_path, image, output)
