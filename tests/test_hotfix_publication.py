"""Publication guards with an in-memory GitHub API; never contact the service."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def publisher(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools/publish_hotfix_github.py"
    spec = importlib.util.spec_from_file_location("hotfix_publication_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("GITHUB_REPOSITORY", module.REPOSITORY)
    monkeypatch.setenv("GITHUB_SHA", "1" * 40)
    return module


def test_remote_assets_must_have_exact_names_sizes_hashes_and_uploaded_state(publisher):
    expected = [{"name": "a.zip", "bytes": 3, "sha256": "a" * 64}]
    good = {"assets": [{"name": "a.zip", "size": 3, "state": "uploaded", "digest": "sha256:" + "a" * 64}]}
    assert publisher.check_assets(good, expected)
    for field, value in [("size", 4), ("state", "uploading"), ("digest", None)]:
        bad = copy.deepcopy(good)
        bad["assets"][0][field] = value
        assert not publisher.check_assets(bad, expected)
    bad = copy.deepcopy(good)
    bad["assets"].append(dict(bad["assets"][0], name="extra.zip"))
    assert not publisher.check_assets(bad, expected)


def test_different_published_release_is_never_overwritten(publisher, monkeypatch, tmp_path):
    (tmp_path / "release-notes.md").write_text("notes", encoding="utf-8")
    monkeypatch.setattr(publisher, "PAYLOAD", tmp_path)
    calls = []
    def fake(path, method="GET", data=None, binary=False):
        calls.append(method)
        if path == "repos/" + publisher.REPOSITORY:
            return {"full_name": publisher.REPOSITORY, "visibility": "public"}
        return {"draft": False, "assets": []}
    monkeypatch.setattr(publisher, "api", fake)
    with pytest.raises(RuntimeError, match="refusing replacement"):
        publisher.publish({"assets": [{"name": "new.zip", "bytes": 1, "sha256": "a" * 64}]})
    assert set(calls) == {"GET"}


def test_publish_only_after_verified_uploads_and_preserves_visibility(publisher, monkeypatch, tmp_path):
    (tmp_path / "release-notes.md").write_text("notes", encoding="utf-8")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/a.zip").write_bytes(b"abc")
    monkeypatch.setattr(publisher, "PAYLOAD", tmp_path)
    expected = [{"name": "a.zip", "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}]
    state = {"release": None, "visibility": "public"}
    def fake(path, method="GET", data=None, binary=False):
        if path == "repos/" + publisher.REPOSITORY:
            return {"full_name": publisher.REPOSITORY, "visibility": state["visibility"]}
        if "/tags/" in path:
            return state["release"]
        if method == "POST" and path.endswith("/releases"):
            assert data["draft"]
            state["release"] = {"id": 7, "draft": True, "assets": [], "target_commitish": "1" * 40}
        elif path.startswith("https://uploads.github.com/"):
            assert binary and data == b"abc" and state["release"]["draft"]
            state["release"]["assets"] = [{"name": "a.zip", "size": 3, "state": "uploaded", "digest": "sha256:" + expected[0]["sha256"]}]
        elif method == "PATCH":
            assert publisher.check_assets(state["release"], expected)
            state["release"]["draft"] = False
        return copy.deepcopy(state["release"])
    monkeypatch.setattr(publisher, "api", fake)
    release, visibility = publisher.publish({"assets": expected})
    assert not release["draft"] and visibility == "public"


def test_cross_repository_context_is_rejected_before_any_api_call(publisher, monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "other/repository")
    monkeypatch.setattr(publisher, "api", lambda *a, **k: pytest.fail("API must not be called"))
    with pytest.raises(RuntimeError, match="intended repository"):
        publisher.publish({"assets": []})
