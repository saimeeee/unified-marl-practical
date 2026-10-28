"""Verify and publish the prepared 0.8.2 assets. No installs or simulations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "saimeeee/unified-marl"
TAG = "v0.8.2-engineering.1"
PAYLOAD = ROOT / "delivery/hotfix-release/0.8.2"


def verify_payload():
    plan = json.loads((PAYLOAD / "release-plan.json").read_text(encoding="utf-8"))
    if plan["repository"] != REPOSITORY or plan["tag"] != TAG:
        raise ValueError("Unexpected release destination")
    names = set()
    for item in plan["assets"]:
        name = item["name"]
        if Path(name).name != name or name in {".", ".."} or ":" in name or "/" in name or "\\" in name or name in names:
            raise ValueError("Unsafe or duplicate asset name")
        names.add(name)
        file = PAYLOAD / "assets" / name
        if not file.resolve().is_relative_to((PAYLOAD / "assets").resolve()):
            raise ValueError("Asset path escapes the payload")
        if file.stat().st_size != item["bytes"] or hashlib.sha256(file.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Release asset mismatch: " + name)
    return plan


def api(path, method="GET", data=None, binary=False):
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Authenticated GitHub connection is required; no token is printed or stored")
    url = path if path.startswith("https://uploads.github.com/") else "https://api.github.com/" + path
    if not url.startswith(("https://api.github.com/", "https://uploads.github.com/")):
        raise ValueError("Unexpected API host")
    headers = {"Authorization": "Bearer " + token, "User-Agent": "unified-marl-release",
               "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if data is not None:
        headers["Content-Type"] = "application/octet-stream" if binary else "application/json"
        if not binary:
            data = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, method=method, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404 and method == "GET":
            return None
        raise RuntimeError(f"GitHub API returned HTTP {error.code} for {method} request") from None


def check_assets(release, expected):
    actual = {item["name"]: item for item in release.get("assets", [])}
    if set(actual) != {item["name"] for item in expected}:
        return False
    return all(actual[item["name"]].get("state") == "uploaded"
               and actual[item["name"]]["size"] == item["bytes"]
               and actual[item["name"]].get("digest") == "sha256:" + item["sha256"] for item in expected)


def publish(plan):
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
        raise RuntimeError("Publishing requires the intended repository context")
    commit = os.environ.get("GITHUB_SHA")
    if not commit:
        raise RuntimeError("An exact source commit is required")
    repo = api("repos/" + REPOSITORY)
    if not repo or repo["full_name"] != REPOSITORY:
        raise RuntimeError("Repository access check failed")
    visibility = repo.get("visibility")
    notes = (PAYLOAD / "release-notes.md").read_text(encoding="utf-8")
    release = api(f"repos/{REPOSITORY}/releases/tags/{TAG}")
    if release and not release["draft"]:
        if check_assets(release, plan["assets"]):
            return release, visibility
        raise RuntimeError("A published tag already exists with different assets; refusing replacement")
    if release:
        if release["target_commitish"] != commit or release.get("body", "").replace("\r\n", "\n").strip() != notes.strip():
            raise RuntimeError("An existing draft differs from this source commit or notes")
    else:
        release = api(f"repos/{REPOSITORY}/releases", "POST", {
            "tag_name": TAG, "target_commitish": commit, "name": "unified-marl 0.8.2 工程修复版",
            "body": notes, "draft": True, "prerelease": False,
        })
    uploaded = {item["name"]: item for item in release.get("assets", [])}
    for item in plan["assets"]:
        if item["name"] in uploaded:
            existing = uploaded[item["name"]]
            if existing["size"] != item["bytes"] or existing.get("digest") != "sha256:" + item["sha256"]:
                raise RuntimeError("Existing draft asset differs: " + item["name"])
            continue
        path = PAYLOAD / "assets" / item["name"]
        api(f"https://uploads.github.com/repos/{REPOSITORY}/releases/{release['id']}/assets?name="
            + urllib.parse.quote(item["name"], safe=""), "POST", path.read_bytes(), binary=True)
        print("Uploaded " + item["name"], flush=True)
    for attempt in range(8):
        release = api(f"repos/{REPOSITORY}/releases/{release['id']}")
        if check_assets(release, plan["assets"]):
            break
        if attempt == 7:
            raise RuntimeError("Remote release asset hashes did not become consistent")
        time.sleep(2)
    release = api(f"repos/{REPOSITORY}/releases/{release['id']}", "PATCH", {"draft": False})
    if release["draft"] or not check_assets(release, plan["assets"]):
        raise RuntimeError("Publication verification failed")
    if api("repos/" + REPOSITORY).get("visibility") != visibility:
        raise RuntimeError("Repository visibility changed unexpectedly")
    return release, visibility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-only", action="store_true")
    group.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    plan = verify_payload()
    if args.verify_only:
        print(json.dumps({"status": "PAYLOAD_VERIFIED", "assets": len(plan["assets"]), "published": False}))
        return
    release, visibility = publish(plan)
    receipt = {"status": "PUBLISHED", "tag": TAG, "repository": REPOSITORY,
               "url": release["html_url"], "release_id": release["id"], "source_commit": release["target_commitish"],
               "visibility_preserved": visibility, "asset_hashes_verified": True, "assets": plan["assets"]}
    output = ROOT / ".work/hotfix-release-082-receipt.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
