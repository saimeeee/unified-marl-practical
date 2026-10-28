"""Verify audit artifacts and source preservation; does not test MARL training."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re


def main():
    project = Path(__file__).resolve().parents[1]
    workspace = project.parent
    evidence = project / "docs/evidence"

    def load(name):
        return json.loads((evidence / name).read_text(encoding="utf-8-sig"))

    required = ["WORK_ORDER.md", "docs/INTEGRATION_SPEC.md", "docs/MIGRATION_MATRIX.csv",
                "docs/PLAN.md", "docs/STATUS.md", "docs/CODE_AUDIT.md"]
    assert all((project / n).is_file() and (project / n).stat().st_size for n in required)
    for path in evidence.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8-sig"))
    for path in (project / "tools").glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"))
    with (project / "docs/MIGRATION_MATRIX.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == load("inventory_summary.json")["matrix_rows"] == 340
    assert len({r["id"] for r in rows}) == len(rows)
    assert all((workspace / r["source_path"]).exists() for r in rows)
    assert all(r["migration_status"] == "NOT_STARTED" for r in rows)
    assert all(r[k] == "NO" for r in rows for k in ["implementation_verified", "local_test_verified", "real_environment_verified"])
    manifest = load("source_manifest.json") + load("supplemental_manifest.json")
    changed = [r["path"] for r in manifest if hashlib.sha256((workspace / r["path"]).read_bytes()).hexdigest() != r["sha256"]]
    assert not changed, changed
    probes = load("baseline_probes.json")
    assert len(probes) == 9
    assert all(r["returncode"] == 1 and not r["training_verified"] for r in probes)
    assert load("component_smoke.json")["result"] == "PASS"
    data = load("data_inventory.json")
    assert len(data) == 18 and sum(r["bytes"] for r in data) == 3736627170
    assert len(load("data_metadata_tokens.json")) == 18
    assert len(load("config_findings.json")) == 3
    assert not load("syntax_findings.json")
    assert all("/components/MAPT/" in r["source"] for r in load("local_import_findings.json"))
    for doc in project.rglob("*.md"):
        for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if not target.startswith(("https:", "http:", "#")):
                assert (doc.parent / target.split("#")[0]).exists(), (doc, target)
    # Stage-1-only checks: these assertions intentionally become obsolete after stage 2 begins.
    assert not (project / "src").exists()
    assert not (project / "pyproject.toml").exists()
    result = {"result": "PASS", "required_documents": len(required), "matrix_rows": len(rows),
              "source_paths_exist": True, "unique_ids": True, "all_migrations_not_started": True,
              "source_hash_records_verified": len(manifest), "changed_source_files": changed,
              "original_entry_probes": len(probes), "training_baselines_passed": 0,
              "component_smoke_passed": True, "data_archives_metadata_checked": len(data),
              "markdown_links_resolve": True, "stage_2_not_started": True}
    (evidence / "final_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
