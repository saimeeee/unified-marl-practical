"""Run existing tests while retaining their evidence in a new run directory.

Historical test modules otherwise write their evidence to old stage reports.
This pytest hook redirects only those output constants, never assertions or
test data. Usage: python tools/run_practical_regression.py [pytest arguments].
"""
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid
import os
import tempfile
import pytest

ROOT = Path(__file__).resolve().parents[1]


class EvidenceDestination:
    def __init__(self, directory):
        self.directory = directory

    def pytest_collection_modifyitems(self, items):
        for module in {item.module for item in items if hasattr(item, "module")}:
            if hasattr(module, "EVIDENCE"):
                destination = self.directory / "evidence" / module.__name__
                destination.mkdir(parents=True, exist_ok=True)
                module.EVIDENCE = destination


if __name__ == "__main__":
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = ROOT / "results" / "practical-v1-tests" / stamp
    directory.mkdir(parents=True)
    scratch = directory / "system-temp"
    scratch.mkdir()
    os.environ["TMP"] = os.environ["TEMP"] = str(scratch)
    tempfile.tempdir = str(scratch)
    print(f"Regression evidence: {directory}", flush=True)
    arguments = sys.argv[1:] or ["tests", "--ignore=tests/test_zmq.py", "-q"]
    arguments += ["--basetemp", str(directory / "tmp"), "--junitxml", str(directory / "junit.xml")]
    (directory / "command.txt").write_text("\n".join([sys.executable, *arguments]), encoding="utf-8")
    status = pytest.main(arguments, plugins=[EvidenceDestination(directory)])
    (directory / "exit-code.txt").write_text(str(status), encoding="utf-8")
    raise SystemExit(status)
