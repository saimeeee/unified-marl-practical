"""Prepare local asset paths and start this computer's live simulation service."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=HERE / "bundle")
    parser.add_argument("--output", type=Path, default=HERE / "local-state")
    parser.add_argument("--host", choices=["127.0.0.1", "0.0.0.0"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    bundle, output = args.bundle.expanduser().resolve(), args.output.expanduser().resolve()
    for scene in ("warehouse", "mpe", "traffic", "powergrid"):
        checkpoint = bundle / "models" / scene / "policy.pt"
        if not checkpoint.is_file():
            raise SystemExit(f"Missing local model: {checkpoint}. Use the complete GitHub local-platform download.")
    output.mkdir(parents=True, exist_ok=True)
    prepare = HERE / "prepare_local.py"
    if not prepare.is_file():
        raise SystemExit(f"Incomplete source download: missing {prepare}")
    subprocess.run([sys.executable, "-B", str(prepare), "--bundle", str(bundle), "--output", str(output)], check=True)
    catalog = output / "models.json"
    if not catalog.is_file():
        raise SystemExit("Local asset preparation did not produce models.json")
    print(f"Local inference: {bundle}\nRun data: {output / 'runs'}", flush=True)
    os.execv(sys.executable, [sys.executable, "-u", "-B", str(HERE / "server.py"),
        "--host", args.host, "--port", str(args.port), "--catalog", str(catalog), "--output", str(output / "runs")])


if __name__ == "__main__":
    main()
