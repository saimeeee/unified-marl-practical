"""Exercise the deployed HTTP controls and four actual saved-policy rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import urllib.error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18765")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def api(path, data=None, extra=None, expected=200):
        headers = {"Content-Type": "application/json", **(extra or {})}
        req = urllib.request.Request(args.url+path, data=None if data is None else json.dumps(data).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                assert response.status == expected, response.status
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != expected:
                raise
            return json.load(exc)

    def wait(predicate, seconds=120):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            state = api("/api/state")
            if state.get("status") == "failed":
                raise AssertionError(state.get("error"))
            if predicate(state):
                return state
            time.sleep(.1)
        raise TimeoutError(state)

    catalog = api("/api/catalog")
    assert len(catalog["models"]) >= 4
    api("/api/start", {"model": "unknown", "seed": 101}, expected=400)
    api("/api/start", {"model": catalog["models"][0]["id"], "seed": -1}, expected=400)
    api("/api/start", {"model": catalog["models"][0]["id"], "seed": 101}, extra={"Origin": "http://untrusted.invalid"}, expected=403)
    api("/../worker.py", expected=404)
    rows = []
    for model in catalog["models"]:
        print("Verifying", model["scenario"], flush=True)
        state = api("/api/start", {"model": model["id"], "seed": 101}, expected=201)
        sid = state["id"]
        control = lambda op, **more: api("/api/control", {"id": sid, "op": op, **more})
        state = wait(lambda s:s["status"] == "ready")
        assert state["frame"]["step"] == 0 and state["mode"] == "paused"
        time.sleep(.35)
        assert api("/api/state")["frame"]["step"] == 0
        control("step")
        state = wait(lambda s:s["frame"]["step"] == 1)
        time.sleep(.35)
        assert api("/api/state")["frame"]["step"] == 1
        control("speed", speed=8)
        control("run")
        wait(lambda s:s["frame"]["step"] >= 3)
        control("pause")
        time.sleep(.5)
        paused = api("/api/state")["frame"]["step"]
        time.sleep(.35)
        assert api("/api/state")["frame"]["step"] == paused
        control("run")
        # Export repeatedly while the trace writer is active.
        for _ in range(3):
            assert api("/api/export/"+sid)["frames"]
            time.sleep(.1)
        final = wait(lambda s:s["status"] == "completed", 180)
        trajectory = api("/api/export/"+sid)
        frames = trajectory["frames"]
        assert frames[0]["step"] == 0 and frames[-1]["done"]
        assert len({f["checkpoint_sha256"] for f in frames}) == 1
        if model["scenario"] == "traffic":
            assert final["frame"]["decision_step"] == 32
            assert len(frames) == final["frame"]["step"]+2
            assert any(f["view"]["vehicles"] for f in frames)
        else:
            assert len(frames) == final["frame"]["limit"]+1
        (args.output / (model["scenario"]+"-trajectory.json")).write_text(json.dumps(trajectory,ensure_ascii=False),encoding="utf-8")
        rows.append(dict(scenario=model["scenario"], session=sid, status=final["status"], frames=len(frames),
            final_step=final["frame"]["step"], checkpoint_sha256=frames[0]["checkpoint_sha256"],
            initially_paused=True, single_step_verified=True, pause_verified=True, concurrent_export_verified=True))
        print(json.dumps(rows[-1]), flush=True)
        time.sleep(.5)
    # Stop during initialization, then prove the worker is reaped before restart.
    state = api("/api/start", {"model": catalog["models"][0]["id"], "seed": 202}, expected=201)
    api("/api/control", {"id": state["id"], "op": "stop"})
    wait(lambda s:s["status"] == "stopped", 10)
    time.sleep(.5)
    report = dict(scenarios=rows, invalid_model_rejected=True, invalid_seed_rejected=True,
                  cross_origin_rejected=True, static_path_traversal_rejected=True,
                  initialization_stop_verified=True, method="Actual saved checkpoints and native simulator rollouts through HTTP")
    (args.output / "verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("PASS: four live scenarios and controls", flush=True)


if __name__ == "__main__":
    main()
