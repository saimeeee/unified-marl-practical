"""Run short native CityFlow rollouts through an isolated local HTTP service.

Requires the bundled CityFlow extension and the core CPU dependencies. This is
integration verification, not training or evidence that a policy beats a rule.
The service uses an ephemeral loopback port and its own output directory; it
never connects to, stops, or changes a user's running dashboard.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time
import traceback
import urllib.error
import urllib.request
import zipfile

from prepare_local import prepare
from server import Handler, Platform, TERMINAL
from training_bridge import export_traffic_training, register_experiment


HERE = Path(__file__).resolve().parent
CASES = (
    ("pressure-1x1", 1, 1, 80, "pressure"),
    ("pressure-2x3", 2, 3, 300, "pressure"),
    ("pressure-3x3", 3, 3, 400, "pressure"),
    ("pressure-4x4", 4, 4, 800, "pressure"),
    ("pressure-zero", 1, 1, 0, "pressure"),
    ("pressure-high", 2, 2, 3000, "pressure"),
    ("trained-2x2", 2, 2, 100, "trained"),
)


def write_report(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), status="running",
                  method="Real localhost HTTP controls and fresh native CityFlow workers",
                  scope="Short integration runs, not training or policy-quality evaluation", cases=[])
    report_path = output / "verification.json"
    write_report(report_path, report)
    server = None
    thread = None
    app = None
    try:
        import cityflow
        report["cityflow_extension"] = Path(cityflow.__file__).name
        prepared = prepare(HERE / "bundle", output / "prepared")
        models = json.loads(prepared.read_text(encoding="utf-8"))["models"]
        traffic_model = next(row for row in models if row["scenario"] == "traffic")
        catalog = output / "catalog.json"
        write_report(catalog, dict(models=[traffic_model]))
        app = Platform(catalog, output / "runs")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        server.app = app
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        # Ignore system proxies even if localhost is not in the host's NO_PROXY.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def api(path, data=None, expected=200):
            request = urllib.request.Request(url + path,
                data=None if data is None else json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            try:
                with opener.open(request, timeout=30) as response:
                    assert response.status == expected, (path, response.status, expected)
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                if exc.code != expected:
                    raise AssertionError(f"{path}: HTTP {exc.code}: {exc.read().decode('utf-8')}") from exc
                return json.load(exc)

        def wait(predicate, seconds=90):
            deadline = time.monotonic() + seconds
            state = {}
            while time.monotonic() < deadline:
                state = api("/api/state")
                if state.get("status") == "failed":
                    raise AssertionError(state.get("error"))
                if predicate(state):
                    return state
                time.sleep(.04)
            raise TimeoutError(f"Last state: status={state.get('status')}, error={state.get('error')}")

        def stable_pause():
            state = wait(lambda s: s["status"] == "ready" and s["mode"] == "paused")
            # An already-started native step may finish after pause. Require the
            # step to settle, then remain unchanged for several observations.
            deadline = time.monotonic() + 5
            same = 0
            last = state["frame"]["step"]
            while time.monotonic() < deadline:
                time.sleep(.15)
                state = api("/api/state")
                assert state["status"] == "ready" and state["mode"] == "paused", state.get("error")
                current = state["frame"]["step"]
                same = same + 1 if current == last else 0
                if same >= 4:
                    return current
                last = current
            raise AssertionError("Paused simulation continued to advance")

        schema = api("/api/scenarios")["traffic"]
        assert schema["fields"]["rows"]["max"] == 4
        invalid = dict(model=traffic_model["id"], seed=101,
                       parameters=dict(rows=3, cols=3, controller="trained"))
        api("/api/start", invalid, expected=400)
        assert api("/api/state")["status"] == "idle"
        report["incompatible_trained_grid_rejected"] = True
        digest = hashlib.sha256(Path(traffic_model["checkpoint"]).read_bytes()).hexdigest()
        for name, rows, cols, vehicles, controller in CASES:
            print(f"Verifying {name}", flush=True)
            item = dict(name=name, status="running")
            report["cases"].append(item)
            write_report(report_path, report)
            parameters = dict(rows=rows, cols=cols, vehicle_count=vehicles,
                              decision_limit=8, controller=controller)
            started = api("/api/start", dict(model=traffic_model["id"], seed=101,
                                             parameters=parameters), expected=201)
            sid = started["id"]
            item["session"] = sid

            def control(op, **values):
                return api("/api/control", dict(id=sid, op=op, **values))

            initial = wait(lambda s: s["status"] == "ready")
            assert initial["frame"]["step"] == 0 and initial["mode"] == "paused"
            assert initial["parameters"] == parameters
            assert stable_pause() == 0
            control("step")
            wait(lambda s: s["frame"]["step"] == 1)
            assert stable_pause() == 1
            control("speed", speed=8)
            control("run")
            wait(lambda s: s["frame"]["step"] >= 3)
            control("pause")
            stable_pause()
            control("run")
            for _ in range(2):
                partial = api("/api/export/" + sid)
                assert partial["frames"] and partial["request"]["parameters"] == parameters
                time.sleep(.06)
            final = wait(lambda s: s["status"] == "completed")
            trajectory = api("/api/export/" + sid)
            frames = trajectory["frames"]
            expected_algorithm = "rule/pressure" if controller == "pressure" else traffic_model["algorithm"]
            expected_source = "pressure_rule" if controller == "pressure" else "trained_checkpoint"
            expected_digest = None if controller == "pressure" else digest
            assert final["model"]["algorithm"] == expected_algorithm
            assert final["frame"]["decision_step"] == parameters["decision_limit"]
            assert len(frames) == final["frame_count"] == final["frame"]["step"] + 2
            assert frames[0]["step"] == 0 and frames[-1]["done"]
            assert all(f["checkpoint_sha256"] == expected_digest and f["policy_source"] == expected_source for f in frames)
            assert all(len([i for i in f["view"]["intersections"] if not i.get("virtual")]) == rows * cols for f in frames)
            has_vehicles = any(f["view"]["vehicles"] for f in frames)
            assert has_vehicles == bool(vehicles), (name, has_vehicles)
            generated = output / "runs" / sid / "scenario"
            flow = json.loads((generated / "flow.json").read_text(encoding="utf-8"))
            assert len(flow) == vehicles
            history = next(row for row in api("/api/history") if row["id"] == sid)
            assert history["parameters"] == parameters
            assert history["model"]["algorithm"] == expected_algorithm
            assert history["frame_count"] == len(frames)
            assert api("/api/health")["current"]["parameters"] == parameters
            # Reload metadata through a fresh service object to prove persistence,
            # rather than only observing the live in-memory session.
            reloaded = Platform(catalog, output / "runs")
            persisted = next(row for row in reloaded.history() if row["id"] == sid)
            assert persisted["parameters"] == parameters and persisted["model"]["algorithm"] == expected_algorithm
            app.session.thread.join(timeout=5)
            assert not app.session.thread.is_alive() and app.session.proc.poll() is not None
            item.update(status="passed", parameters=parameters, frames=len(frames),
                        final_physical_second=frames[-1]["step"], checkpoint_sha256=expected_digest,
                        initial_pause=True, single_step=True, pause=True, concurrent_export=True,
                        history_reload=True, worker_reaped=True)
            write_report(report_path, report)
            print(json.dumps(item), flush=True)

        # Exercise the generic inference protocol with a freshly trained model
        # on a grid that cannot use the bundled 2x2 application checkpoint.
        print("Verifying 1x2 training export -> registration -> live comparison", flush=True)
        bridge = dict(status="running", algorithm="pymarl/qmix", rows=1, cols=2)
        report["training_bridge"] = bridge
        write_report(report_path, report)
        package = export_traffic_training(traffic_model,
            dict(rows=1, cols=2, vehicle_count=80, decision_limit=8, controller="pressure"),
            seed=11, algorithm="pymarl/qmix")
        export_dir = output / "exported-training"
        with zipfile.ZipFile(io.BytesIO(package)) as archive:
            archive.extractall(export_dir)
        import torch
        from unified_marl.experiments import run_experiment
        torch.set_num_threads(1)
        training_dir = output / "trained-1x2"
        trained = run_experiment(export_dir / "experiment.yaml", output_dir=training_dir)
        assert trained["status"] == "success" and trained["train"]["updates"] == 2
        assert trained["train"]["parameter_delta_l2"] > 0
        state_dir = output / "registered"
        registered = register_experiment(training_dir, state_dir, "verify-traffic-qmix-1x2")
        original = Path(trained["train"]["checkpoint"])
        copied = state_dir / registered["checkpoint"]
        assert original.read_bytes() == copied.read_bytes()
        trained_digest = hashlib.sha256(copied.read_bytes()).hexdigest()
        # The model must use copied registration assets, not absolute paths in
        # its training checkpoint. Retain the original inputs under a new name.
        (export_dir / "scenario").rename(export_dir / "scenario-training-only")
        registered_catalog = prepare(HERE / "bundle", state_dir)
        app = Platform(registered_catalog, output / "runs")
        server.app = app
        assert registered["id"] in app.models
        assert app.models[registered["id"]]["inference_protocol"] == "generic_traffic_v1"
        bridge.update(updates=trained["train"]["updates"],
                      parameter_delta_l2=trained["train"]["parameter_delta_l2"],
                      checkpoint_sha256=trained_digest, registered_id=registered["id"],
                      registration_preserved_weights=True, source_scene_relocated=True, runs=[])
        write_report(report_path, report)
        for parameters in ({}, {"controller": "pressure"}):
            initial = api("/api/start", dict(model=registered["id"], seed=101,
                                             parameters=parameters), expected=201)
            sid = initial["id"]
            ready = wait(lambda s: s["status"] == "ready")
            assert ready["frame"]["step"] == 0 and ready["mode"] == "paused"
            assert stable_pause() == 0
            api("/api/control", dict(id=sid, op="step"))
            wait(lambda s: s["frame"]["step"] == 1)
            assert stable_pause() == 1
            api("/api/control", dict(id=sid, op="speed", speed=8))
            api("/api/control", dict(id=sid, op="run"))
            final = wait(lambda s: s["status"] == "completed")
            trajectory = api("/api/export/" + sid)
            frames = trajectory["frames"]
            is_rule = parameters.get("controller") == "pressure"
            expected_algorithm = "rule/pressure" if is_rule else "pymarl/qmix"
            expected_digest = None if is_rule else trained_digest
            expected_source = "pressure_rule" if is_rule else "trained_checkpoint"
            assert final["model"]["algorithm"] == expected_algorithm
            assert frames[-1]["decision_step"] == 8 and frames[-1]["done"]
            assert all(len([i for i in f["view"]["intersections"] if not i.get("virtual")]) == 2 for f in frames)
            assert all(f["checkpoint_sha256"] == expected_digest and f["policy_source"] == expected_source for f in frames)
            assert any(f["view"]["vehicles"] for f in frames)
            keys = {f["comparison_key"] for f in frames}
            assert len(keys) == 1
            history = next(row for row in api("/api/history") if row["id"] == sid)
            assert history["comparison_key"] == frames[-1]["comparison_key"]
            app.session.thread.join(timeout=5)
            assert not app.session.thread.is_alive() and app.session.proc.poll() is not None
            bridge["runs"].append(dict(session=sid, algorithm=expected_algorithm,
                checkpoint_sha256=expected_digest, comparison_key=frames[-1]["comparison_key"],
                parameters=final["parameters"], frames=len(frames), initially_paused=True,
                single_step=True, export_verified=True, history_verified=True, worker_reaped=True))
            write_report(report_path, report)
        comparison = api("/api/compare", dict(ids=[run["session"] for run in bridge["runs"]]))
        assert comparison["scene"] == "traffic" and comparison["seed"] == 101
        assert {run["model"]["algorithm"] for run in comparison["runs"]} == {"pymarl/qmix", "rule/pressure"}
        assert len({run["comparison_key"] for run in bridge["runs"]}) == 1
        bridge.update(status="passed", comparison=comparison)
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        if report["cases"] and report["cases"][-1]["status"] == "running":
            report["cases"][-1]["status"] = "failed"
        if report.get("training_bridge", {}).get("status") == "running":
            report["training_bridge"]["status"] = "failed"
        traceback.print_exc()
        raise
    finally:
        if app and app.session and app.session.thread.is_alive():
            if app.session.status not in TERMINAL:
                app.session.control("stop")
            app.session.thread.join(timeout=5)
        if server:
            server.shutdown()
            server.server_close()
        if thread:
            thread.join(timeout=5)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_report(report_path, report)
    print(f"PASS: {len(report['cases'])} native traffic cases and 1x2 training bridge; report: {report_path}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="New output directory for isolated verification data")
    args = parser.parse_args()
    verify(args.output)
