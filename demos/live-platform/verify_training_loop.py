"""Verify CPU training -> registration -> live MPE -> same-scene comparison.

Each model receives two optimizer updates. This proves the integration path,
not convergence or superiority to the centralized greedy reference controller.
The script creates a new output directory and its own ephemeral local server.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import traceback
import urllib.error
import urllib.request

import yaml

from prepare_local import prepare
from server import Handler, Platform, TERMINAL
from training_bridge import register_experiment


HERE = Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify(output):
    import torch
    from unified_marl.experiments import run_experiment

    torch.set_num_threads(1)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "verification.json"
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), status="running",
                  scope="Two-update CPU pipeline verification, not policy-quality evaluation",
                  training=[], runs=[])
    write_json(report_path, report)
    app = server = thread = None
    try:
        env_args = dict(scenario="simple_spread", n_agents=3, n_landmarks=3, episode_limit=25)
        state_dir = output / "state"
        for name in ("qmix", "vdn"):
            experiment = dict(schema_version=1, name=f"mpe-{name}-integration", algorithm=f"pymarl/{name}",
                              env="mpe", seed=11, device="cpu", env_args=env_args,
                              training=dict(t_max=100, max_updates=2, batch_size=2,
                                            buffer_size=8, rnn_hidden_dim=16),
                              evaluation=dict(episodes=1, seeds=[101]))
            source = output / f"{name}.yaml"
            source.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8")
            print(f"Training {name}: two CPU updates", flush=True)
            result = run_experiment(source, output_dir=output / f"train-{name}")
            trained = result["train"]
            assert result["status"] == "success" and trained["updates"] == 2
            assert trained["parameter_delta_l2"] > 0
            row = register_experiment(output / f"train-{name}", state_dir, f"verify-mpe-{name}")
            copied = state_dir / row["checkpoint"]
            original = Path(trained["checkpoint"])
            assert copied.read_bytes() == original.read_bytes()
            report["training"].append(dict(algorithm=result["algorithm"], model_id=row["id"],
                updates=trained["updates"], env_steps=trained["env_steps"],
                parameter_delta_l2=trained["parameter_delta_l2"],
                checkpoint_sha256=hashlib.sha256(copied.read_bytes()).hexdigest(),
                registration_preserved_weights=True))
            write_json(report_path, report)

        catalog = prepare(HERE / "bundle", state_dir)
        app = Platform(catalog, state_dir / "runs")
        assert all(item["model_id"] in app.models for item in report["training"])
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        server.app = app
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def api(path, data=None, expected=200):
            request = urllib.request.Request(base + path,
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

        def wait(predicate, timeout=60):
            deadline = time.monotonic() + timeout
            state = {}
            while time.monotonic() < deadline:
                state = api("/api/state")
                if state.get("status") == "failed":
                    raise AssertionError(state.get("error"))
                if predicate(state):
                    return state
                time.sleep(.04)
            raise TimeoutError(f"Session did not reach expected state: {state.get('status')}")

        def rollout(model_id, parameters=None, seed=101):
            parameters = parameters or {}
            state = api("/api/start", dict(model=model_id, seed=seed, parameters=parameters), expected=201)
            sid = state["id"]

            def control(operation, **values):
                return api("/api/control", dict(id=sid, op=operation, **values))

            initial = wait(lambda s: s["status"] == "ready")
            assert initial["mode"] == "paused" and initial["frame"]["step"] == 0
            time.sleep(.2)
            assert api("/api/state")["frame"]["step"] == 0
            control("step")
            wait(lambda s: s["frame"]["step"] == 1)
            time.sleep(.2)
            assert api("/api/state")["frame"]["step"] == 1
            control("speed", speed=8)
            control("run")
            final = wait(lambda s: s["status"] == "completed")
            trajectory = api("/api/export/" + sid)
            frames = trajectory["frames"]
            assert len(frames) == env_args["episode_limit"] + 1
            assert frames[0]["step"] == 0 and frames[-1]["done"]
            assert final["seed"] == seed and final["parameters"] == parameters
            is_rule = parameters.get("controller") == "greedy"
            expected_digest = None if is_rule else hashlib.sha256(Path(app.models[model_id]["checkpoint"]).read_bytes()).hexdigest()
            assert all(f["checkpoint_sha256"] == expected_digest for f in frames)
            assert all(f["policy_source"] == ("greedy_rule" if is_rule else "trained_checkpoint") for f in frames)
            assert len({f["comparison_key"] for f in frames}) == 1
            history = next(row for row in api("/api/history") if row["id"] == sid)
            assert history["comparison_key"] == frames[-1]["comparison_key"]
            app.session.thread.join(timeout=5)
            assert not app.session.thread.is_alive() and app.session.proc.poll() is not None
            item = dict(session=sid, model_id=model_id, algorithm=final["model"]["algorithm"],
                        seed=seed, parameters=parameters, comparison_key=frames[-1]["comparison_key"],
                        checkpoint_sha256=expected_digest, frames=len(frames), metrics=frames[-1]["metrics"],
                        initially_paused=True, single_step=True, export_verified=True, worker_reaped=True)
            report["runs"].append(item)
            write_json(report_path, report)
            return sid

        ids = [rollout(item["model_id"]) for item in report["training"]]
        ids.append(rollout(report["training"][0]["model_id"], {"controller": "greedy"}))
        comparison = api("/api/compare", dict(ids=ids))
        assert len(comparison["runs"]) == 3 and comparison["seed"] == 101
        assert {r["model"]["algorithm"] for r in comparison["runs"]} == {
            "pymarl/qmix", "pymarl/vdn", "rule/greedy_landmark"}
        assert len({r["comparison_key"] for r in report["runs"]}) == 1
        report["comparison"] = comparison
        different_seed = rollout(report["training"][0]["model_id"], seed=102)
        rejection = api("/api/compare", dict(ids=[ids[0], different_seed]), expected=400)
        assert "error" in rejection
        report["different_seed_comparison_rejected"] = True
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
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
        write_json(report_path, report)
    print(f"PASS: two trained MPE algorithms, registration, live comparison and baseline; {report_path}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New directory for training and verification outputs")
    verify(parser.parse_args().output)
