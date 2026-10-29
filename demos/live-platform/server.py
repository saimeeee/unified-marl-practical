"""Local live MARL dashboard. Each session owns an isolated worker.

The host defaults to loopback. Containers can explicitly bind 0.0.0.0 while
their published host port remains restricted to 127.0.0.1.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse, parse_qs
import uuid

HERE = Path(__file__).resolve().parent
TERMINAL = {"completed", "stopped", "failed"}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


class Session:
    def __init__(self, model, seed, speed, root):
        self.id = uuid.uuid4().hex
        self.folder = root / self.id
        self.folder.mkdir(parents=True)
        self.condition = threading.Condition()
        self.mode = "paused"
        self.speed = speed
        self.permits = 0
        self.touched = time.monotonic()
        self.seq = 0
        self.frame = None
        self.series = []
        self.status = "loading"
        self.error = None
        self.proc = None
        self.started = now()
        self.request = dict(model=model, seed=seed, session=self.id, created_at=self.started)
        save(self.folder / "request.json", self.request)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.condition:
            return dict(id=self.id, seq=self.seq, status=self.status, mode=self.mode,
                        speed=self.speed, seed=self.request["seed"], error=self.error,
                        model={k: self.request["model"][k] for k in ("id", "label", "scenario", "algorithm")},
                        frame=self.frame, series=list(self.series), started_at=self.started)

    def changed(self):
        self.seq += 1
        self.condition.notify_all()

    def control(self, operation, speed=None):
        with self.condition:
            if self.status in TERMINAL:
                raise ValueError("本次运行已结束，请重新初始化。")
            if operation in {"run", "step"} and self.status != "ready":
                raise ValueError("请等待模型和环境加载完成。")
            if operation == "speed":
                self.speed = speed
            elif operation == "run":
                self.mode, self.permits = "running", 0
            elif operation == "pause":
                self.mode, self.permits = "paused", 0
            elif operation == "step":
                self.mode = "paused"
                self.permits = min(1, self.permits + 1)
            elif operation == "stop":
                self.mode = "stopping"
                if self.proc and self.proc.poll() is None:
                    self.proc.terminate()
            else:
                raise ValueError("未知控制操作")
            self.touched = time.monotonic()
            self.changed()

    def run(self):
        events = queue.Queue(maxsize=4)
        with (self.folder / "worker.log").open("w", encoding="utf-8") as log:
            try:
                with self.condition:
                    if self.mode == "stopping":
                        return
                    self.proc = subprocess.Popen([sys.executable, "-u", "-B", str(HERE / "worker.py"),
                        str(self.folder / "request.json")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=log, text=True, encoding="utf-8", bufsize=1,
                        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})

                def read():
                    try:
                        for line in self.proc.stdout:
                            events.put(json.loads(line))
                    except Exception as exc:
                        events.put({"error": str(exc)})
                    finally:
                        events.put(None)

                threading.Thread(target=read, daemon=True).start()
                with (self.folder / "frames.jsonl").open("w", encoding="utf-8") as trace:
                    while True:
                        packet = events.get(timeout=120)
                        if self.mode == "stopping":
                            break
                        if packet is None:
                            raise RuntimeError("仿真进程提前退出，详见本次运行日志。")
                        if "error" in packet:
                            raise RuntimeError(packet["error"])
                        with self.condition:
                            trace.write(json.dumps(packet, ensure_ascii=False, allow_nan=False) + "\n")
                            trace.flush()
                            self.frame = packet
                            self.status = "finalizing" if packet.get("done") else "ready"
                            self.series.append(dict(step=packet["step"], value=packet["chart"]["value"]))
                            self.series = self.series[-1500:]
                            self.changed()
                            if packet.get("done"):
                                break
                            while self.mode == "paused" and not self.permits:
                                self.condition.wait(timeout=10)
                                if time.monotonic() - self.touched > 1800:
                                    self.mode = "stopping"
                            if self.mode == "stopping":
                                break
                            if self.mode == "running":
                                self.condition.wait(timeout=0.35 / self.speed)
                                if self.mode != "running":
                                    while self.mode == "paused" and not self.permits:
                                        self.condition.wait(timeout=10)
                                        if time.monotonic() - self.touched > 1800:
                                            self.mode = "stopping"
                            if self.mode == "stopping":
                                break
                            if self.permits:
                                self.permits -= 1
                        self.proc.stdin.write("advance\n")
                        self.proc.stdin.flush()
            except Exception as exc:
                with self.condition:
                    if self.mode != "stopping":
                        self.status, self.error = "failed", str(exc)
            finally:
                if self.proc:
                    if self.proc.poll() is None:
                        try:
                            self.proc.stdin.write("stop\n")
                            self.proc.stdin.flush()
                            self.proc.wait(timeout=3)
                        except (OSError, subprocess.TimeoutExpired):
                            self.proc.kill()
                    self.proc.wait()
                with self.condition:
                    if self.status not in TERMINAL:
                        self.status = "completed" if self.frame and self.frame.get("done") else "stopped"
                    self.mode = "paused"
                    save(self.folder / "result.json", {**self.snapshot(), "finished_at": now(),
                         "checkpoint_sha256": self.frame.get("checkpoint_sha256") if self.frame else None})
                    self.changed()


class Platform:
    def __init__(self, catalog, output):
        self.catalog = json.loads(catalog.read_text(encoding="utf-8"))
        self.models = {row["id"]: row for row in self.catalog["models"]}
        if len(self.models) != len(self.catalog["models"]):
            raise ValueError("Duplicate model ids")
        for row in self.models.values():
            if row["scenario"] not in {"warehouse", "mpe", "traffic", "powergrid"}:
                raise ValueError("Unsupported scene adapter")
            for key in ("checkpoint", "config"):
                if row.get(key):
                    row[key] = str((catalog.parent / row[key]).resolve())
                    if not Path(row[key]).is_file():
                        raise FileNotFoundError(row[key])
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.session = None

    def start(self, data):
        model = self.models.get(data.get("model"))
        seed = data.get("seed", 101)
        if model is None or type(seed) is not int or not 0 <= seed <= 2147483647:
            raise ValueError("请选择有效模型，种子为0至2147483647的整数。")
        with self.lock:
            if self.session and self.session.status in TERMINAL:
                self.session.thread.join(timeout=1)
            if self.session and self.session.thread.is_alive():
                raise ValueError("请先结束当前运行，再切换场景。")
            self.session = Session(model, seed, 1, self.output)
            return self.session.snapshot()


class Handler(BaseHTTPRequestHandler):
    server_version = "MARLLive/1"

    def log_message(self, *args):
        return

    def send(self, value, status=200, content_type="application/json; charset=utf-8", download=None):
        body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def allowed_host(self):
        return self.headers.get("Host", "").split(":")[0] in {"127.0.0.1", "localhost"}

    def do_GET(self):
        if not self.allowed_host():
            return self.send({"error": "Use localhost or 127.0.0.1"}, 403)
        parsed = urlparse(self.path)
        app = self.server.app
        static = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8")}
        if parsed.path in static:
            file, mime = static[parsed.path]
            return self.send((HERE / "static" / file).read_bytes(), content_type=mime)
        if parsed.path == "/api/catalog":
            return self.send({"models": [{k: v for k, v in row.items() if k not in {"checkpoint", "config"}}
                                          for row in app.models.values()], "mode": "live_inference"})
        if parsed.path == "/api/state":
            session = app.session
            if not session:
                return self.send({"status": "idle", "seq": 0})
            after = parse_qs(parsed.query).get("after", ["-1"])[0]
            try:
                after = int(after)
            except ValueError:
                return self.send({"error": "Invalid sequence"}, 400)
            with session.condition:
                session.touched = time.monotonic()
                if after == session.seq and session.status not in TERMINAL:
                    session.condition.wait(timeout=12)
                value = session.snapshot()
            return self.send(value)
        if parsed.path == "/api/history":
            rows = []
            for path in sorted(app.output.glob("*/result.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
                item = json.loads(path.read_text(encoding="utf-8"))
                rows.append({k: item.get(k) for k in ("id", "model", "seed", "status", "started_at")})
            return self.send(rows)
        match = re.fullmatch(r"/api/export/([0-9a-f]{32})", parsed.path)
        if match:
            folder = app.output / match[1]
            if not (folder / "request.json").is_file():
                return self.send({"error": "Run not found"}, 404)
            # The current writer uses this same lock for write + flush. Older
            # sessions have fully exited before a replacement can be started.
            session = app.session
            lock = session.condition if session and session.id == match[1] else threading.Lock()
            with lock:
                raw = (folder / "frames.jsonl").read_text(encoding="utf-8") if (folder / "frames.jsonl").is_file() else ""
            frames = [json.loads(line) for line in raw.splitlines()]
            return self.send(json.dumps(dict(request=json.loads((folder / "request.json").read_text(encoding="utf-8")), frames=frames), ensure_ascii=False).encode(), download=match[1]+".json")
        return self.send({"error": "Not found"}, 404)

    def do_POST(self):
        origin = self.headers.get("Origin")
        if not self.allowed_host() or (origin and urlparse(origin).netloc != self.headers.get("Host")):
            return self.send({"error": "Cross-origin request denied"}, 403)
        if self.headers.get_content_type() != "application/json":
            return self.send({"error": "JSON required"}, 415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Request size invalid")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected JSON object")
            if self.path == "/api/start":
                return self.send(self.server.app.start(data), 201)
            if self.path == "/api/control":
                session = self.server.app.session
                if not session or data.get("id") != session.id:
                    raise ValueError("运行已切换，请刷新页面。")
                speed = data.get("speed", 1)
                if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or not .25 <= speed <= 8:
                    raise ValueError("速度必须在0.25至8之间")
                session.control(data.get("op"), speed)
                return self.send(session.snapshot())
            return self.send({"error": "Not found"}, 404)
        except (ValueError, TypeError, KeyError) as exc:
            return self.send({"error": str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--host", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1",
                        help="Use 0.0.0.0 only inside a container with a loopback-only published port")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.app = Platform(args.catalog.resolve(), args.output.resolve())
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    print(f"MARL LIVE bind={args.host}:{args.port}; browser=http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server.app.session and server.app.session.thread.is_alive():
            if server.app.session.status not in TERMINAL:
                server.app.session.control("stop")
            server.app.session.thread.join(timeout=5)
        server.server_close()


if __name__ == "__main__":
    main()
