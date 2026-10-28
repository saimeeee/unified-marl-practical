"""One bounded, independent evaluation process using immutable policy snapshots."""
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
import time


def evaluation_worker(checkpoint, episodes, device, seed, writer):
    try:
        import torch
        torch.set_num_threads(1)
        from unified_marl import evaluate
        result = evaluate(checkpoint, episodes=episodes, device=device, seed=seed)
        writer.send({"status": "PASS", "evaluation": asdict(result)})
    except BaseException as exc:
        writer.send({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        writer.close()


class PeriodicEvaluator:
    def __init__(self, config, directory):
        self.interval = config.get("evaluation_interval_updates", 0)
        self.episodes = config.get("evaluation_episodes", 2)
        self.timeout = config.get("evaluation_timeout_s", 45)
        if type(self.interval) is not int or self.interval < 0:
            raise ValueError("evaluation_interval_updates must be a nonnegative integer")
        if type(self.episodes) is not int or self.episodes < 1:
            raise ValueError("evaluation_episodes must be a positive integer")
        if type(self.timeout) not in (int, float) or not 0 < self.timeout <= 300:
            raise ValueError("evaluation_timeout_s must be finite and in (0, 300]")
        self.directory = Path(directory)
        self.device = config.get("evaluation_device", "cpu")
        self.seed = config["seed"]
        self.pending = None
        self.records = []
        self.skipped_busy = []

    def submit(self, checkpoint_factory, update, version):
        self.poll()
        if not self.interval or update % self.interval:
            return
        if self.pending is not None:
            self.skipped_busy.append(update)
            return
        checkpoint = checkpoint_factory()
        context = mp.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(target=evaluation_worker, args=(checkpoint, self.episodes, self.device, self.seed, writer),
                                  name="unified-marl-evaluator")
        try:
            process.start()
        except BaseException:
            reader.close()
            writer.close()
            raise
        writer.close()
        self.pending = dict(process=process, reader=reader, started=time.monotonic(), update=update, version=version, checkpoint=checkpoint)

    def poll(self):
        if self.pending is None:
            return
        job = self.pending
        process, reader = job["process"], job["reader"]
        if reader.poll():
            try:
                result = reader.recv()
            except EOFError as exc:
                raise RuntimeError("Evaluation process exited without a result") from exc
            process.join(1)
            if process.is_alive() or process.exitcode != 0:
                raise RuntimeError("Evaluation process did not exit cleanly")
            reader.close()
            self.pending = None
            record = dict(update=job["update"], policy_version=job["version"], checkpoint=job["checkpoint"], **result)
            self.records.append(record)
            from unified_marl.api import _write_json
            _write_json(self.directory / f"evaluation-{job['update']}.json", record)
            if result["status"] != "PASS":
                raise RuntimeError("Evaluation failed: " + result["error"])
        elif time.monotonic() - job["started"] > self.timeout:
            raise TimeoutError("Evaluation process timeout")
        elif process.exitcode is not None:
            raise RuntimeError(f"Evaluation process exited without a result: {process.exitcode}")

    def finish(self):
        while self.pending is not None:
            self.pending["reader"].poll(0.05)
            self.poll()

    def close(self):
        forced = False
        if self.pending is not None:
            process = self.pending["process"]
            if process.is_alive():
                forced = True
                process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join(1)
            self.pending["reader"].close()
            self.pending = None
        return dict(records=self.records, skipped_busy_updates=self.skipped_busy, forced=forced)
