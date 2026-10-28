import json
import logging
from pathlib import Path


class Logger:
    def __init__(self, path=None):
        self.console_logger = logging.getLogger("unified_marl")
        self.path = Path(path) if path else None
        self.latest = {}

    def log_stat(self, key, value, step):
        value = float(value)
        self.latest[key] = value
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": int(step), "name": key, "value": value}, allow_nan=False) + "\n")

    def record(self, metrics, step):
        for key, value in metrics.items():
            self.log_stat(key, value, step)
