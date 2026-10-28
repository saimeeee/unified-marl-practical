from pathlib import Path
import os


def make_smac(**kwargs):
    try:
        from smac.env import StarCraft2Env
    except ImportError as exc:
        raise ImportError("SMAC is optional. Install unified-marl[smac] and configure a StarCraft II installation.") from exc
    path = os.environ.get("SC2PATH")
    if path and not Path(path).is_dir():
        raise FileNotFoundError(f"SC2PATH is not a directory: {path}")
    return StarCraft2Env(**kwargs)
