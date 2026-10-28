from .episode_runner import EpisodeRunner
from .parallel_runner import ParallelRunner

REGISTRY = {"episode": EpisodeRunner, "parallel": ParallelRunner}
