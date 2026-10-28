"""Public API. Importing this package creates no environments or processes."""
__version__ = "0.9.0.dev1"


def train(*args, **kwargs):
    from .api import train as implementation
    return implementation(*args, **kwargs)


def evaluate(*args, **kwargs):
    from .api import evaluate as implementation
    return implementation(*args, **kwargs)


def load_policy(*args, **kwargs):
    from .api import load_policy as implementation
    return implementation(*args, **kwargs)

def train_reward_model(*args, **kwargs):
    from .rewards import train_reward_model as implementation
    return implementation(*args, **kwargs)

def load_reward_model(*args, **kwargs):
    from .rewards import load_reward_model as implementation
    return implementation(*args, **kwargs)


def run_experiment(*args, **kwargs):
    from .experiments import run_experiment as implementation
    return implementation(*args, **kwargs)


def run_suite(*args, **kwargs):
    from .experiments import run_suite as implementation
    return implementation(*args, **kwargs)


def check_experiment(*args, **kwargs):
    from .experiments import check_experiment as implementation
    return implementation(*args, **kwargs)


from .registry import AlgorithmSpec, register_algorithm
from .envs import EnvironmentSpec, register_environment

__all__ = ["train", "evaluate", "load_policy", "train_reward_model", "load_reward_model",
           "run_experiment", "run_suite", "check_experiment", "AlgorithmSpec", "register_algorithm",
           "EnvironmentSpec", "register_environment"]
