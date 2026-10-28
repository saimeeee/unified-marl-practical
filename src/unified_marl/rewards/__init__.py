"""Lazy access to delivered preference reward models."""
def train_reward_model(*args, **kwargs):
    from .service import train_reward_model as implementation
    return implementation(*args, **kwargs)

def load_reward_model(*args, **kwargs):
    from .service import load_reward_model as implementation
    return implementation(*args, **kwargs)
