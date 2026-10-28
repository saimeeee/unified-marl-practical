"""powergrid application facade. See the Chinese application guide."""
def train(config, output_dir=None, *, overrides=None, resume=None):
    from .service import train as implementation
    return implementation("powergrid", config, output_dir, overrides=overrides, resume=resume)


def evaluate(checkpoint, *, config=None, episodes=1, seed=None, device="cpu"):
    from .service import evaluate as implementation
    return implementation("powergrid", checkpoint, config=config, episodes=episodes, seed=seed, device=device)


def load_policy(checkpoint, *, config=None, device="cpu"):
    from .service import load_policy as implementation
    return implementation("powergrid", checkpoint, config=config, device=device)
