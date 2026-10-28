"""Source and adaptation record for the bundled PowerGrid simulator."""

SOURCE = (
    "marl平台代码/易才润/论文/SADMA/sadma源代码开源版本/"
    "envs/powergrid/smart_grid.py and energy_models.py"
)

ADAPTATIONS = (
    "Moved the environment into the unified_marl package and replaced the SMAC "
    "base import with unified_marl.envs.multiagentenv.MultiAgentEnv; kept all "
    "external climate/building assets as explicit caller paths; added early "
    "asset checks; replaced process-wide quit() calls with chained RuntimeError; "
    "made episode_limit configurable while retaining source default 92; report "
    "the actual action-space shape instead of the hard-coded four; build action "
    "dimensions in the order consumed by Building.step and save their physical "
    "names; parameterize building sampling weights while preserving the source "
    "default distribution where its nine-ID assumptions apply; preserve "
    "agent/action row alignment by returning all agents in action order when "
    "percent_rl=1. The source "
    "still builds pandapower case33bw and generates buildings from its JSON/CSV "
    "assets; its net_path and agent_path arguments remain unused in the source. "
    "The adapter only supports this source case33bw topology, not arbitrary grids."
)

PPO_DIFFERENCES = (
    "The new bounded_ppo engine reuses the public source's shared recurrent "
    "Gaussian actor, central recurrent critic, PPO clipping, entropy, and GAE "
    "structure, but it is a separate adapter implementation. It adds tanh plus "
    "affine action-bound conversion and the matching transform Jacobian in "
    "stored and recomputed physical-action log probabilities, sums log-probs "
    "over action dimensions once, and separates terminal bootstrap masks from "
    "truncation trace masks. The source continuous branch returns an unbounded "
    "Normal sample and does not correct its log probability for action bounds; "
    "therefore this package does not claim a direct migration or parity with "
    "the source continuous PPO implementation. Entropy regularization uses the "
    "base Gaussian entropy in latent space."
)
