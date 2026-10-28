from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class UpdateContext:
    env_steps: int
    episodes: int
    update_step: int


class LearnerAPI:
    def update(self, batch, context):
        if batch["filled"][:, :-1].sum() <= 0:
            raise ValueError("Cannot update from an empty transition batch")
        metrics = self.train(batch, context.env_steps, context.episodes)
        if not all(torch.isfinite(torch.tensor(v)) for v in metrics.values()):
            raise FloatingPointError(f"Non-finite learner metrics: {metrics}")
        return metrics

    @staticmethod
    def check_finite(loss):
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss; optimizer step aborted")

    def to(self, device):
        for name in ["mac", "target_mac", "mixer", "target_mixer", "critic", "target_critic", "role_mixer", "target_role_mixer", "mf_critic", "target_mf_critic", "attention"]:
            value = getattr(self, name, None)
            if value is not None:
                value.to(device)
        return self
