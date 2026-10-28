"""Diagnostic probes for source files lacking a confirmed main branch.

Temporary in-process registry additions are confined to this diagnostic. A pass
does not register a public algorithm or establish its full execution schedule.
"""
from dataclasses import asdict
from pathlib import Path
import json
import traceback
import uuid
from unified_marl import train
from unified_marl.algorithms.hmf.adapter import IMPLEMENTATIONS
from unified_marl.registry import HMF_ALGORITHMS


def main():
    root = Path(__file__).resolve().parents[1]
    candidates = {
        "hierarchical_qmix": ("hierarchical_qmix", "Hierarchical_Qmix"),
        "hierarchical_vdac": ("hierarchical_vdac", "Hierarchical_VDN"),
        "hierarchical_vdn_copy": ("hierarchical_vdn copy", "Hierarchical_VDN"),
        "hierarchical_vdn_meanq": ("hierarchical_vdn_02_formulate_meanQ", "Hierarchical_VDN_02_Formulate_MeanQ"),
        "hierarchical_vdn_dynamic_ipc": ("hierarchical_vdn_IPC", "Hierarchical_VDN"),
        "hierarchical_vdn_k_means": ("hierarchical_vdn_k_means", "Hierarchical_VDN_K_Means"),
        "hierarchical_vdn_roma": ("hierarchical_vdn_roma", "Hierarchical_VDN_ROMA"),
        "mfac": ("mfac", "MFAC"),
    }
    run_root = root / ".work/variant-probes" / uuid.uuid4().hex[:8]
    results = []
    for name, implementation in candidates.items():
        previous = IMPLEMENTATIONS.get(name)
        was_registered = name in HMF_ALGORITHMS
        IMPLEMENTATIONS[name] = implementation
        HMF_ALGORITHMS.add(name)
        try:
            result = train("wang_hmf/" + name, "toy", env_args={"n_agents": 4, "mask_actions": False},
                output_dir=run_root / name, overrides={"batch_size": 4, "buffer_size": 32, "t_max": 16,
                "max_updates": 1, "hmf_norm_rewards": False, "hidden_dim": 16, "n_heads": 2, "embed_dim": 8,
                "hmf_steps_per_update": 1, "hmf_num_batch": 1})
            results.append({"name": name, "status": "DEFAULT_UPDATE_PASS", "train": asdict(result),
                            "scope": "Candidate default update only; dynamic/high-level schedule not established"})
        except Exception as exc:
            results.append({"name": name, "status": "BLOCKED", "error": str(exc), "traceback": traceback.format_exc()})
        finally:
            if previous is None:
                IMPLEMENTATIONS.pop(name)
            else:
                IMPLEMENTATIONS[name] = previous
            if not was_registered:
                HMF_ALGORITHMS.discard(name)
        print(name, results[-1]["status"], results[-1].get("error", ""), flush=True)
    (root / "docs/evidence/stage3/remaining_variant_probes.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
