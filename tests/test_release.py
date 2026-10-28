"""Release-facing contracts and reproducible training continuation."""
import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch
from test_offline import raw_data, cpu
from test_rewards import model_options
from unified_marl import train, evaluate
from unified_marl.config import dotted_overrides, load_config
from unified_marl.datasets import EpisodeDataset
from unified_marl.rewards.service import RewardModel, load_reward_model


@pytest.mark.parametrize("overrides", [{"seed": -1}, {"seed": True}, {"seed": 2**32}, {"num_threads": 0}])
def test_all_configuration_families_validate_common_settings(overrides):
    for algorithm in ["pymarl/qmix", "sadma/ppo", "kou/bc", "wang/rode", "wang_hmf/mfq"]:
        with pytest.raises(ValueError):
            load_config(algorithm, overrides=overrides)


@pytest.mark.parametrize("values", [{"": 1}, {"a..b": 2}, {"a.b": 1, "a": 2}, {"a": 1, "a.b": 2}])
def test_ambiguous_config_override_is_rejected(values):
    with pytest.raises(ValueError):
        dotted_overrides(values)


def test_evaluation_is_seeded_and_does_not_change_training_rng(tmp_path):
    result = train(overrides={"batch_size": 2, "buffer_size": 4, "t_max": 16, "max_updates": 1}, output_dir=tmp_path/"train")
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    before = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    first = evaluate(result.checkpoint, episodes=3, seed=17)
    torch.testing.assert_close(torch.get_rng_state(), before[0], rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
    assert random.getstate() == before[2]
    assert evaluate(result.checkpoint, episodes=3, seed=17) == first


@pytest.mark.parametrize("name", ["MR", "NMR", "PrefTransformer", "MultiPrefTransformerDivide", "old_global", "old_reward_global"])
def test_reward_training_checkpoint_restores_next_update(name, raw_data, tmp_path):
    pair = EpisodeDataset(raw_data).preference([2, 3], trim=False)
    torch.manual_seed(66)
    model = RewardModel(name, EpisodeDataset(raw_data).env_info, model_config=model_options(name))
    model.train_step(pair)
    checkpoint = model.save(tmp_path/"reward-training.pt", training=True)
    expected = model.train_step(pair)
    expected_state = {key: value.clone() for key, value in model.network.state_dict().items()}
    loaded = load_reward_model(checkpoint, for_training=True)
    actual = loaded.train_step(pair)
    assert loaded.steps == model.steps == 2
    assert actual == pytest.approx(expected, abs=1e-7)
    for key, value in loaded.network.state_dict().items():
        torch.testing.assert_close(value, expected_state[key], atol=0, rtol=0)
    inference = model.save(tmp_path/"reward.pt")
    with pytest.raises(ValueError, match="reward-training"):
        load_reward_model(inference, for_training=True)


def test_offline_resume_rejects_ignored_config_and_algorithm(raw_data, tmp_path):
    data = EpisodeDataset(raw_data)
    result = train("kou/bc", dataset=data, overrides={"training_updates": 1, "off_batch_size": 2}, output_dir=tmp_path/"first")
    evaluated = evaluate(result.checkpoint, env="toy", env_args={"n_agents": 3, "episode_limit": 5, "mask_actions": False}, episodes=2)
    assert evaluated.episodes == 2  # Replaces SC2 args instead of leaking state_last_action/map_name.
    with pytest.raises(ValueError, match="replacement config"):
        train(resume=result.training_checkpoint, dataset=data, config={"seed": 8})
    with pytest.raises(ValueError, match="change algorithm"):
        train("kou/cpl", resume=result.training_checkpoint, dataset=data)


@pytest.mark.parametrize("algorithm,agent", [("kou/bc","rnn_ns"),("kou_old/qmix","mlp"),("kou_old/qmix","mlp_ns")])
def test_additional_source_agent_variants(algorithm, agent, raw_data, tmp_path, monkeypatch):
    import test_offline as reference
    from unified_marl import load_policy
    evidence=Path(__file__).resolve().parents[1]/"docs/evidence/stage7"/("agent-"+agent)
    evidence.mkdir(parents=True,exist_ok=True)
    original_config=reference.config
    def variant(name):
        cfg=original_config(name)
        cfg.update(agent=agent,hidden_dim=16)
        return cfg
    monkeypatch.setattr(reference,"config",variant)
    monkeypatch.setattr(reference,"EVIDENCE",evidence)
    reference.test_source_policy_loss_and_parameter_parity(algorithm,raw_data,monkeypatch)
    data=EpisodeDataset(raw_data)
    result=train(algorithm,dataset=data,output_dir=tmp_path/"first",
                 overrides={"agent":agent,"hidden_dim":16,"rnn_hidden_dim":16,"training_updates":2,"off_batch_size":4,"batch_size":4})
    assert result.parameter_delta_l2>0
    policy=load_policy(result.checkpoint)
    batch=data.batch([0])
    action=policy.act(batch['obs'][0,0],batch['avail_actions'][0,0])
    assert batch['avail_actions'][0,0].numpy()[np.arange(3),action].all()
    continued=train(resume=result.training_checkpoint,dataset=data,output_dir=tmp_path/"continued",overrides={"training_updates":3})
    assert continued.updates==3 and continued.parameter_delta_l2>0


def test_zmq_periodic_independent_evaluation(tmp_path):
    result = train("sadma/qmix", "toy", execution="zmq", output_dir=tmp_path/"distributed",
        overrides={"t_max": 64, "max_updates": 2, "train_batch_size": 2, "train_buffer_size": 8,
                   "hidden_dim": 16, "evaluation_interval_updates": 1, "evaluation_episodes": 2})
    report = json.loads((Path(result.output_dir)/"execution.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert not report["evaluation"]["forced"]
    assert report["evaluation"]["records"]
    for row in report["evaluation"]["records"]:
        assert row["status"] == "PASS" and row["evaluation"]["episodes"] == 2
        assert row["policy_version"] == row["update"]
    evidence = Path(__file__).resolve().parents[1]/"docs/evidence/stage7"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence/"periodic-evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


@pytest.mark.parametrize("failure", ["error", "timeout"])
def test_periodic_evaluator_failure_is_bounded_and_cleaned(failure, tmp_path):
    import multiprocessing
    from unified_marl.execution.evaluation import PeriodicEvaluator
    before={p.pid for p in multiprocessing.active_children()}
    evaluator=PeriodicEvaluator({"seed":1,"evaluation_interval_updates":1,
                                "evaluation_timeout_s":0.000001 if failure=="timeout" else 15},tmp_path)
    evaluator.submit(lambda:str(tmp_path/"missing.pt"),1,1)
    try:
        with pytest.raises((RuntimeError,TimeoutError),match="Evaluation"):
            evaluator.finish()
    finally:
        result=evaluator.close()
    assert {p.pid for p in multiprocessing.active_children()}==before
    if failure=="error": assert result['records'][0]['status']=='FAILED'
    else: assert result['forced']
