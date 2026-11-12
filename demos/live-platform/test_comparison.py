import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from comparison import compare_runs, scene_signature, GreedyLandmarkPolicy


class ComparisonChecks(unittest.TestCase):
    def test_same_scene_compares_models_and_rule_but_rejects_other_seed_or_scene(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = ['a' * 32, 'b' * 32, 'c' * 32]
            for i, sid in enumerate(ids):
                folder = root / sid
                folder.mkdir()
                row = dict(id=sid, status='completed', seed=101,
                           model=dict(id=str(i), algorithm=['qmix', 'vdn', 'rule/greedy'][i]),
                           frame=dict(done=True, scene='mpe', step=25, metrics=[],
                                      comparison_key=scene_signature('mpe', {'n_agents': 3})))
                (folder / 'result.json').write_text(json.dumps(row))
            self.assertEqual(len(compare_runs(root, ids)['runs']), 3)
            row['seed'] = 102
            (folder / 'result.json').write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, '随机种子'): compare_runs(root, ids)
            row['seed'] = 101
            row['frame']['comparison_key'] = scene_signature('mpe', {'n_agents': 4})
            (folder / 'result.json').write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, '场景配置'): compare_runs(root, ids)
            row['status'] = 'stopped'
            (folder / 'result.json').write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, '完整'): compare_runs(root, ids)

    def test_rejects_duplicate_ids_and_paths(self):
        for ids in (['a' * 32] * 2, ['../outside', 'b' * 32], [], 'ids'):
            with self.assertRaises(ValueError): compare_runs('.', ids)

    def test_malformed_json_structures_get_a_clear_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / ('a' * 32)
            folder.mkdir()
            for value in ([], {'frame': []}, {'frame': {'metrics': [None]}, 'model': {}}):
                (folder / 'result.json').write_text(json.dumps(value))
                with self.assertRaises(ValueError): compare_runs(root, ['a' * 32, 'b' * 32])

    def test_rule_obeys_mask_and_produces_deterministic_legal_rollout(self):
        import numpy as np
        from unified_marl.envs.mpe import MPEEnv
        traces = []
        for _ in range(2):
            env = MPEEnv(seed=42, episode_limit=5)
            try:
                env.reset()
                policy = GreedyLandmarkPolicy(env)
                mask = np.zeros((env.n_agents, 5), dtype=int)
                mask[:, 0] = 1
                self.assertEqual(policy.act(env.get_obs(), mask), [0] * env.n_agents)
                trace = []
                for step in range(5):
                    action = policy.act(env.get_obs(), env.get_avail_actions())
                    reward, done, _ = env.step(action)
                    trace.append((action, reward, done))
                traces.append(trace)
            finally:
                env.close()
        self.assertEqual(traces[0], traces[1])


if __name__ == '__main__':
    unittest.main()
