"""Compare complete runs only when their scene contract and seed agree."""
import hashlib
import json
from pathlib import Path
import re


def scene_signature(scene, definition):
    data = json.dumps(dict(version=1, scene=scene, definition=definition),
                      sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def traffic_signature(config):
    args = dict(config['env_args'])
    source = Path(args.pop('cityflow_config_file'))
    args.pop('seed', None)
    native = json.loads(source.read_text(encoding='utf-8'))
    root = (source.parent / native.get('dir', '.')).resolve()
    assets = {key: json.loads((root / native[key]).read_text(encoding='utf-8'))
              for key in ('roadnetFile', 'flowFile')}
    native = {k: v for k, v in native.items() if k not in {
        'dir', 'roadnetFile', 'flowFile', 'roadnetLogFile', 'replayLogFile', 'saveReplay', 'seed'}}
    return scene_signature('traffic', dict(env_args=args, native=native, assets=assets))


def compare_runs(root, ids):
    if (not isinstance(ids, list) or not 2 <= len(ids) <= 6 or
            any(not isinstance(sid, str) or not re.fullmatch(r'[0-9a-f]{32}', sid) for sid in ids) or
            len(set(ids)) != len(ids)):
        raise ValueError('请选择 2–6 条不同的完整运行记录。')
    rows, identity = [], None
    for sid in ids:
        try:
            result = json.loads((Path(root) / sid / 'result.json').read_text(encoding='utf-8'))
            if not isinstance(result, dict):
                raise ValueError('运行记录格式错误。')
            frame = result.get('frame') or {}
            if not isinstance(frame, dict) or not isinstance(result.get('model'), dict):
                raise ValueError('运行记录格式错误。')
            if (not isinstance(frame.get('metrics'), list) or
                    any(not isinstance(m, dict) or not isinstance(m.get('label'), str) or
                        type(m.get('value')) not in (int, float) for m in frame.get('metrics', []))):
                raise ValueError('运行指标格式错误。')
            key = frame.get('comparison_key')
            if result.get('id') != sid or result.get('status') != 'completed' or not frame.get('done'):
                raise ValueError('只能比较已完整结束的回合。')
            if not isinstance(key, str) or not re.fullmatch(r'[0-9a-f]{64}', key):
                raise ValueError('记录缺少场景校验信息，请使用新版平台重新运行。')
            current = (frame['scene'], key, result['seed'])
            if identity is not None and identity != current:
                raise ValueError('场景配置或随机种子不同，不能直接比较；请复用相同配置重新运行。')
            identity = current
            rows.append(dict(id=sid, model=result['model'], seed=result['seed'],
                             controller=result.get('effective_controller', 'trained'),
                             metrics=frame['metrics'], physical_steps=frame['step'],
                             checkpoint_sha256=result.get('checkpoint_sha256')))
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError('运行记录缺失或损坏，无法比较。') from exc
    json.dumps(rows, allow_nan=False)
    expected_metrics = [(m['label'], m.get('unit', '')) for m in rows[0]['metrics']]
    if any([(m['label'], m.get('unit', '')) for m in row['metrics']] != expected_metrics for row in rows):
        raise ValueError('运行指标定义不同，不能放入同一对比表。')
    return dict(scene=identity[0], comparison_key=identity[1], seed=identity[2], runs=rows,
                note='同场景同种子的单回合比较；不代表多种子效果评估或策略收敛。交通决策次数相同时，切换相位仍可能带来不同物理时长。')


class GreedyLandmarkPolicy:
    """Centralized, deterministic reference with access to all entity positions."""
    def __init__(self, env):
        self.env = env

    def reset(self):
        pass

    def act(self, observation, avail_actions):
        import numpy as np
        world = self.env.world
        available = set(range(len(world.landmarks)))
        actions = []
        for index, agent in enumerate(world.agents):
            candidates = available or set(range(len(world.landmarks)))
            target = min(candidates, key=lambda i: (float(np.linalg.norm(
                world.landmarks[i].state.p_pos - agent.state.p_pos)), i))
            available.discard(target)
            delta = world.landmarks[target].state.p_pos - agent.state.p_pos
            desired = delta - .35 * agent.state.p_vel
            axis = int(np.argmax(np.abs(desired)))
            action = (1 if desired[0] > 0 else 2) if axis == 0 else (3 if desired[1] > 0 else 4)
            if np.linalg.norm(delta) < .05 and np.linalg.norm(agent.state.p_vel) < .1:
                action = 0
            valid = np.flatnonzero(avail_actions[index])
            if not len(valid):
                raise ValueError('No legal MPE action available')
            actions.append(action if action in valid else int(valid[0]))
        return actions
