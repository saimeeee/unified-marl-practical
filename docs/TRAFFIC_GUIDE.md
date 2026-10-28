# CityFlow traffic control adapter

This guide describes the direct native <code>cityflow.Engine</code> adapter in <code>unified_marl.envs.traffic_engine</code>. It removes the dependency on the third-party <code>cityflowenv</code> wrapper for this new traffic path. The native simulator, its roadnet, flow, and any replay files remain user supplied.

The module imports the native <code>cityflow</code> package only when <code>CityFlowEngineEnv</code> is constructed. <code>read_traffic_spec</code> is simulator-free: it reads only the local CityFlow config and roadnet JSON, then returns fixed agent order, phase mappings, inbound lane metadata, and shape information for application validation or checkpoint metadata.

## Files and path resolution

Pass a native CityFlow JSON config through <code>cityflow_config_file</code>. The config's <code>dir</code> is resolved relative to the directory containing the config file when it is relative. <code>roadnetFile</code> and <code>flowFile</code> are then resolved relative to that <code>dir</code>; absolute asset paths are also accepted. Both assets must already exist locally.

Example native config:

~~~json
{
  "interval": 1.0,
  "seed": 7,
  "dir": "scenario",
  "roadnetFile": "roadnet.json",
  "flowFile": "flow.json",
  "rlTrafficLight": true,
  "laneChange": false,
  "saveReplay": false
}
~~~

If this JSON is saved as <code>maps/jinan/config.json</code>, the adapter looks for <code>maps/jinan/scenario/roadnet.json</code> and <code>maps/jinan/scenario/flow.json</code>. It builds a temporary normalized copy of the config so CityFlow receives stable absolute directory resolution; the supplied config and assets are not edited. The roadnet and flow must be on the same filesystem volume as the configured <code>dir</code>.

The config must enable <code>rlTrafficLight</code>. Traffic lights are controlled through CityFlow's <code>Engine.set_tl_phase(intersection_id, phase_id)</code>; the phase ID is an index in that intersection's <code>trafficLight.lightphases</code> array. See the official [CityFlow Engine API](https://github.com/cityflow-project/CityFlow/blob/master/docs/source/start.rst) and [roadnet format](https://github.com/cityflow-project/CityFlow/blob/master/docs/source/roadnet.rst).

## Map actual phases to agent actions

The adapter controls every non-virtual intersection in the roadnet. Its stable <code>agent_ids</code> order is the lexicographically sorted intersection ID list. Each intersection must have at least one inbound road lane and a traffic light phase table.

<code>phase_ids</code> is required and maps each <code>agent_id</code> to an ordered list of actual green phase indices from its own roadnet entry. A selectable phase must exist and list at least one <code>availableRoadLinks</code> entry. Policy action index 0 means the first item in that intersection's <code>phase_ids</code> list, action index 1 means the second, and so on. This lets intersections have different phase counts while sharing one padded discrete action dimension. Padded actions are masked unavailable.

For example, if the roadnet has two road links and phase 0 is an explicitly configured all-red clearance phase:

~~~json
{
  "trafficLight": {
    "lightphases": [
      {"time": 3, "availableRoadLinks": []},
      {"time": 30, "availableRoadLinks": [0]},
      {"time": 30, "availableRoadLinks": [1]}
    ]
  }
}
~~~

the mapping for this intersection can be:

~~~python
phase_ids = {
    "intersection_1_1": [1, 2]
}
yellow_phase_ids = {
    "intersection_1_1": 0
}
~~~

Provide an entry for every non-virtual intersection. <code>phase_ids</code> order is the action-index order and is retained in <code>env.phase_ids</code> and the static specification. The values in <code>env.phase_ids</code> are the actual roadnet phase indices, so callers can persist the map and translate actions for traffic operations.

CityFlow's native phase structure exposes <code>availableRoadLinks</code>, not lamp colors. For that reason this adapter treats an explicitly selected phase with an empty <code>availableRoadLinks</code> list as an all-red clearance phase. It never guesses an all-red phase. If <code>yellow_time</code> is positive, provide a valid <code>yellow_phase_ids</code> entry for every intersection; those IDs must be real, non-selectable phases with no available road links. This is a safe clearance interval, not a claim that CityFlow displays yellow lamps. Set <code>yellow_time=0</code> to switch directly between configured green phases.

## Static validation and environment use

<code>read_traffic_spec</code> validates and loads metadata without importing or starting the engine. It verifies that phase mappings exactly cover the non-virtual intersection set, that green and clearance phases are valid for each intersection, and that local roadnet and flow files exist.

~~~python
from unified_marl.envs.traffic_engine import CityFlowEngineEnv, read_traffic_spec

spec = read_traffic_spec(
    "maps/jinan/config.json",
    phase_ids=phase_ids,
    yellow_phase_ids=yellow_phase_ids,
    control_interval=10.0,
    yellow_time=3.0,
    episode_limit=360,
    seed=7,
    thread_num=1,
    queue_reward_scale=1.0,
)
# JSON-friendly checkpoint/application metadata:
agent_ids = spec["agent_ids"]
phase_map = spec["phase_ids"]
observation_size = spec["obs_size"]

env = CityFlowEngineEnv(
    cityflow_config_file="maps/jinan/config.json",
    phase_ids=phase_ids,
    yellow_phase_ids=yellow_phase_ids,
    control_interval=10.0,
    yellow_time=3.0,
    episode_limit=360,
    seed=7,
    thread_num=1,
    queue_reward_scale=1.0,
)
try:
    obs, state = env.reset()
    action_indices = [0] * env.n_agents
    business_actions = env.actions_to_phase_ids(action_indices)
    # [{"intersection_id": "...", "phase_id": ...}, ...]
    reward, done, info = env.step(action_indices)
finally:
    env.close()
~~~

Constructor arguments:

| Argument | Meaning |
|---|---|
| <code>cityflow_config_file</code> | Path to the existing native CityFlow JSON config. |
| <code>control_interval</code> | Green control duration in simulated seconds per decision. Converted to whole engine steps by rounding up with the config's <code>interval</code>. |
| <code>yellow_time</code> | Clearance duration in simulated seconds. Positive values require explicit <code>yellow_phase_ids</code>. It is also rounded up to whole engine steps. |
| <code>episode_limit</code> | Maximum number of agent decision cycles; the limit is reported as <code>info["episode_limit"]</code> so the runner can mark it truncated. |
| <code>phase_ids</code> | Required mapping from every non-virtual intersection ID to its ordered selectable green phase indices. |
| <code>yellow_phase_ids</code> | Optional mapping to an explicit all-red phase; required for every intersection when <code>yellow_time &gt; 0</code>. |
| <code>seed</code> | Base simulator seed. When the engine exposes <code>set_random_seed</code>, resets use this base plus the episode index. |
| <code>thread_num</code> | Native CityFlow engine thread count. |
| <code>queue_reward_scale</code> | Positive multiplier for the negative total waiting-vehicle reward. |

Each decision sets clearance phases for all intersections whose selected green changes, advances the engine for the clearance duration, applies the selected greens, then advances for <code>control_interval</code>. Therefore, a decision with a phase change advances simulated time by both intervals. <code>episode_limit</code> counts these decisions, not seconds.

## Observation, actions, reward, and statistics

Observations are a <code>float32</code> matrix with shape <code>(n_agents, obs_size)</code>. Per agent they contain:

1. A one-hot current selectable phase in the action-index order from <code>phase_ids</code>.
2. For each inbound lane, waiting vehicle count followed by total vehicle count.
3. Zero padding for phase slots and lane slots unused by that intersection.

Inbound lane order is sorted by road ID and lane index. Lane IDs use the CityFlow convention <code>&lt;road_id&gt;_&lt;lane_index&gt;</code>. Lane length is read from an explicit lane <code>length</code> when present; otherwise it is computed from the road geometry points. <code>inbound_lane_ids</code> and <code>inbound_lane_lengths</code> are exposed per intersection for downstream feature processing.

<code>get_avail_actions()</code> returns one row per <code>agent_ids</code> entry. Only indices represented in that intersection's <code>phase_ids</code> are marked available. <code>get_total_actions()</code> is the maximum selectable green count across intersections.

The scalar team reward after each decision is <code>-queue_reward_scale * total_waiting_vehicles</code> over controlled inbound lanes. <code>get_stats()</code> reports CityFlow's average travel time plus the average and current queue counts. The average queue count is measured over internal simulator steps during the current episode.

The <code>step()</code> result follows the package contract: <code>(reward, done, info)</code>. Reaching <code>episode_limit</code> sets <code>done=True</code> and <code>info["episode_limit"]=True</code>. The adapter has no custom renderer or replay exporter; configure native CityFlow replay output in the supplied config if needed.

## Resource boundary and validation status

<code>close()</code> calls an engine <code>close</code> method if that installed CityFlow build exposes one, releases the Python engine reference, and removes the temporary normalized config. Builds without a native <code>close</code> method do not provide evidence that all simulator resources were explicitly released.

The integration step performs static source checks only. It does not install CityFlow, load a native engine, run a traffic simulation, train a policy, download map assets, or establish simulator resource-release behavior.

## 已封装的业务调用

```python
from unified_marl.applications import traffic
result = traffic.train("my-traffic/traffic.yaml", output_dir="results/traffic-001")
policy = traffic.load_policy(result.checkpoint)
policy.reset()
# 当前路口状态按真实ID构造：
# decision = policy.act(state)
```

state 可用 traffic_state.json 的命名结构：intersections 列表每项提供 intersection_id、当前绿色phase_id、每条进口道的waiting_vehicles与vehicles字典。字典必须包含该路口全部进口道，车道顺序由路网固定；示例只有格式用的单车道，必须按真实路网替换。等待数不得大于总车辆数，必须在绿色决策边界采样，过渡全红期间不要发起这类决策。返回decisions为intersection_id、action_index、实际phase_id。封装会自动填补不同路口相位数量的动作掩码，也会将调用方掩码与实际可选相位取交集。

外部执行端必须遵守相同控制周期与安全过渡规则；返回目标绿灯相位不代表硬件控制器已执行过渡。直接观测数组的布局以本指南前述顺序为准。
