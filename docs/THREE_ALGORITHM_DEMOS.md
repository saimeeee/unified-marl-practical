# 三算法本机调用与视频交付（2026-10-24）

按用户最新授权，在 Windows 本机 Docker Desktop 的 Linux CPU 容器中实际运行 MFVFD 仓储、SADMA-PPO 交通及 bounded_ppo 电网。完整演示源码在 `demos/three-algorithm-videos`，完整可打开项目交付到 `C:\Users\71919\Desktop\三算法仿真演示`，含原始轨迹、模型、MP4、参数模板、VS Code 工作区、中文说明及校验。当前没有上传 GitHub，不改原始三份交付或全局 Python。

## 实际通过

| 能力 | 本机结果 | 边界 |
|---|---|---|
| 仓储 MFVFD / 固定分组HMF | 120更新、2904训练步、240次真实非零优化器变化；保存/加载；24期完整仿真与24秒MP4 | 2仓库×2SKU合成需求的InventoryEnv，不等同原ReplenishmentEnv及200/1000agent实验，无策略效果结论 |
| 交通 SADMA-PPO＋可选服务约束 | 80更新、20480训练步、320次非零变化；重新加载；352秒原生车辆/灯记录与35.17秒MP4 | 新生成2×2路网、16条流各15秒发车、绿20/服务阈值40秒；不是原实验资产；约束是公开工程规则 |
| 电网 bounded_ppo | 12更新、192训练步、24次非零变化；保存/加载；96控制步+1独立预热；32秒视频 | 公开CityLearn v1替代数据、case33bw 33bus/32建筑；是工程有界适配，无原Normal数学完全一致或效果结论 |

三个算法训练损失/梯度/参数均有限；离散动作满足可用掩码，电网动作满足连续边界及SOC。模型独立加载初始动作一致，三份额外推理脚本进程均exit0。全视频帧解码通过，1280×720/H.264/12fps。原生位置、相位、时间、库存守恒、电网潮流与数据维度单列核对，不把仿真通过解释为物理现场验收。

首次all命令的电网CSV导出KeyError导致exit1，训练已完成；修复导出后从同一个模型重新录制成功，原failure与summary保留。最终交通来自补训后的第三个run，首版堵车视频保留，不改写为优化成功。原始交付没有发现可用于这三场景的预训练权重，SMAC数据集.th不当作模型。

## 交通问题与处理

用户指出首版40更新视频堵车后，检查发现两个路口32次决策不换相位，另两个主要只选一个相位。高车流相同seed1/300秒下，旧SADMA-PPO平均排队292.590辆，固定周期235.227辆，压力规则251.873辆。规则对照同样拥堵；未把压力规则宣称有效。

按用户允许减车流的授权，发车间隔从5秒改为15秒。恢复原SADMA-PPO主要设置：hidden128、batch/buffer8、update_rate8个完成回合、mini_epochs4、lr0.0005、entropy0、GAE0.95、clip0.2、max_steps10000000。实际补训80次，未跑完整原1000万步。

核心CityFlow环境新增默认关闭的 `max_green_time`/`max_red_time`。启用后，根据真实roadLink/laneLink入车道服务、排队及原生时间生成动态动作掩码；训练和推理使用同一规则，step前拒绝非法动作。保持显式全红过渡，reset清零。max_red是决策边界服务触发阈值，不能保证绝对等待上限；只有一个动作时来自约束，不宣称是策略学会。未改变SADMA-PPO损失或更新数学。

新场景下相同路网/车流/约束、seed1、严格300秒，各方法均观测320个原生车辆池ID：

| 方法 | 平均路口队列 | 末时队列 | 消失车辆池ID |
|---|---:|---:|---:|
| SADMA-PPO＋服务约束 | 10.913 | 8 | 236 |
| 固定周期＋相同约束 | 22.787 | 30 | 212 |
| 压力规则＋相同约束 | 19.313 | 24 | 220 |

仅在该单种子短时场景，组合平均队列比同条件固定周期低约52.1%。不能把高低流量前后差异全部归因于训练；没有多种子统计或泛化结论。车辆池ID消失来自固定CityFlow源码的采样计数，非额外官方finished接口；原生平均时间包含未完成车辆。实际新视频四路口均使用全部4绿相位，最大总排队21辆，平均约11.09辆。

## 命令与证据

桌面项目内实际运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 all
# 首次all因电网CSV导出返回1，其他两项成功；原记录保留。
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 powergrid -Checkpoint outputs\20261023-163204-064456\powergrid\model\policy.pt
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 traffic -TrafficUpdates 80
```

后两条exit0；运行路径来自容器UTC，文件夹20261023对应本地东京2026-10-24。完整Docker构建、benchmark、推理命令、退出码和限制见演示项目 `logs/validation-record.md`。PowerShell5.1实际执行启动脚本，局部处理正常stderr并按退出码判断，路径映射静态检查通过。

主整合项目实际单元命令：

```powershell
.venv/Scripts/python.exe -m pytest tests/test_traffic_service_limits.py tests/test_engineering_hotfix.py tests/test_hotfix_publication.py -q
```

34 passed，其中18项新约束检查；新/旧/变更限制metadata错配拒绝通过，默认关闭保持旧行为。378个运行源/资源与vendor副本逐字节一致。公开CityLearn10份数据原始SHA未变；运行副本仅规范两处JSON非法0.为0.0，保留转换记录。

验证栈：Windows Docker Desktop4.96.3、Engine29.9.2、Linux6.18.54.1-WSL2/amd64、Python3.12.15、torch2.6.0+cpu、NumPy1.26.4、Gym0.26.2、pandas2.2.3、SciPy1.13.1、pandapower发行版2.14.11、CityFlow0.1、Pillow11.3.0、imageio-ffmpeg0.6.0。当前本地服务扩展镜像ID `sha256:ee98d85031b5a0ec5c40593405ab7543e7c5f0db0fae28f09176992efbe65d4b`；正式GitHub0.8.2 Release未被更改。

机器可读摘要和回执保存在 `docs/evidence/three-algorithm-demos/`；完整视频、模型和原生轨迹在演示项目/桌面。源码、默认模板、单场景/批量命令、重新推理和重新编码都可调用。尚未验证原论文规模/全部环境、长期收敛、多种子、物理现场、GPU或跨机；仓储、电网效果未通过验收。
