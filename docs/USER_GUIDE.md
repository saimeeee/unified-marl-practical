# unified-marl 内部使用说明书

适用版本：核心 0.7.0，内部分发修订 1。本包用于内部整理、训练与复用多智能体强化学习代码，可直接由 Python 和命令行调用，无须搭建网页、数据库或集群管理系统。

## 1. 下载哪个文件

从本仓库的 [Releases](https://github.com/saimeeee/unified-marl/releases) 下载对应系统的 ZIP，并同时下载 SHA256SUMS.txt。私有仓库中的下载需要仓库访问权限；也可以由已有访问权限的同事下载 ZIP 后内部转发，接收方使用 ZIP 不需要 GitHub。

| 文件 | 环境要求 | 安装时联网 |
|---|---|---|
| unified-marl-0.7.0-windows-x64-cpu-py312-offline.zip | Windows x64，CPython 3.12 x64，含 venv/pip | 无须联网，依赖已附齐 |
| unified-marl-0.7.0-linux-x86_64-cpu-py312.zip | Linux x86_64，glibc >= 2.28，CPython 3.12，含 venv/pip | 需从 PyPI 补齐 hf-xet>=1.1.3,<2.0.0 |

两版均为 CPU 版，Python 解释器和系统运行库不随包提供。不包含真实数据、checkpoint、预训练权重、StarCraft II、地图、ReplenishmentEnv 或 CityFlow。真实环境按需另行准备。

不要把 GitHub 的自动生成 Source code ZIP 当成上述安装包；自动源码归档没有数百 MB 的依赖 wheel。需要本地安装包时请选择表中完整文件名。

## 2. Windows 安装

先安装 CPython 3.12 x64 并包含 venv/pip。将 ZIP 解压到普通可写目录，在该目录打开 PowerShell，执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

若找不到 Python，指定实际解释器路径：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -PythonPath "C:\Python312\python.exe"
```

安装器校验文件，在解压目录创建 `.venv`，按固定版本和 SHA-256 从本地 wheelhouse 安装，并默认执行一次短 QMIX 安装检查。不会修改全局 Python 或持久更改 PowerShell 执行策略。若 PyTorch 报缺少系统 DLL，接收方需先准备相应的 Microsoft Visual C++ 运行库。

中文和空格路径可用，但路径不能过深。安装器会在 pip 安装前检查长路径；过长时换较短的可写目录（如 D:\marl）重新解压安装。脚本不会修改系统长路径注册表设置。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 --help
.\.venv\Scripts\python.exe -c "import unified_marl; print(unified_marl.__version__)"
```

后续示例中的 `python`，Windows 使用 `.\.venv\Scripts\python.exe`。安装检查报告位于解压目录 `results/install-check-*/report.json`。

## 3. Linux 安装

先准备 CPython 3.12、venv/pip 和目标系统所需运行库，再解压 Linux ZIP：

```bash
bash install.sh
bash run.sh --help
.venv/bin/python -c 'import unified_marl; print(unified_marl.__version__)'
```

脚本使用本地 wheelhouse 安装锁定依赖，再从 `https://pypi.org/simple` 补齐 `requirements-network.txt` 中的 hf-xet。该补充项未预先锁定具体版本或 wheel 哈希；本包不能称完整离线。脚本不使用 sudo、不安装系统包，也不自动启动训练。

按项目要求，这台 Windows 电脑只准备了 Linux 文件，未执行 Linux 安装或训练测试。接收方若需要自行检查，可运行：

```bash
.venv/bin/python -X utf8 -I smoke_test.py --output ./results/manual-check
```

后续示例中的 `python`，Linux 使用 `.venv/bin/python`。使用其他 Python 主次版本、ARM64、Alpine/musl 或 GPU 时，需要另行准备和验证相应依赖，不能直接套用本 ZIP。

## 4. 第一条在线训练与评估

先进入解压目录。下例使用 toy 测试环境，无须外部数据或仿真器：

```bash
python examples/online.py --output ./results/demo-online
python examples/inference.py ./results/demo-online/first/policy.pt
```

online.py 会训练、保存、加载评估，并从 training.pt 恢复训练。每次换新的输出目录，已有训练输出不会被覆盖。toy 仅验证接口与参数更新，不代表真实环境效果。

命令行方式（Windows 将 `bash run.sh` 替换为 `powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1`）：

```bash
bash run.sh train --algorithm pymarl/qmix --env toy --output-dir ./results/qmix --set batch_size=2 --set buffer_size=8 --set t_max=32
bash run.sh evaluate ./results/qmix/policy.pt --episodes 2 --seed 7
```

Python 调用：

```python
from unified_marl import train, evaluate, load_policy

result = train(
    algorithm="pymarl/qmix", env="toy", device="cpu", seed=7,
    output_dir="./results/my-task",
    overrides={"batch_size": 2, "buffer_size": 8, "t_max": 32},
)
print(result.metrics, result.checkpoint)
score = evaluate(result.checkpoint, episodes=2, seed=7)
policy = load_policy(result.checkpoint)
policy.reset()  # 每个新回合都调用；后续以 policy.act(obs, avail_actions) 选择动作。
```

自己的项目只要选用本包 `.venv` 中的解释器即可 import，不需要复制源码、设置原始交付目录或修改 sys.path。Windows 多进程训练或 ZMQ 的 Python 调用必须放在 `if __name__ == "__main__":` 保护下。

## 5. 离线、偏好与奖励模型

以下命令生成明确标记的 toy 合成演示数据，然后分别调用 BC、CPL 和 MR 奖励模型。它不是 SMAC 真实数据：

```bash
python examples/make_demo_data.py ./data/demo.th
python examples/offline.py --dataset ./data/demo.th --output ./results/bc
python examples/offline.py --algorithm kou/cpl --dataset ./data/demo.th --output ./results/cpl
python examples/reward.py --dataset ./data/demo.th --output ./results/reward
```

真实数据通过 `--dataset` 或 Python 的 dataset 参数指定；离线/偏好预算由 `training_updates` 推进，不依赖在线采样步数。偏好示例按前后半段配对，label=1 偏好后半段、0 偏好前半段、0.5 平局。不要复制团队奖励冒充个体奖励；需要独立个体字段的算法必须提供真实字段。详见 [示例说明](../examples/README.md) 和 [支持矩阵](SUPPORT_MATRIX.md)。

## 6. 配置、模型与恢复

配置优先级从低到高：公共默认 → 算法配置 → 环境配置 → 用户配置 → overrides/CLI --set → 显式 Python 调用参数。CLI `--set env_args.episode_limit=4` 支持嵌套键；值按 YAML 标量读取。具体算法和组件组合会校验，不能任意搭配。

每次训练通常生成 config.json、metrics.jsonl、result.json、policy.pt 和 training.pt：

- policy.pt 用于加载策略、推理和评估；循环网络每回合先 reset。
- training.pt 用于继续训练，包含该实现所需的优化器、目标网络、计数和回放等状态；不能用 policy.pt 代替。
- 奖励模型的推理状态和训练状态同样分开，使用 reward-training.pt 才能恢复相应训练状态。

```python
continued = train(
    resume=result.training_checkpoint,
    output_dir="./results/continued",
    overrides={"t_max": 64},
)
```

恢复支持的具体算法和边界见 [支持矩阵](SUPPORT_MATRIX.md)。外部仿真器内部状态不保存在 checkpoint 中；ZMQ worker 重启后的随机过程和异步调度不保证逐步相同。

## 7. SADMA 与 ZMQ

SADMA 本地 PPO 示例：

```bash
python examples/online.py --algorithm sadma/ppo --output ./results/ppo
```

ZMQ 示例：

```bash
bash run.sh train --algorithm sadma/qmix --env toy --execution zmq --config examples/zmq_managed.yaml --output-dir ./results/zmq-qmix
```

Windows 使用同参数的 run.ps1。ZMQ 仅接入支持矩阵中的 SADMA 配置：Q/VDN 支持 sync/async，普通 PPO 仅 sync，异步 PPO 必须显式 `use_vtrace=true`。不默认支持其他算法家族的分布式组合。已记录的是 Windows 本机回环验证；Linux、本次跨机器运行和加速效果未验证。详细角色、地址、超时和故障处理见 [ZMQ 文档](ZMQ_PROTOCOL.md)。

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| import 找不到 unified_marl | 使用解压目录中 .venv 的 Python，先确认安装成功 |
| Python 版本/架构不匹配 | 使用 CPython 3.12 x64；不要用 3.11、3.13、32 位或 ARM64 的解释器安装本包依赖 |
| Windows 长路径失败 | 换较短的可写目录重新解压；安装器不会更改系统设置 |
| 已有 .venv 没有安装标记 | 换新目录解压，避免覆盖已有环境 |
| 搬目录或换电脑后不能运行 | 从原 ZIP 重新解压安装，不转发或搬运已生成的 .venv |
| Linux 补充依赖下载失败 | 确认接收方可访问 PyPI；本版不能完全离线安装 |
| 训练输出已存在 | 换新 output_dir，保留已有模型 |
| SMAC/补货/CityFlow 找不到后端 | 按环境要求另装后端并提供真实任务数据、地图和配置 |
| 有算法注册名但不支持某组合 | 查支持矩阵和具体错误；不能靠更改名称绕过数据或环境约束 |

Windows 安装包已完成本机全新环境安装、训练更新、保存加载与恢复检查；Linux 按要求只准备文件。历史测试和真实数据验证不等于真实仿真环境、GPU、跨机或长期效果验证。全部未通过项保留在 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)，不要把 383 行迁移记录解释为 383 种算法。

## 9. 后续更新

新版本安装到新目录；保留自己的数据、配置和输出，按版本使用匹配的 checkpoint。发布记录、安装包和 SHA-256 放在 GitHub Release；源码、构建脚本、示例和文档放在仓库。来源说明和原许可文件随包保留。

## 0.8.0 工程应用入口

仓储补货、交通控制、电网连续控制从[工程应用说明](APPLICATIONS_GUIDE.md)进入，使用unified-marl-app及applications.warehouse/traffic/powergrid。新版本没有训练或场景实验，旧验证只适用于旧记录中的功能。Windows/Linux工程包见ENGINEERING_DISTRIBUTION.md。
