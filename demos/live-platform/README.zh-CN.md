# MARL Lab 本地实时仿真平台

从 [GitHub 仓库](https://github.com/saimeeee/unified-marl-practical) 下载完整代码，在自己的电脑启动，然后通过 `http://127.0.0.1:18765/` 使用。网页、模型推理、四个仿真环境和运行记录都在本机。界面采用暖白、米白与石墨色，橙色表示主要操作，绿色表示连接和正常状态；宽屏显示侧栏与双列工作区，窄屏采用上下布局。

## Windows 快速启动

1. 安装并启动 Docker Desktop，启用 WSL 2，使用 **Linux 容器**模式。需要 Docker Compose v2，Docker Desktop 已包含它。平台使用 CPU，不需要显卡，也不需要在 Windows 全局安装 Python。
2. 打开 GitHub 仓库，选择 **Code → Download ZIP**，然后完整解压。也可以使用 `git clone https://github.com/saimeeee/unified-marl-practical.git`。请保留整个仓库目录，不要只复制 `demos/live-platform` 子目录。
3. 双击 `demos/live-platform/打开实时平台.cmd`。
4. 首次启动会联网构建本地镜像，安装 CPU 依赖并编译随包附带的 CityFlow，需要等待数分钟；耗时取决于网络和电脑性能。构建完成后会自动打开浏览器。
5. 在网页选择场景和模型，点击“初始化”。环境暂停在第 0 步，点击“开始运行”连续推进，点击“单步”执行下一步。

完整下载包含 `bundle/`：四个模型权重、配套场景数据、CityFlow 源码和文件校验清单，无需另外寻找模型或数据。首次镜像构建成功后，再次启动会复用本机镜像；仿真与模型推理可以离线使用。首次构建、重新构建时下载缺失依赖仍需要网络。

关闭网页不会停止平台。用 `停止实时平台.cmd` 停止本地容器；模型和 `local-state/` 中的运行记录都会保留。默认只开放本机地址，不向局域网或公网开放。当前是单用户、单运行会话。

## 启动参数与代码更新

以下 PowerShell 命令在 `demos/live-platform` 目录中运行：

```powershell
# 启动并打开浏览器
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Open-LivePlatform.ps1

# 下载新版本或 git pull 后，重新构建并启动
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Open-LivePlatform.ps1 -Rebuild

# 默认端口被其他程序占用时
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Open-LivePlatform.ps1 -Port 18766

# 启动，但不自动打开浏览器
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Open-LivePlatform.ps1 -NoBrowser

# 停止平台，保留本地数据
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Stop-LivePlatform.ps1
```

`-ExecutionPolicy Bypass` 仅作用于这次 PowerShell 进程，不修改系统执行策略。使用 ZIP 更新时，将原目录的 `local-state/` 保留到新版本对应位置，再使用 `-Rebuild`。重新构建或停止容器会终止当前仿真，请先导出需要的轨迹。

Linux 或 macOS 安装 Docker 与 Compose v2 后，可在同一目录运行：

```bash
bash start-local.sh

# 更新代码后重新构建
bash start-local.sh --rebuild

# 停止，保留 local-state
docker compose down
```

## 四个场景

| 场景 | 随包模型 | 一次单步 | 显示内容 |
|---|---|---|---|
| 库存补货 | wang_hmf/mfvfd，1000 更新 | 1 个补货周期，共 24 周期 | 库存、在途、下单、销量、利润和满足率 |
| MPE 协作 | pymarl/qmix，1000 更新 | 1 个粒子物理步，共 50 步 | 实际位置、速度、地标距离、覆盖和碰撞 |
| 交通信号 | sadma/ppo，300 更新 | 通常推进物理时间 1 秒，共 32 次信号决策；末尾另有完成状态帧 | CityFlow 原生车辆坐标、相位、队列和行程时间 |
| 电网储能 | bounded_ppo，100 更新 | 1 个 15 分钟控制步，共 96 步 | 33 节点电压、线路功率、网损和储能动作 |

随包权重来自 2026-10-28 的训练产物。初始化后，后端实时调用模型与仿真器，前端显示刚生成的状态，不读取 MP4 或预录轨迹。实际推进速度受本机 CPU 性能限制；倍速只调整帧间等待时间，不改变环境物理步长、模型动作或奖励计算。

暂停在当前物理或决策步完成后生效，不撤销已开始的计算。交通一个控制动作跨多个物理秒，因此“已完成决策累计奖励”在决策结束后更新；平均行程时间使用 CityFlow 原生指标，包含其定义下的在途车辆。电网控制时长不含初始化预热，电压越限按展示阈值 `[0.95, 1.05] p.u.` 标识。

当前模型效果仍有限：库存弱于规则基线，MPE 严格覆盖率低，电网仍有越限且综合奖励未改善。平台可用于查看实际策略行为与仿真过程，演示视频及平台可运行本身不构成策略效果验收。

## 添加同环境的新模型

`worker.py` 定义四个环境的状态获取、模型加载和逐步仿真适配器，`runtime.py` 提供运行所需的公共逻辑；`server.py` 管理会话、控制、导出与工作进程；`static/` 消费状态 JSON 并展示画面。模型清单与环境配置独立于前端。

默认模型来自 `bundle/catalog.json`。启动时，`prepare_local.py` 校验 bundle 文件并生成本机可用的配置与 `local-state/models.json`。**不要直接修改生成的 `models.json`**，下次启动会重新生成。

新增已训练、兼容当前推理协议的模型时，把权重及配套文件放进 `local-state/`，然后创建 `local-state/extra-models.json`。例如：

```text
local-state/
├── extra-models.json
└── custom/
    └── mpe-vdn/
        └── policy.pt
```

```json
{
  "models": [
    {
      "id": "mpe-vdn-custom",
      "scenario": "mpe",
      "label": "MPE 协作",
      "algorithm": "pymarl/vdn",
      "training": "自定义模型训练说明",
      "detail": "3 个智能体 · 3 个地标",
      "note": "填写该模型的评估范围和限制",
      "checkpoint": "custom/mpe-vdn/policy.pt",
      "config": null
    }
  ]
}
```

`id` 必须唯一；`checkpoint` 与非空 `config` 都相对于 `local-state/`，并且必须指向该目录内实际存在的文件。停止后重新启动平台，新模型会进入网页列表；仅修改这些本地模型文件无需重新构建镜像。

模型必须支持已有 `load_policy` 或应用层推理协议，并与场景配置、观测及动作空间兼容；登记算法名称不能改变权重中的模型结构。MPE 当前视图按 `simple_spread` 的三智能体、三地标设计。库存、交通、电网还需提供匹配的应用 YAML、`application.json` 元数据及模型目录配套文件；应用配置中引用的数据也需一并保留，路径以配置文件所在目录为基准。添加全新环境需要实现适配器与对应画布，不能只改模型清单。

平台负责加载训练好的模型并实时仿真。网页目前不发起训练，也没有训练任务队列或多用户调度；新算法训练继续使用仓库的训练入口与配置，训练完成后再登记模型。

## 本地文件与复现

```text
demos/live-platform/
├── bundle/                 # 随包模型、配置、场景数据与 CityFlow 源码
├── static/                 # 网页界面
├── compose.yaml            # 本地容器与端口配置
├── Dockerfile              # CPU 运行环境
└── local-state/            # 启动后生成，保留在本机
    ├── configs/            # 启动时生成的应用配置
    ├── models.json         # 启动时生成的完整模型清单
    ├── extra-models.json   # 可选，用户登记的额外模型
    └── runs/               # 运行记录
```

`local-state/runs/<运行 ID>/` 保存请求、模型 SHA-256、逐步 `frames.jsonl`、工作进程日志和最终状态。网页可在运行中导出一致的 JSON 轨迹，历史列表显示最近 20 次已结束运行。平台不自动删除旧运行；停止容器和重新构建镜像也不删除这个目录。

每次初始化启动独立 Python 工作进程；结束、失败和停止都会回收进程。无浏览器访问且暂停 30 分钟后自动停止运行。平台校验 Host 和同源写请求，只允许选择模型清单中的条目，不接受网页传入任意文件路径或命令。

## 常见问题与验证

- 提示找不到 Docker 或 Linux 引擎未启动：打开 Docker Desktop，等待其启动完成，并确认使用 Linux 容器模式。
- 提示端口占用：先停止旧平台；若端口由其他程序使用，用 `-Port 18766` 启动并访问对应地址。
- 提示 bundle 缺失或校验失败：重新下载完整仓库并完整解压，不要只下载网页目录或单个脚本。
- 首次构建下载失败：检查 Docker Desktop 的网络配置，修复后重新运行启动脚本。已下载的构建层会尽量复用。
- 更新代码后界面仍旧：使用 `-Rebuild` 重新构建镜像，然后刷新浏览器。

在 `demos/live-platform` 目录查看日志：

```bash
docker compose logs --tail 100 platform
```

运行实际环境验收脚本：

```bash
docker compose exec -T platform python demos/live-platform/verify_live.py --output /var/lib/unified-marl/verification
```

该脚本会顺序运行四个真实模型完整回合，检查初始暂停、单步、暂停不继续推进、运行中导出、初始化时停止，以及无效模型、种子和跨域写请求的拒绝行为。仅在没有用户运行时执行。报告保存在本机 `local-state/verification/`，实际验收结果以其中的 `verification.json` 为准。
