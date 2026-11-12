# 自定义交通场景：使用与开发交接

2026-11-12：将此前保留在本地的路口、车辆数量与决策次数配置整理到当前 `main` 源码，供继续开发。旧 `v0.9.0-local.1` Release 附件保持原版；获取本功能请下载仓库当前源码 ZIP 或 `git pull`。

后续验收已完成：[GitHub Actions 43200090637](https://github.com/saimeeee/unified-marl-practical/actions/runs/43200090637) 的 portable 与 native-traffic 两项任务均成功。7 种原生交通情况及 1×2 路网训练、登记、实时推理和规则对比通过；当前还支持训练包下载和模型登记，见 [训练流程](TRAINING_WORKFLOW.zh-CN.md)。以下旧首轮验证记录保留时间语境。

## 使用范围

在网页选择“交通信号”，启用“自定义场景”，填写参数后初始化；已有预设为单路口轻车流、2×2 训练策略、3×3 压力规则。运行中参数锁定，结束后可修改；历史列表可以复用配置。关闭自定义开关会使用原场景配置。

| 参数 | 范围与默认值 | 含义 |
|---|---|---|
| `rows` / `cols` | 各 1–4，默认 2 | 有信号控制的路口数为行数乘列数，支持矩形网格 |
| `vehicle_count` | 0–3000，默认 400 | 整个回合计划发车总量，不是同时在网数量 |
| `decision_limit` | 8–180，默认 32 | 信号控制决策次数；切换相位可能增加仿真物理时间 |
| `controller` | `trained` / `pressure`，默认 `trained` | 已训练策略或压力规则控制器 |

计划车辆按直行路线分批生成；拥堵可能导致车辆尚未进入路网或在回合结束前尚未完成行程。实时在网车辆数与计划总量不同。

现有 `sadma/ppo` 权重仅兼容原始 2×2 路网。2×2 自定义场景复制原始 roadnet 字节，保留模型结构及路网摘要约束；可修改车流量及回合长度，但不代表策略已适应新的需求。其他网格必须选择压力规则，此时展示算法为 `rule/pressure`、帧中 `policy_source` 为 `pressure_rule`、`checkpoint_sha256` 为 `null`，不把规则运行当作模型推理。库存、MPE、电网目前继续使用各自既有场景配置。

## 接口与文件入口

`GET /api/scenarios` 返回参数范围；`GET /api/health` 返回目录资产可用性与当前运行摘要，不能代替原生仿真验收。`POST /api/start` 示例（`model` 使用 `/api/catalog` 返回的交通模型 ID）：

```json
{
  "model": "<交通模型 ID>",
  "seed": 101,
  "parameters": {
    "rows": 3,
    "cols": 2,
    "vehicle_count": 600,
    "decision_limit": 48,
    "controller": "pressure"
  }
}
```

不传 `parameters` 或传 `{}` 保留旧接口行为。范围外值、非整数、未知参数、非 2×2 配 `trained` 均拒绝，不静默替换用户选择。

| 入口 | 职责 |
|---|---|
| `demos/live-platform/scenarios.py` | 参数校验、网格/车流生成、压力规则 |
| `demos/live-platform/worker.py` | 场景配置应用、策略选择、CityFlow 逐秒状态 |
| `demos/live-platform/server.py` | 会话、参数持久化、历史、导出、健康状态 |
| `demos/live-platform/static/` | 参数表单、预设、兼容提示、历史配置复用与画布 |
| `demos/live-platform/source_fingerprint.py` | 检查源码与已构建镜像是否匹配 |
| `demos/live-platform/Open-LivePlatform.ps1` / `start-local.sh` | 按需重建后启动本地服务 |

各回合在 `local-state/runs/<ID>/` 独立保存 `request.json`、解析后的 `config.yaml`、`scenario.json`、`scenario/roadnet.json`、`scenario/flow.json`、状态帧与结果。默认场景不生成自定义场景文件。导出的轨迹包含参数与实际控制器信息；原模型登记信息另存于 `source_model`。不要把 `local-state/`、个人运行日志、虚拟环境或 Docker 数据上传仓库。

## 验证与后续工作

纯 Python 回归覆盖参数边界、16 种网格的路口/路线连通关系、0–3000 辆计划发车、2×2 模型协议、合法相位约束、HTTP 参数传递与拒绝、历史恢复、资源健康状态和镜像源码指纹。

2026-11-12 本地验证：Python 32 项通过（平台 15、运行逻辑 8、交通场景 6、源码指纹 3）；JavaScript 语法检查及 PowerShell 语法检查通过；Windows 启动器的缺少镜像、源码一致、源码更新、强制重建、校验错误共 5 种假 Docker 测试通过。尚未执行 Bash 语法与浏览器交互验收。

在安装核心库及其依赖的 Python 环境中，从仓库根目录执行：

```bash
python -m unittest discover -s demos/live-platform -p 'test_*.py' -v
node --check demos/live-platform/static/app.js
```

首次上传时，本机 Docker Linux 引擎不可连接，原生自定义路网验收未完成。后续已在上述 Linux CI 验证 1×1、2×3、3×3、4×4、零车流、3000 辆高车流和 2×2 PPO 的初始化、暂停/单步、完整回合、导出、历史重载及进程回收；并验证 1×2 QMIX 短训练模型的登记与运行。浏览器另验证 MPE 的刷新恢复和对比表。假 Docker 启动器检查仍不能代替真实镜像重建；本轮没有修复本机 WSL 或重建本机镜像。

扩展到其他网格的学习策略，需要训练兼容的模型并调整适配器与兼容元数据，不能仅修改下拉选项或放宽路网摘要检查。继续保留核心算法注册与环境注册的分离；当前网页负责推理和仿真，不发起训练任务。
