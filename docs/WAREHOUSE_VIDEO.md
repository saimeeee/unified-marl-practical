# Windows 本机仓储补货视频

这是现有 `unified_marl` 的真实调用例子：MFVFD 在库存环境短训练，保存并加载模型，记录完整回合，最后把记录渲染为 MP4。不是车辆配送或仓间运输路线。视频中的动画圆点只是到货事件的可视化，不改变环境数据。

默认包含 2 个仓库、每仓 2 个商品，共 4 个智能体；24 个周期；120 次有效更新。CPU 即可，不需要 CUDA、CityFlow、电网资产或远程机器。需求由固定随机种子生成，模型反复使用此小场景训练和展示，不能当成独立测试、收敛或实际收益证据。短训练表现不好也会原样记录，不挑片段或替换为规则策略。

## 在 VS Code 运行

1. 打开项目根目录的 `仓储补货演示.code-workspace`。
2. 打开 `examples/applications/warehouse_video.py` 查看调用代码。
3. 按 `Ctrl+Shift+B`，运行“生成仓储补货视频（CPU）”。这个入口不需要安装 Python 调试扩展。
4. 完成后，终端会打印视频和数据路径。每次运行在 `results/warehouse-video-时间戳/` 新建目录，不覆盖之前的模型和视频。

也可以在该工作区的 PowerShell 终端运行：

```powershell
& .\.venv\Scripts\python.exe -X utf8 -B .\examples\applications\warehouse_video.py --updates 120
```

本机已将绘图与编码依赖单独安装到 `.work/scheduling-video-deps`，未改动全局 Python。换机器或目录缺失时，在项目终端安装：

```powershell
& .\.venv\Scripts\python.exe -m pip install --target .\.work\scheduling-video-deps -r .\requirements-video.txt
```

不要把当前机器的虚拟环境直接复制到另一台机器。先按项目安装说明建立接收方环境及 HMF 功能依赖，再安装上述可选视频依赖。本例渲染器使用 Windows 微软雅黑字体，未在 Linux 验证。

## 对应的核心调用

```python
from unified_marl.applications import warehouse

# 配置提供库存、需求、容量、成本与提前期，输出到新目录。
result = warehouse.train("scene/warehouse.yaml", output_dir="results/my-model")
policy = warehouse.load_policy(result.checkpoint, device="cpu")
policy.reset()  # 每个回合开始必须调用。
decision = policy.act(state, available_actions)
# decision["decisions"] 包含 agent_id、action_index、requested_order、order_quantity。
```

`state` 按仓储指南提供当前库存、上期需求与在途订单；不向模型提供未来需求。示例脚本调用真实 `InventoryEnv.step`，并核对决策映射、容量限制、库存守恒和终止位置。

## 改自己的场景或只录已有模型

默认生成的 `scene/warehouse.yaml`、`warehouse_agents.csv` 和 `warehouse_demand.csv` 是参数模板。复制到自己的目录后替换业务字段：

```powershell
& .\.venv\Scripts\python.exe -X utf8 -B .\examples\applications\warehouse_video.py --config .\my-scene\warehouse.yaml --updates 120

& .\.venv\Scripts\python.exe -X utf8 -B .\examples\applications\warehouse_video.py --checkpoint .\results\已有任务\model\policy.pt
```

第二条只做模型推理和录制，不重新训练。必须保留 checkpoint 旁的 `application.json`、`application-result.json` 及其中引用的数据。若传入 `--config`，场景维度、智能体顺序与动作语义必须匹配。当前视频排版限制为 1–4 个智能体、`inventory` 后端及 `backlog=false`；更多智能体需要扩展布局。

## 产物

- `warehouse_mfvfd.mp4`：1280×720，默认 12 fps、每周期 1 秒，H.264 编码，可插入 PPT。
- `preview.png`：视频首帧。
- `rollout.csv`、`rollout.json`：逐周期真实补货、到货、需求、销售、库存和利润。
- `scene/`：默认合成场景参数和需求。
- `model/`：推理模型、完整训练状态、配置与训练指标。
- `manifest.json`：解释器、包版本、更新次数、数据边界与视频/模型 SHA-256。

运行使用项目隔离解释器。生成后会完整解码 MP4 检查编码，不能据此声称业务效果验证通过。
