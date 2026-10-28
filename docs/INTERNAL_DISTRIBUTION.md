# 内部可分发安装包

日期：2026-10-21；核心版本 0.7.0。用户要求内部整理和复用，分别提供 Windows x64 与 Linux x86_64 的 CPU / CPython 3.12 版本。本次不建设 Web、账号权限或集群管理系统；不改变算法支持范围。

## 交付方式

dist/internal 下每个平台一个 ZIP。发给同事原 ZIP，解压到可写目录后安装；不发送本机 .venv。接收方需预先安装 CPython 3.12 x64 及 venv/pip 和必要系统运行库。Python 解释器未捆绑，安装脚本不自动修改系统运行库。

每版提供核心 wheel、对应平台的依赖、安装清单、SHA-256 文件清单、安装与调用脚本、短训练检查脚本、中文文档和示例、源码发行包。Windows 的依赖完整，安装只从包内 wheelhouse 读取，不访问网络。Linux 为混合依赖版：已准备的依赖按固定版本/哈希从本地安装；huggingface-hub 的 Linux 条件依赖 hf-xet 未获本机下载许可，由接收方安装时从 PyPI 按 requirements-network.txt 的明确约束补齐，不能标为完整离线包。两版均使用独立 venv，不改变全局 Python。完整性清单用于发现损坏或内容变化，接收方仍应通过可信内部渠道获取 ZIP 及其外部 SHA-256。

Windows 保留系统默认路径策略。若解压目录过深，PyTorch 长文件名可能超过 260 字符；安装器先估算完整路径并明确报错，要求换较短的可写目录，不自动修改系统注册表。中文和空格本身不构成限制。

真实数据、模型、训练结果、外部仿真器及 GPU/CUDA 不在 ZIP 中；data/README.md 说明数据放置。需要真实环境时，按支持矩阵准备后端与资产。固定依赖是本次分发的安装约束，不能当成任意 OS/版本兼容承诺。

## 接收方操作

Windows：先安装 CPython 3.12 x64，打开 PowerShell，进入解压目录：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
# 无法自动找到 Python 时：
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -PythonPath "C:\Python312\python.exe"
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 --help
.\.venv\Scripts\python.exe examples\online.py --output .\results\demo
```

Linux：先安装系统适用的 CPython 3.12 与 venv/pip，解压对应 ZIP，然后：

```bash
bash install.sh
bash run.sh --help
.venv/bin/python examples/online.py --output ./results/demo
```

具体系统库/架构边界和参数以每个 ZIP 内的 README 为准。Windows 安装默认执行 QMIX toy 的实际更新、保存加载、评估和恢复检查。按用户最新指令，Linux 只准备安装文件，不在本机执行功能测试，安装脚本也不自动启动训练；接收方可自行运行 `python smoke_test.py --output ./results/manual-check`。toy 不代表真实仿真器或算法效果。

自己的项目可使用本包 .venv 的解释器，直接 `from unified_marl import train, evaluate, load_policy`。正常使用不需要原始交付目录，不需要 GitHub 访问权限。换电脑或移动目录后从 ZIP 重新解压安装，避免虚拟环境中绝对路径失效。

## 验证记录

Windows 最终包的全新目录安装、离线依赖、中文/空格路径、重复安装、命令行、损坏检测和最小训练结果在 `.work/win-final/validation.json`，摘要见 INTERNAL_WINDOWS_REPORT.md 及 dist/internal/windows-verification.json；早期记录保留在 `.work/internal-windows`。Linux 的构建与验证层次见 INTERNAL_LINUX_REPORT.md。最终结果以本次收尾的 STATUS 和实际产物清单为准，不回写历史验收证据冒充旧记录。

目前没有另一台实体电脑的验收证据；本机全新解压及环境验证必须按此标注。核心包既有支持/未支持清单和 383 行迁移记录继续保留。
