# Windows 内部分发包验收

日期：2026-10-21；核心 0.7.0，分发修订 1。目标为 CPython 3.12 / Windows x64 / CPU；不捆绑 Python、系统运行库、真实数据、仿真器、GPU 或本机虚拟环境。

## 最终产物

`dist/internal/unified-marl-0.7.0-windows-x64-cpu-py312-offline.zip`，307,140,455 字节，SHA-256 `1a824094e15a6a5ebd9c2f730a5e16d7fce0601230221c824da199a565e193f8`。77 个清单文件（另有 manifest 自身），36 个 wheel，覆盖基础与 role/hmf/reward/distributed 的固定依赖。核心 wheel 保持原始 0.7.0 哈希，354 个运行时源码/资源和当前 src 内容一致。

## 实际验收

Windows 11 / CPython 3.12.14 x64 / PyTorch 2.6.0+cpu。使用 Windows PowerShell 5.1，从最终 ZIP 解压到新建的中文与空格目录，使用全新 venv、关闭索引及缓存、仅从包内 wheelhouse 按哈希安装。

- 安装退出 0；pip check 退出 0，完整可选依赖导入退出 0。
- 默认安装检查实际执行 QMIX toy 两次更新、有限损失、参数变化、动作掩码与隐藏状态重置、推理保存加载、两回合评估、完整状态恢复至三次更新。
- 原位置重复安装、从其他当前目录调用 CLI 均退出 0。
- 修改 README 后校验器退出 1；恢复内容后退出 0。篡改检测是完整性检查，不能替代可信渠道与外部 ZIP 哈希。
- 过深路径预检查在 pip 执行前明确退出 1，不修改系统注册表。

实际命令和每步退出码在 `.work/win-final/validation.json`，逐步日志在同目录；训练报告路径由该 JSON 的 smoke_report 给出。外部摘要为 `dist/internal/windows-verification.json`。

执行命令：

```powershell
.venv-release/Scripts/python.exe -B tools/build_windows_internal_delivery.py --output .work/internal-windows/final-build-r1
.venv-release/Scripts/python.exe -B .work/internal-windows/validate_bundle.py --work .work/win-final
```

功能验收后仅更正两份跨平台分发说明并重算 manifest/ZIP 哈希；安装脚本、wheel、示例和运行代码与已验收 ZIP 逐字节相同，未重复运行相同训练。基准 ZIP 哈希及文档差异保存在 .work/internal-windows/final-documentation-refresh.json。

## 发现与处理

首次普通深度中文/空格目录离线安装通过。中间验证目录比首次多一层，PyTorch 长头文件名超过系统默认 260 字符限制，pip 实际退出 1；日志保留在 `.work/internal-windows/final-validation/offline-install.log`。最终安装器新增提前估算与清晰提示，换较短目录后最终整条验收通过；过深目录仍为明确拒绝，不能声称任意路径深度支持。

Python 调用显式使用 `-X utf8`，因为 `-I` 会忽略 PYTHONUTF8 等环境变量。没有修改算法、全局 Python、系统长路径策略或原始交付。

本次是同一实体电脑的全新解压/全新环境验收，不是另一台实体电脑验证；其他 Windows 版本、Python 补丁版本、GPU、真实仿真与长期效果均未增加验证声明。Linux 见对应报告，不能由 Windows 结果外推。
