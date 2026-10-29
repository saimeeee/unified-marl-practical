# 本机平台发布回执

**已发布并完成下载校验。** [v0.9.0-local.1 预发布](https://github.com/saimeeee/unified-marl-practical/releases/tag/v0.9.0-local.1) 提供完整本机运行源码包、验收结果和 SHA-256 清单。仓库 `saimeeee/unified-marl-practical` 为 public。

| 项目 | 回执 |
|---|---|
| Release ID | `427379363` |
| 源码提交 | `67ff095063b7e8ae8bb5e758c6913610a03e7fe4` |
| 发布时间（UTC） | `2026-10-29T01:50:41Z` |
| 发布类型 | 预发布（prerelease） |
| 附件验证 | 3 份远端摘要及实际重新下载 SHA-256 全部一致 |
| Code → Download ZIP | 已匿名下载，671 个 bundle 文件摘要全部通过 |
| 下载包实际启动 | 中文/空格目录内通过入口 CMD（`-NoBrowser`）启动成功；复用已构建的本机镜像，空记录首屏与 CityFlow 策略实际运行通过 |
| CI | [run 40802692446](https://github.com/saimeeee/unified-marl-practical/actions/runs/40802692446) 已完成，结果 `success` |

## 附件摘要

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `unified-marl-local-v0.9.0-local.1.zip` | 5,506,772 | `7c03fbca2eaf62757bd37723128893e47f934f50907b1f40d15325eee8bc9ce3` |
| `local-verification.json` | 2,128 | `eb9d5832f48c787a2d972e9e823b2caf6890870a0cc141542f4136fcb3d80876` |
| `SHA256SUMS.txt` | 196 | `5fb92f0ec28065c6df2a6b4a3e056974b8a43869a4757dfed04e0473ce8ab40d` |

Release 完整 ZIP 与仓库 Code Download ZIP 均包含本机启动所需源码、四场景模型/资产及 CityFlow 源码。首次 Docker 构建需要网络，构建后 CPU 离线推理已验证；无需 SSH。平台为单会话实时推理与仿真，不提供网页训练队列。四场景验收和效果限制见 [本地验收报告](LOCAL_PLATFORM_REPORT.md)，启动步骤见 [平台说明](../demos/live-platform/README.zh-CN.md)。

此回执记录上述源码提交与发布附件；后续文档补充不改变标签对应的已测源码及附件摘要。原始发布回执保存在本地 `.work/local-platform-publication-receipt.json`，不要求接收方具备该本地文件。
