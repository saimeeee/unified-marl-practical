# 独立复核报告：unified-marl 0.7.0

日期：2026-10-21（UTC+08:00）

## 结论

本次独立复核确认：本地 0.7.0 源码、wheel、sdist、迁移追踪和已记录限制彼此一致；选取的 CPU 更新、真实离线数据、保存恢复、来源对照及本机 ZMQ 检查通过。适用范围仍仅为支持矩阵中列出的子集。没有真实仿真环境、GPU、跨机或策略效果验证。

GitHub 私有仓库的当前可见性、`main` 提交/tree 和远端 687 个 blob **本次无法独立确认**：运行环境没有 GitHub 仓库内容连接器，项目目录也没有 `.git`。`.work/github-handoff/published.json` 是先前生成的本地回执，虽声称仓库为 private、提交 `0f0320d649f2f3106db910297ba84e5a5eeb3f8a`、tree `46763107a15dbd0805d47ca681a48a3a7f2afe27` 和 687 个远端文件，本次不能把该回执当成远端证明。因此，本地发行包可作为**有明确范围限制的源码交付候选**；当前 GitHub 仓库仍未取得独立远端验收结论。

## 严重问题与重要限制

1. **远端交付状态未验证（交付证明阻塞，不是代码失败）。** 可用工具中没有读取私有 GitHub 仓库内容或提交/tree 的连接器，且本地不是 Git 仓库，无法检查 visibility、`main` HEAD 或任一远端 blob。没有尝试绕过权限，也没有读取或输出凭据。若要把 GitHub 仓库作为已验收交付，仍需在有授权 GitHub 连接器的环境复核回执中列出的 commit/tree 与文件。
2. **发行验收脚本在当前 Windows 编码环境有可复现性问题（低至中风险，非包运行缺陷）。** 未设置 UTF-8 环境变量时，基础验收在启动中文/空格路径 CLI 后因 `gbk` 子进程输出被 `tools/validate_release.py` 按 UTF-8 读取而退出 1（异常位置约第 153 行，`UnicodeDecodeError`，后续 `child.stdout` 为 `None`）。本次只在该验证进程设置 `PYTHONIOENCODING=utf-8` 和 `PYTHONUTF8=1` 后，基础验收退出 0。现有报告称已覆盖中文路径，但没有记录这两个运行时变量；这是验收命令的环境要求遗漏。
3. **许可和算法范围限制仍成立。** `docs/LICENSES.md` 对王思哲、寇谦及易闭源配置的授权仍待确认。支持矩阵注明 HMF 适配没有通用动作掩码处理，本次 HMF toy 验收关闭了动作掩码；不能据此宣称 HMF 任意受限动作环境通过。RODE 扩角色优化器、HMF 动态/高层流程、Kou 缺失 Learner/数据协议、连续 PowerGrid、真实环境、跨机、GPU 和长期策略效果仍按 `KNOWN_ISSUES.md` 保留。

本次没有发现足以推翻 0.7.0 支持子集结论的新算法缺陷。来源损失对照通过不证明来源算法理论正确、长期稳定或收敛。

## 本地交付、发行包和迁移追踪

- 本地上传清单有 687 条唯一相对路径，合计 8,836,014 字节。逐文件重新计算后，缺文件、大小/SHA 不符、重复/越界路径和非声明根目录均为 0。清单路径未包含原始数据树、模型、虚拟环境、缓存或 `dist`。
- 常见私钥标记、AWS access key、GitHub token 和 Bearer credential 模式扫描未命中；没有大于等于 10 MB 的清单文件。`docs/evidence/data_metadata_tokens.json` 因文件名包含 `tokens` 被路径规则标出；其记录字段是 `path`、`pickle_string_tokens`、`method`，未发现上述凭据模式。该结论仅覆盖本地清单和这些模式，不能代替远端扫描。
- 1,228 条来源清单哈希全部匹配；353 条迁移追踪目标哈希全部匹配，所列来源文件均存在。迁移矩阵保留 383 行：301 行 `local_test_verified=YES`，82 行为 `NO`。状态栏将真实数据文件读取、真实数据训练更新和真实环境分别区分；383 行的 `real_environment_verified` 全为 `NO`，没有 `policy_effect_verified=YES` 或跨机通过项。已知问题文档把 82 行解释为 78 个第 1—7 阶段未验项加 4 个可选阶段 8 项，和阶段边界一致。
- `real_data_verified` 的矩阵值包括 62 个 `UPDATE_PASS`、3 个 `YES`、1 个 `UPDATE_INPUT_PASS` 和 19 个 `READ_PASS`；`READ_PASS` 只表示输入/归档可读，不能称为策略训练通过。本次额外复跑实际载入 3m 数据并执行短更新，数据 SHA-256 为 `11163b7809c9258f37954777c2491334b4ac535f01ca1c24d20a157adc77c8d0`。
- wheel 为 516,838 字节，SHA-256 `ba6fec1d7eb805a86ef62ae61fa9e7a866bdbfc49ad6a1c8be1d3cbe27e445e7`；sdist 为 545,900 字节，SHA-256 `c4319d229cb702b0fe9c8010158a6275497b8a74d77f8ce1d10570cd05aeea7e`，均与 `dist/SHA256SUMS.txt` 一致。wheel 中 354 个运行时 `.py/.yaml/.json` 文件与 `src/unified_marl` 字节相同；wheel 363 个条目、sdist 546 个条目均未发现数据/模型/字节码/缓存条目。
- 源码外发行验证从 `.venv-sdist` / `.venv-release` 的 `site-packages` 导入 0.7.0；脚本审计钩子拒绝读取原始源码树中的 Python/YAML 文件，最终报告 `source_imports=[]`。显式输入 3m 数据不等于包依赖原始源码目录。full 运行结束后没有遗留子进程。

## 本次实际运行

运行栈：Windows x64、Python 3.12.14、PyTorch 2.6.0+cpu、NumPy 1.26.4；只使用项目虚拟环境，没有改全局 Python。没有重跑全部 220 项回归。

| 命令/检查 | 结果 | 覆盖与边界 |
|---|---|---|
| `.venv-sdist` 的 `validate_release.py --mode base`，UTF-8 环境变量设置后 | 退出 0，12 条链路通过 | PyMARL、SADMA、离线 BC/CPL；可选依赖确实缺失时，role/HMF/reward/ZMQ 明确报错；有限损失、正参数变化、掩码动作、保存加载、toy 评估 |
| `.venv-release` 的 `validate_release.py --mode full`，经授权在沙箱外使用回环 TCP | 退出 0，28 条链路通过 | RODE/RODE-HMF、两个 HMF 变体、7 个 SADMA 配置、4 个离线策略和 3 个额外 Agent 变体、6 个奖励模型，以及 ZMQ Q sync/async、PPO sync/async V-trace |
| 复核隔离副本中的选定 pytest 回归 | 24 passed，0 failed | 分项为 15（terminal/truncation、padding、QMIX 恢复、BC/CPL 离线恢复、6 奖励模型恢复、SADMA 时间/掩码语义）、1（PPO V-trace 旧策略概率/来源更新对照）、4（worker 异常/停滞、连接超时、启动绑定失败清理）、1（ZMQ PPO V-trace 版本/计数/进程和端口释放）、3（PyMARL QMIX、Kou CPL、MR 奖励模型来源损失/参数对照） |
| 原有 `docs/evidence/stage7/tests.xml` | 220 passed、0 failures、0 errors（历史记录） | 本次只读取该证据，没有重跑全部套件 |

真实 3m 数据 full 复跑包含 BC、CPL、ICQ-MA、old QMIX 各 2 次更新并继续到 3 次；BC `rnn_ns`、old QMIX `mlp` / `mlp_ns` 三个变体同样各更新并恢复。六个奖励模型均有实际参数变化，预测保存/加载一致，并恢复到下一次固定批次更新。它们是 CPU 短更新，不是仿真环境评估或策略效果验证。

来源抽查结果：PyMARL QMIX 的网络输出、损失、更新参数与原实现最大误差均为 0；Kou CPL 的损失和更新网络最大误差均为 0；MR 奖励模型的损失、更新网络和 source-offline 预测最大误差均为 0。PPO V-trace 的旧行为概率保持逐位不变，更新损失与 actor 参数和适配后的来源参考对照误差为 0。

### ZMQ 本机回环

本次通过回环运行了 Q sync/async、PPO sync/async（异步 PPO 显式启用 V-trace）。四组 execution 状态为 `PASS`，无强制结束，评估子进程正常，停止后每个端口可重绑。版本确认分别记录了 Q sync `0,1,2`、Q async `0,0,1,1,2,2`、PPO sync/async `0,2,4`。另有一次 pytest 专门断言了版本确认递增、同版本 worker digest 一致、样本/调用计数闭合、所有子进程退出码为 0、评估完成且端口释放。

worker 抛错、worker 停滞、未监听端点超时和推理端口绑定失败四项异常测试均在实际回环执行中通过，报告了有界超时并清理自有进程/端口。默认沙箱创建 ZMQ Context 曾返回 `Permission denied`，只记为权限限制；获准沙箱外复跑后才计为本次通过。**这些结果只验证 Windows 本机回环，不验证跨机网络、真实环境、吞吐或加速。**

## 与先前交付结论的差异

本次复核没有改变 0.7.0 的已支持子集和未完成范围。此前报告中“真实环境、GPU、跨机、效果未验证”的限制继续成立；本次本机 CPU 和 3m 数据短更新复跑为正向补充。新增发现是：GitHub 远端在本次无法独立读取，以及基础验收 CLI 在 GBK 输出环境需显式设置 UTF-8。除此之外未发现将阶段完成等同全部迁移完成的夸大。

## 仍未验证与交付建议

未验证项：GitHub 私有仓库和远端 687 个 blob；真实 SMAC/StarCraft II、ReplenishmentEnv、CityFlow；GPU；Linux/macOS 和其他 Python/PyTorch 版本；跨机 ZMQ；长时间收敛和策略效果；未迁移或部分迁移能力；待确认的来源授权。

建议把**本地** 0.7.0 包作为“支持范围已记录、限制明确的源码交付”继续使用。当前私有 GitHub 仓库在补做远端 connector 核验前，应标为“交付回执待独立确认”，不能称为本次已经验证的远端交付。这个报告只新增于 `docs/INDEPENDENT_REVIEW_LUNA_MAX.md`；原验收报告及原始交付没有改动，也没有推送新提交。

## 复核证据位置

独立证据保存在 `.work/independent-review-luna-max/`：`local-audit.json`、`source-copy-check.json`、`attempts.json`、`base-repeat-utf8/report.json`、`full-repeat-loopback/report.json`，以及 `regression-project/.work/*.xml`。该目录和本报告都不在先前 687 项 GitHub 上传清单中。

