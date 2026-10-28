# 0.8.1：根据 Linux 第一轮报告调整

## 报告结论与修复

用户提供 TEST_REPORT.md：Debian13 x86_64、CPython3.12.15、torch2.6.0+cpu；实际执行在Grok Bot盒子。用户报告0.8.0仓储MFVFD完成合成小样例有效更新、保存、推理及评估，但应用API/CLI恢复失败。本项目未取得该机器和原始日志，记录为“用户报告”，不作为本机或真实业务验证。

原因：核心train保存的env_args带有自动注入的seed，应用层调用的env_args省略seed。原严格字典相等检查把省略的字段视为变更。0.8.1采用与现有SADMA/ZMQ路径一致的检查方式：以已保存参数为基准合并显式参数，所有显式参数必须相等。省略值沿用checkpoint，显式改seed/数据路径/环境选项依然拒绝；应用层仍检查完整场景一致。不改变算法、奖励、损失、采样或训练状态格式。

评估新增`episode_lengths`和`env_steps`；统计Runner记录的真实环境转移数，不含reset、bootstrap状态或策略推理。旧returns/return_mean/return_std等字段保留。PowerGrid独立评估结果不因本次新增字段而被改写。

交通缺CityFlow和路网、电网缺建筑气候资产属于前置条件阻塞，当前报告没有它们的训练失败证据。0.8.1不伪造资产，不声称解决了外部依赖或完成三方向实景验证。

## 小型补丁安装

本补丁复用0.8.0工程包已经安装的依赖，Windows和Linux共用wheel，不用重新下载两个大ZIP。先安装原工程包warehouse功能，再应用补丁。将补丁ZIP解压到独立目录，保留原包的manifest、wheelhouse和旧结果。

Linux：

```bash
"$HOME/marl/.venv/bin/python" "$HOME/marl-hotfix-081/install_patch.py" --bundle "$HOME/marl"
```

Windows示例：

```powershell
& C:\marl\.venv\Scripts\python.exe C:\marl-hotfix-081\install_patch.py --bundle C:\marl
```

安装器校验补丁manifest，限定原包venv且已有0.8.0/0.8.1，只离线更新核心wheel并运行pip check；不会训练。

**原0.8.0安装器会重新安装0.8.0。**如为powergrid等功能运行原安装器，请先装功能、再重新应用补丁，核对最终版本0.8.1。不要改原manifest或锁文件绕过验证。补丁文件SHA256与ZIP旁.sha256用于完整性核对；wheel也可单独通过目标venv的pip以`--no-index --no-deps`安装。

## 二测

分步提示词见[LINUX_ROUND2_PROMPTS.md](LINUX_ROUND2_PROMPTS.md)。补丁自带`round2_warehouse.py`：生成新合成数据、训练一次、经API和CLI各恢复至第二次更新，评估4步、加载推理、拒绝改场景/改seed/policy恢复。提供`--previous-run`可单独兼容旧0.8.0结果（原CSV路径必须仍有效）。脚本必须由用户在测试机器显式运行；本次开发没有执行它。

本机检查为无仿真的单元回归、源码语法、wheel构建、隔离目标安装与资源检查。当前Linux热修复后的实际训练恢复仍待用户二测，交通/电网真实环境和策略效果仍未验证。具体命令见STATUS及evidence/engineering-hotfix。
