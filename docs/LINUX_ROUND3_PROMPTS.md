# 0.8.2 第三轮 Linux 提示词

使用者的当前机器和结果路径以本轮实际输出为准。第二轮报告和最新step0描述的执行位置分别保留，不把旧机器通过记录当成本轮验证。

## 0：安装并核对新包

```text
现在继续第三轮，使用新收到的unified-marl-0.8.2-engineering-hotfix.zip和.sha256。先核对完整SHA256，然后解压到~/marl-hotfix-082。

使用现有venv执行：
~/marl/.venv/bin/python -X utf8 ~/marl-hotfix-082/install_patch.py --bundle ~/marl

必须运行完整安装器，不能只pip装wheel，因为这次也要更新原linux/install.sh。记录真实退出码、当前机器、系统、Python、torch、unified_marl版本和实际导入位置，要求0.8.2，pip check通过。
检查新补丁manifest的fixes包含F1/F2/N1；~/marl/manifest.json应含hotfix_version=0.8.2、base_bundle_version和applied_hotfixes，且原manifest及安装脚本备份存在。用原verify_bundle.py核对新登记文件的完整性，不手工修改manifest来绕过错误。

只操作项目venv，不改全局Python、不改源码。CPU、单进程、seed1，不跑收敛。命令和日志保存到~/marl-test/round3/logs/step0-082.log。失败时保留堆栈，完成后停止。
```

## 1：F1/F2 真实环境检查

```text
复制第二轮已准备好的PowerGrid配置及建筑状态动作JSON到第三轮独立场景目录；保留原数据、物理数值及原JSON不动，注意YAML相对路径必须仍解析到正确资产。
仅在JSON副本对所有选中建筑启用net_electricity_consumption；保留第二轮其他已说明的状态/动作限制与建筑筛选。确保选中至少一个有光伏的建筑。指向副本JSON，设episode_limit=4，CPU、seed1。观测维度会变化，后续必须训练新模型，不加载旧电网模型继续训练。

用新补丁脚本运行：
timeout --kill-after=15s 300s ~/marl/.venv/bin/python -X utf8 -u ~/marl-hotfix-082/round3_powergrid_checks.py --config "第三轮真实YAML路径" --output "本轮新的F1-F2结果JSON路径"

保存日志和退出码。脚本检查光伏逐元素乘法及序列对齐，reset预热保留且control_steps=0；episode_limit分别为1和4时实际产生1和4次外部step，末步info.episode_limit为true，观测和奖励有限。
不能关闭净用电量字段或把limit改为5来通过。该脚本只检查环境，不代表训练或恢复通过。完成后停止。
```

## 2：新模型更新、截断与完整恢复

```text
使用上一步重新启用净用电量状态的第三轮场景，创建新电网模型。CPU、seed1、updates=1、horizon=4、episodes_per_update=1、mini_epochs=1、episode_limit=4；网络可明确设置hidden_dim=32以缩小测试，保留物理场景和奖励不变。

通过公开powergrid.train发起短训练，最多300秒；检查四个真实控制transition、末端bootstrap观测、时间截断与真实终止掩码，以及有限损失和实际参数更新。不能只用“优化器被调用”作为更新证据，可在独立测试脚本记录step前后参数差异，不修改安装包源码。

保存后新进程加载policy.pt，用该环境真实观测推理，核对设备名称、动作形状及Box边界，reset后重载推理一致。
经公开powergrid.train使用完整training.pt恢复到新目录，目标总updates=2，确认计数1→2、环境步数增加、参数再次更新；检查actor/critic/优化器及实现承诺的训练状态确实恢复。policy.pt作resume应被拒绝。
对恢复后模型评估一个回合，应episode_lengths=[4]且env_steps=4。旧0.8.1电网训练状态因回合语义不同被拒绝是明确兼容边界，不修改旧checkpoint绕过。

保存日志、指标及产物路径，不评价收益。完成后停止。
```

## 3：N1 功能安装不降级

```text
先记录当前核心版本0.8.2和setuptools实际版本，再使用原路径执行：
bash ~/marl/linux/install.sh --Feature powergrid

这次必须直接成功并保留核心0.8.2，不允许安装后再打补丁来验收。记录pip check、包版本/导入位置，以及setuptools是否被无意改回旧锁定版本。
重复相同命令一次，确认仍成功、核心不降级；用verify_bundle.py核对完整性，新manifest的旧版本备份记录应保留。
联网失败记录为阻塞，依赖冲突记录原始错误。不得删除已安装依赖或手工改锁文件来伪装成功。此步骤不重新编译CityFlow、不训练。完成后停止。
```

## 4：仓储、交通快速回归

```text
所有操作从安装包调用，使用0.8.2、CPU、seed1和新输出目录。
仓储：运行补丁保留的round2_warehouse.py，一次更新、API/CLI恢复、保存加载、合法补货和4步评估应通过。
交通：复用第二轮固定版本的CityFlow与官方单路口场景，不重新编译。一次PPO有效更新、保存加载、合法相位推理和8步评估。保留yellow_time=0的示例限制，不宣称已验证安全过渡。若声称交通支持完整恢复，再补一次updates1→2恢复检查。
每个训练任务300秒超时，记录真实退出码、参数变化、有限损失和日志。失败只诊断，不改源码。完成后停止。
```

## 5：报告

```text
生成~/marl-test/round3/TEST_REPORT_ROUND3.md及results-round3.json。
分别报告F1、F2、电网有效更新、完整保存恢复、评估计数、N1重复加装、仓储/交通回归。状态仅通过/失败/阻塞/未测试，附实际机器、版本、命令、退出码、指标、日志、场景及结果路径。
列明公开替代数据、仍关闭的其他状态、排除建筑及启用的设备动作，不能写成全部组合通过。
全部必测项通过后结束本轮最小功能验收；仍有失败时只指出需要修复和复测的具体项目，不反复重跑所有测试。不宣称收敛、收益或真实工程效果。
```
