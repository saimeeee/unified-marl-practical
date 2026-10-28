# 0.8.2：F1 / F2 / N1 修复包

本包升级已经安装的0.8.0/0.8.1 Windows或Linux工程环境，保留仓储0.8.1的恢复和评估修复。不是新的平台依赖大包，不含CityFlow、建筑数据或模型。旧0.8.0/0.8.1文件保留，不要再安装旧补丁覆盖本版。

## 修复内容

- **F1**：净用电量观测边界计算中的光伏序列先转换为浮点数值数组，再乘装机容量；检查光伏、负荷、热水和制冷序列长度一致且有限，并修复相同的光伏历史日志列表乘法。不改变物理数值、奖励或边界公式。第三轮须在配置副本重新启用`net_electricity_consumption`，不能继续关闭它来验收。
- **F2**：保留来源reset的随机初始化动作及潮流计算，`initialization_steps=1`；`steps`从0开始，只统计外部策略控制。`episode_limit=4`代表4次外部动作；info提供control_steps、initialization_steps和episode_limit截断标志。新checkpoint标记`control_transitions_v2`。旧电网checkpoint仍可不创建环境地加载推理，但恢复/环境评估遇到旧回合语义会明确拒绝；本轮必须重新训练电网模型。启用新观测也会改变观测布局。仓储、交通不受此语义标记影响。
- **N1**：更新解压目录中的`linux/install.sh`或`windows/install.ps1`，原命令不变，转由新功能安装器处理。该安装器不再安装旧锁文件中的unified-marl条目，保留已安装依赖及构建工具，仅补缺失的本地依赖；联网功能使用核心版本约束，并在结束时检查依赖和核心版本。重复执行也不能降级。存在依赖冲突时明确报错，不自动覆盖已有工具版本。CityFlow不由该命令重新编译。

## Linux 安装（现有 ~/marl/.venv）

先核对ZIP旁的.sha256。解压到独立目录，确保直接包含install_patch.py及wheel：

```bash
unzip unified-marl-0.8.2-engineering-hotfix.zip -d "$HOME/marl-hotfix-082"
"$HOME/marl/.venv/bin/python" -X utf8 "$HOME/marl-hotfix-082/install_patch.py" --bundle "$HOME/marl"
"$HOME/marl/.venv/bin/python" -I -c 'import unified_marl; print(unified_marl.__version__, unified_marl.__file__)'
```

必须完整执行install_patch.py，**单独pip安装wheel不能更新旧功能安装器，不能算N1修复完成**。安装器先验证原包完整性，再离线安装本版wheel，备份原安装脚本和manifest，并登记本次明确变更的文件大小及哈希。新manifest保留base_bundle_version和applied_hotfixes记录；备份在`.unified-marl-hotfix/backups/pre-0.8.2/`。这是受控的发行文件更新，不是关闭原校验或接受任意修改。原包已损坏/被手工改动时会拒绝更新，不刷新错误文件的哈希。

安装完成后，原命令可以直接加装功能，不需要再次打补丁：

```bash
bash "$HOME/marl/linux/install.sh" --Feature powergrid
"$HOME/marl/.venv/bin/python" -m pip check
"$HOME/marl/.venv/bin/python" -I -c 'import unified_marl; assert unified_marl.__version__=="0.8.2"'
```

PowerGrid可选依赖可能联网下载；补丁本身不下载依赖。可以先运行`bash ~/marl/linux/install.sh --Feature powergrid --dry-run`查看计划，dry-run不等于安装通过。原始wheelhouse及其0.8.0锁文件保留供追踪，不应手动改写。

## Windows

```powershell
& C:\marl\.venv\Scripts\python.exe C:\marl-hotfix-082\install_patch.py --bundle C:\marl
powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\marl\windows\install.ps1 -Feature powergrid
```

## 第三轮

参阅[LINUX_ROUND3_PROMPTS.md](LINUX_ROUND3_PROMPTS.md)。新脚本`round3_powergrid_checks.py`可检查真实资产下的F1计算和F2实际step计数，它不训练策略；有效更新、截断批次、完整恢复和回归仍按提示词单独进行。`round2_warehouse.py`也随包保留并接受0.8.2。

本机28项无模拟器数学/调用计数/安装事务单元回归通过；Windows隔离venv实际升级、重复基础功能加装及重复补丁通过（依赖只读复用既有release环境，非干净依赖验证）。另完成构建、源码归档及包资源检查。未在本机启动CityFlow、PowerGrid潮流、训练或Linux环境。原报告中的公开替代数据、关闭的其他状态、建筑筛选和三设备动作范围均保留，不能宣称全部建筑/设备控制已验证。0.8.2真实Linux结果以用户第三轮报告为准。

## 第三轮结果已收到

上述等待第三轮为构建当时的历史状态。用户现已提供七项限定通过报告，正式发行保留本补丁与wheel原始字节；新验收与安装说明见[VALIDATION_REPORT_082.md](VALIDATION_REPORT_082.md)和[ENGINEERING_RELEASE_082.md](ENGINEERING_RELEASE_082.md)。reset观察与干净依赖路径继续保留，不代表全部组合通过。
