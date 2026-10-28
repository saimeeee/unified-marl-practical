# 0.8.2 工程修复版使用说明

本版供Windows x64与Linux x86_64、CPython3.12、CPU使用。第三轮用户报告中的仓储MFVFD、交通PPO和电网bounded_ppo最小链路已通过，范围及未验证项见[验收说明](VALIDATION_REPORT_082.md)。

## 新安装

1. 从`v0.8.0-engineering.1`下载对应系统的工程基础ZIP，解压到短且可写的目录。
2. 先用基础包安装器安装`warehouse`或所需基础功能，创建自己的`.venv`。不安装到全局Python。
3. 下载本次`unified-marl-0.8.2-engineering-hotfix.zip`及`.sha256`，核对校验值后解压到另一目录。
4. 用基础包`.venv`的Python执行补丁中的`install_patch.py --bundle 基础包目录`。它同时更新核心和原路径功能安装器。

本版小补丁不包含平台依赖wheelhouse，也不包含Python解释器；无需重下已持有的基础ZIP。已装0.8.0或0.8.1可以直接升级。已完整安装同一0.8.2补丁时重复安装是幂等操作。

## Linux

```bash
sha256sum -c unified-marl-0.8.2-engineering-hotfix.zip.sha256
unzip unified-marl-0.8.2-engineering-hotfix.zip -d "$HOME/marl-hotfix-082"
"$HOME/marl/.venv/bin/python" -X utf8 "$HOME/marl-hotfix-082/install_patch.py" --bundle "$HOME/marl"
bash "$HOME/marl/linux/install.sh" --Feature powergrid
"$HOME/marl/.venv/bin/python" -I -c 'import unified_marl; assert unified_marl.__version__=="0.8.2"'
```

## Windows

```powershell
& C:\marl\.venv\Scripts\python.exe C:\marl-hotfix-082\install_patch.py --bundle C:\marl
powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\marl\windows\install.ps1 -Feature powergrid
```

必须执行完整补丁安装器。只安装wheel不能更新旧功能安装入口。原脚本和manifest保存在`.unified-marl-hotfix/backups/pre-0.8.2/`；新清单记录精确变更，不手工改清单绕过损坏检测。

## 使用与模型

包内`unified-marl-app init warehouse|traffic|powergrid --directory 场景目录`导出模板；`check`检查参数，`train`训练，`evaluate`评估，`act`推理。完整Python接口和各方向参数见现有工程指南。应用模型必须与`application.json`一起保留。

本版修复仓储恢复时自动seed字段误判、电网光伏列表乘法、电网预热与控制计数、功能加装降级。电网旧回合语义checkpoint不能继续训练或创建环境评估，应新建模型；纯推理加载和仓储/交通接口不因此改写。场景观测字段发生变化也需要匹配的新模型。

CityFlow编译、路网/车流和电网建筑气候资产仍由使用者准备；第三轮实际准备脚本尚未取得。本版不假装包含这些资产。原报告中的固定CityFlow/pybind11版本及场景限制已记录于验收说明。

## 可选定向诊断

补充文件ZIP含`check_powergrid_seed.py`。在已准备PowerGrid资产的机器运行：

```bash
"$HOME/marl/.venv/bin/python" check_powergrid_seed.py --config 场景.yaml --output seed-check.json --mode both
```

该诊断新建环境并执行固定动作，不训练；完整初始化随机源与私有工厂路径分别启动两个进程比较。尚未定位的reset差异继续保留，不能仅凭策略对相同输入一致就声称环境可复现。

本版发布不包含新的长训练、真实业务效果或全功能兼容承诺。
