# 工程应用参数与调用示例

所有示例数据仅展示字段，不是实验结果。三个方向的新代码没有执行训练/环境实验。

- warehouse.yaml + warehouse_agents.csv + warehouse_demand.csv：内置库存补货；warehouse_state.json为模型推理的当前业务状态。
- traffic.yaml + traffic_cityflow_config.json：修改真实路径、路口与信号相位；原生路网/车流和CityFlow引擎需自行提供。
- powergrid.yaml：修改真实建筑/天气/设备数据路径，安装PowerGrid功能依赖。
- train_application.py：显式调用后才训练，需输出空目录。
- decide.py：加载模型和application.json，将当前输入转换为业务决策。

```sh
python train_application.py warehouse.yaml --output-dir ./results/run1
python decide.py warehouse ./results/run1/policy.pt warehouse_state.json
```

完整说明见 docs/APPLICATIONS_GUIDE.md 及各方向指南。Windows使用包内windows/run.ps1，Linux使用linux/run.sh；二者透传unified-marl-app参数。CLI act的--steps保留同回合状态，Python在线调用需保留同一个policy对象。
