# 0.8.0 工程应用实现与检查说明

本轮按用户要求实现和打包，由使用方测试。没有运行训练、参数更新、库存仿真、CityFlow、pandapower、电网或真实场景实验；没有新增收敛或策略效果声明。

## 交付代码

- warehouse：新增参数化InventoryEnv；业务CSV校验、库存/在途/提前期、容量预留、利润奖励、MFVFD/HMF等现有学习器接入、补货业务输出。
- traffic：新增原生CityFlow.Engine适配；真实路口/车道/phase解析，动作mask，显式全红过渡、排队奖励、路口相位业务输出。
- powergrid：将来源GridEnv/Building移到包内并参数化路径；新增独立bounded Gaussian PPO，提供连续动作概率变换、旧概率、GAE、完整训练状态与设备控制输出。它是新增适配实现，不声称与来源PPO数值一致。
- applications：统一config/train/evaluate/load_policy与init/check/train/evaluate/act CLI；场景manifest、智能体排序/动作语义检查及模型可搬迁使用。
- Windows/Linux：复用原CPU CPython3.12目标wheelhouse，新核心wheel、源码包、应用模板、示例和中文说明。

## 实际检查范围

已做源码AST解析、应用API导入、YAML/JSON/CSV参数契约检查、资源一致性及文档链接检查。PowerShell安装/运行脚本Parser检查；当前电脑没有可用bash，未执行Linux脚本。发行构建与归档完整性结果另保存于dist/internal/engineering-delivery-build.json及发布DELIVERY_MANIFEST、SHA256SUMS。

检查参数模板未创建环境或调用策略。示例库存/需求数值仅展示数据格式，不作为实验数据或结果。交通和电网模板需用户填真实资产，缺失资产会给出错误，不能称为场景通过。

## 仍有边界

仓储是库存补货，非车辆路线或共享仓储容量模型；内置新环境不等于原ReplenishmentEnv。HMF动态分组/高层调度、RODE动态扩角色及缺失源协议等旧欠项保留。交通需自行提供CityFlow引擎、路网、车流，过渡phase必须显式指定。电网来源依赖case33bw配电网结构，不宣称任意拓扑的通用调度；需建筑、天气、设备数据和pandapower兼容依赖。

没有行业预训练模型；先训练或自行提供匹配的应用checkpoint。0.8.0新增代码、安装脚本和运行行为均待用户测试，历史0.7.0证据仅为历史参考。Linux未实机测试，GPU/跨机/真实效果未验。新bounded_ppo为本地同步实现，不默认支持ZMQ。
