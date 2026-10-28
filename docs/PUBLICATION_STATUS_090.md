# 0.9.0.dev1 实用首版独立仓库

用户于2026-10-28明确要求新建仓库上传，目标为 [saimeeee/unified-marl-practical](https://github.com/saimeeee/unified-marl-practical)。仓库已创建为 **private**，源码和 [v0.9.0.dev1 预发布](https://github.com/saimeeee/unified-marl-practical/releases/tag/v0.9.0.dev1) 已上传完成；原 saimeeee/unified-marl 未改动。

本次只上传源码、配置、中文说明、示例、小型场景、测试和校验摘要。虚拟环境、模型、完整运行目录、原始交付、视频与历史大型依赖安装包不纳入新仓库。历史文档和发布工具保留来源上下文；其中的原仓库地址及0.7/0.8版本不属于本仓库当前分发入口。旧仓库专用发布工作流不复制到新仓库。

Release为已发布的 `v0.9.0.dev1` 预发布。核心wheel保留本地验收时字节；其内嵌README为当时的文档快照，最新说明以新仓库README及Release附带指南为准。wheel不包含Python、PyTorch等依赖、CityFlow或其他外部模拟器。无本轮长期收敛、GPU或生产部署结论。

历史319项回归发生在包含原始交付的开发工作区。新仓库的首版专项验证单独记录，不能将历史总数当作独立clone的全部测试结果。

## 实际远端回执

- 源码与标签提交：`8ed0a3733e0ead0f49b0aebc62e460c46944ac92`，首次上传563个文件；后续文档提交只补充发布回执。
- 发布时间：2026-10-28 07:40:25 UTC（东京16:40:25），Release ID `426375545`。
- 7份附件：核心wheel、源码ZIP、使用指南、算法接入指南、验收报告、验证摘要、SHA256SUMS。远端大小和SHA-256逐项一致；另已登录下载全部7份，再算SHA-256全部通过。
- wheel：`unified_marl-0.9.0.dev1-py3-none-any.whl`；SHA-256 `2c564a1df7fd0fc24112dfd6ef0d76b9132266956ba6823be17685231caa3f5a`。
- Windows独立源码目录：98 passed、1 skipped（缺少仓库外原始交付的对照测试）。
- [GitHub Actions](https://github.com/saimeeee/unified-marl-practical/actions/runs/40651414339)：Linux/Python 3.12从依赖安装开始运行四份首版专项测试，98 passed、1 skipped，30.06秒；Job `122692110688` 成功。CI没有运行原生CityFlow或长期训练。
- 机器回执：[Release附件校验](releases/v0.9.0.dev1.verification.json)、[CI回执](evidence/practical-v1/ci-verification.json)、[本地验收摘要](evidence/practical-v1/validation-summary.json)。

源码ZIP与标签保留发布时快照，其中准备中的状态说明由main分支本回执补充；不重写已验收附件或标签。原始运行证据继续保存在开发工作区，仓库只含不带模型和本地绝对路径的摘要。
