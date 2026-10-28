# 许可与来源核查

本次为用户工作区构建安装产物，没有向包仓库、网站或外部服务发布。

| 来源 | 实际发现 | 保留位置 | 待确认事项 |
|---|---|---|---|
| 提供的 PyMARL | 根 LICENSE 为 Apache-2.0 | LICENSE、licenses/PyMARL-Apache-2.0.txt | 本项目适配改动与来源哈希另有记录 |
| 易才润 SADMA 开源版和 ZMQ 公开版 | 两处 LICENSE 均为 MIT，Copyright (c) 2024 XJTU | licenses/SADMA-Yi-MIT.txt、NOTICE | 不能从开源目录名推断，也不能扩大至另一个闭源目录 |
| 王思哲 RGHMF/HMF/SADMA | 未发现覆盖交付算法的总体授权文本；保留原注释及外部来源链接 | NOTICE、PROVENANCE.json | 交付权利及其中第三方片段适用条款待确认 |
| 寇谦 ICQ-MA/old_src/MAPT 奖励闭包 | 未发现覆盖这些算法/模型文件的总体授权文本 | NOTICE、PROVENANCE.json | 交付及嵌入代码的权利待确认；未打入 MAPT 无关仿真资产 |
| 易才润闭源配置 | 没有据同名、AST 相同或开源版 MIT 推定授权 | NOTICE | 闭源交付配置条款待确认 |

先前报告对易开源子集“未找到许可证”的判断由本阶段实际文件核查修正；历史证据不删除。项目元数据使用 Apache-2.0、MIT 与明确的 LicenseRef 标识，未将权利不明部分改标 Apache/MIT，也未新增第三方授权承诺。

wheel 带完整许可文本、NOTICE 和包资源 PROVENANCE.json；源码包另带逐文件迁移追踪、规格和矩阵。追踪中的 source_sha256 对照原文件，target_sha256 对照迁移后文件；不要求两者相同，适配和验证由对应阶段报告解释。
