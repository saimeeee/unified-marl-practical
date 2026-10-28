# 当前发布状态：0.8.2已发布

[v0.8.2-engineering.1](https://github.com/saimeeee/unified-marl/releases/tag/v0.8.2-engineering.1)已正式发布，Actions成功；9份附件的远端摘要及实际下载SHA256全部通过。源码提交0c5f2d34bf977a2e75ca5635b16ba5a64f20dbb1，仓库public未改动。具体命令和回执见PUBLICATION_STATUS_082.md。下面早期“private/待连接/待发布”等内容为历史准备记录，以本段和实际回执为准。

---

最新发布：0.8.0工程版已发布 https://github.com/saimeeee/unified-marl/releases/tag/v0.8.0-engineering.1，仓库保持private；9附件大小/SHA-256逐项一致，Actions 39569403057通过。运行时源码与wheel逐字节一致，具体回执见docs/releases/v0.8.0-engineering.1.verification.json。新能力未跑训练或环境实验。

# GitHub 内部分发记录

最新发布状态：已发布 [v0.7.0-internal.1](https://github.com/saimeeee/unified-marl/releases/tag/v0.7.0-internal.1)，仓库仍为 private。9 个附件的远端大小和 SHA-256 全部核对一致；[发布任务](https://github.com/saimeeee/unified-marl/actions/runs/39558172643) 通过。发布提交 a92d0bea9a6acb44e5576ab81d43ac152b6d2c3e，文档提交 21dcb4a8014351c4a18a92b0d0e10de1be67932c。下文准备/发布中的文字保留为过程记录；机器回执在 docs/releases/v0.7.0-internal.1.verification.json。

目标仓库：https://github.com/saimeeee/unified-marl 。保持私有；只更新该仓库，不创建公开仓库或改变访问范围。

当前状态：**发布材料与自动发布流程已准备，Release 是否发布以 Actions 和 Releases 页面实际结果为准。** GitHub 插件连接已恢复，仓库已核对为 private。插件没有直接上传大附件的接口，因此通过仓库内的受限发布流程按固定依赖、哈希和 ZIP 元数据重建安装包，再创建私有仓库 Release；不会运行 Linux 程序或测试。此前源码提交 0f0320d649f2f3106db910297ba84e5a5eeb3f8a 为历史上传记录。

## 计划内容

- main：最新 README、中文 USER_GUIDE.md、内部分发脚本/构建器、报告及进度；保留原 687 文件的来源记录，不移除未通过项。
- Release tag：`v0.7.0-internal.1`，发布前检查该 tag 是否已经存在、main 是否有其他修改，以及仓库仍为私有。
- Release 正文：`docs/releases/v0.7.0-internal.1.md`。
- Release 附件：Windows ZIP、Linux ZIP、核心 wheel、核心 sdist、中文安装说明、SHA256SUMS.txt、交付清单及两份验证摘要，共 9 个。

二进制安装包放 Release，代码仓库保留源码、配置、脚本和说明；不将数百 MB 的 ZIP 或依赖 wheelhouse 填入 Git 历史。真实数据、训练模型、缓存、环境和个人凭据不上传。

## 本地可复核材料

`.work/github-internal-release/assets.json` 保存各附件的实际绝对路径、大小和 SHA-256；assets/ 下是发布专用的中文说明和完整校验清单。source-changes.json 保存相对首次上传清单的新增/变更文件以及哈希。准备状态不会冒充 GitHub 远端证据。

GitHub 连接恢复后，先读取远端状态，保留其他已有修改；更新源码/文档并关联 Release 到实际交付提交。不要强制覆盖不同的远端提交或已存在的同名附件。上传后核对附件名、大小/摘要和仓库 private 状态，再保存发布回执并把当前状态更新为已发布。

## 自动发布的安全边界

`.github/workflows/publish-internal.yml` 在 Windows runner 上运行 `tools/publish_internal_github.py`；仅下载精确版本/哈希的 wheel 和重建文件，不安装 Linux 包、不执行 Linux 脚本、不下载 hf-xet。原核心 wheel、源码包及仅有源包上游的 gym wheel 随发布输入保留，其他大依赖从官方 PyPI/PyTorch 索引按哈希取回。ZIP 大小及 SHA-256 必须与本地已交付产物逐字节一致，否则停止，不发布。

流程使用 GitHub 为当前仓库任务提供的短期令牌，仅声明 contents: write，用于创建本仓库 Release；先核对仓库为 private，默认草稿，全部 9 个附件大小与 GitHub 摘要核对后才发布。已有不同附件或同名已发布版本会明确拒绝覆盖。不会创建用户 API key、改变权限/可见性或上传原始数据/模型/环境。

## 0.8.2正式发行准备

目标tag v0.8.2-engineering.1；新增publish-hotfix.yml及经测试的附件发布器。9个附件按固定哈希准备，保留用户测过的补丁和wheel。仓库当前public（匿名API核对），本次不改可见性。当前连接缺失，只有实际源码上传、Actions完成及远端附件SHA核对后才标为发布。
