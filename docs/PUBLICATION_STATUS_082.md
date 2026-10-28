# 0.8.2 正式发布结果

状态：**PUBLISHED**。已发布到现有saimeeee/unified-marl仓库，保持public可见性。

- [Release及9份下载附件](https://github.com/saimeeee/unified-marl/releases/tag/v0.8.2-engineering.1)
- [成功的发布工作流](https://github.com/saimeeee/unified-marl/actions/runs/39621848630)
- 发行源码提交：`0c5f2d34bf977a2e75ca5635b16ba5a64f20dbb1`。
- Release ID：419693145；draft=false；Actions结论success。

第三轮已测wheel和hotfix ZIP字节保持不变。9份远端附件的名称、大小、GitHub SHA256摘要全部与本地发布计划匹配；另通过公开下载链接实际下载全部9份附件，再次核对大小及SHA256，全部通过。回执在`docs/evidence/release-082/github-publication.json`和`download-verification.json`，没有保存任何登录凭据。

本地运行源码/资源378份与wheel及源码包匹配，398项迁移记录全部保留。发行文件在`dist/releases/v0.8.2-engineering.1`；`dist/release-handoff/0.8.2`是发布前交接快照，其中“未发布”说明已被本记录更新，不需要再用该快照重新发布。

实际操作记录：

1. `.venv-release/Scripts/python.exe -X utf8 -I tools/prepare_hotfix_release.py`：退出0，生成9份附件，仅重建源码包。
2. `tools/publish_hotfix_github.py --verify-only`：本机与独立恢复checkout均退出0，9份附件通过。
3. `.work/verify-release-final.py`：退出0，校验归档、哈希、源码资源及迁移清单。
4. 首次非交互Git推送因未登录退出1；用户确认插件连接后，使用系统Git Credential Manager的浏览器授权成功，未读取或输出token。
5. `git -c credential.interactive=never push origin HEAD:refs/heads/main`：退出0，5741ec0→0c5f2d3，触发publish-hotfix.yml。
6. `.work/check-hotfix-remote.py`：退出0，工作流completed/success、Release已发布、9份附件摘要通过、仓库public。
7. `.work/download-verify-hotfix.py`：退出0，实际匿名下载9份附件，逐份大小与SHA256通过。

新工作流只在新发布输入变化时执行。已有发布附件不一致会拒绝覆盖；本次文档回执更新不改发行附件或标签。核心依赖平台包仍复用0.8.0，不能把小补丁说成完整离线平台。完整Linux/真实环境实验未在本机重跑，第三轮用户证据与剩余限制见VALIDATION_REPORT_082.md。
