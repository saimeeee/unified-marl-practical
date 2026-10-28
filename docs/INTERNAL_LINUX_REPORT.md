# Linux x86_64 内部分发包报告

产物为 [unified-marl-0.7.0-linux-x86_64-cpu-py312.zip](../dist/internal/unified-marl-0.7.0-linux-x86_64-cpu-py312.zip)，绝对路径：`G:\AI\Codex\多智能体代码整合\unified-marl\dist\internal\unified-marl-0.7.0-linux-x86_64-cpu-py312.zip`。文件大小为 **281,863,343 字节**，SHA-256 为 **`3321f210acf859a99f98ad1afb87f6c04d8e43dc8816c5b46d3808cd93e1be3b`**。

包面向 **CPython 3.12、Linux x86_64、CPU、glibc 2.28 或更新版本**。它包含核心 wheel 和 34 个按项目已验环境版本选定的本地依赖 wheel，`requirements-local-linux-cpu.txt` 为这些发行物记录了精确版本和 SHA-256。包名不表示可完全离线安装：`huggingface-hub 0.36.2` 的 Linux x86_64 wheel 元数据要求 `hf-xet>=1.1.3,<2.0.0`，该单个补充依赖在 `requirements-network.txt` 中声明，接收方安装时通过公开 PyPI 联网获取。它未打入 ZIP，也没有预先锁定具体版本或哈希。安装器在安装时会显示联网提示；本地其他依赖从包内 wheelhouse 安装，安装后执行 `pip check` 和核心/已验可选模块导入检查。

`hf-xet` wheel 下载请求曾被工具明确拒绝，回执为 `Rejected("rejected by user")`；这是用户拒绝，不是自动审查拒绝。按后续指示，没有重试或通过其他途径获取该 wheel，也没有在本机尝试 Linux 功能安装或运行测试。

静态打包检查通过：根目录 manifest 验证了 **65 个发行文件**的大小与 SHA-256；ZIP 有 **66 个条目**（含 manifest），CRC 检查通过，所有成员均位于唯一预期根目录下且没有绝对路径或 `..` 路径；wheelhouse 未发现 Windows/macOS wheel。打包器还从目标 Linux wheel 元数据检查了本地依赖闭包，并将唯一缺失的 Linux-only `hf-xet` 单独列为联网补充项。Python 打包脚本及安装/运行脚本内嵌的 Python 片段通过语法检查。由于当前电脑没有可用 Linux 环境且用户明确要求无需实机测试，Linux 安装、`pip check`、导入和可选 QMIX 冒烟测试均留给接收方执行；本报告不把静态检查表述为 Linux 运行验收。

真实数据、checkpoint、预训练权重、仿真器和地图均不随包分发。算法限制及许可说明保留在包内的支持矩阵、已知问题、许可证、NOTICE 和来源文件中。此产物不声明任意 Linux、GPU/CUDA 或真实仿真器支持。
