# 本次提取验证

验证日期：2026-10-18，macOS arm64 / Python 3.12.13。

- 新项目 `.venv` 已独立创建，并用 `uv sync --frozen --group dev` 安装锁定依赖；没有 editable 安装到旧服务环境。
- 原 uv.lock 字节保持一致，`uv lock --check --offline` 通过。
- 新环境导入路径指向本项目 src；`cmr --help` 和使用零来源示例的 `doctor` 通过。
- 完整合成回归：346 passed in 18.87s。使用事先已有的公开 Python 语法缓存（CMR_GRAMMAR_CACHE）及独立临时目录；不使用真实业务源、模型或 daemon。
- 空语法缓存时观察到 5 个 AST 前置条件失败、341 个通过；README 已明确准备命令。没有跳过失败来宣称完整通过。
- `uv build --offline` 成功构建 sdist 和 wheel；sdist 包含 integrations，两个产物都不包含 .venv 或业务运行数据。没有进行 wheel 安装后的模型端到端验收。
- 72 个 Python 文件 AST 解析通过；暂存区空白检查通过。内容扫描未发现真实凭据或固定用户名绝对路径；测试中一枚密钥格式字符串是显式合成假值。
- 98 个受保护文件的 SHA256 与提取前一致，包含既有配置、资料策略、发布/MCP 源码及业务项目 README/USAGE。

真实模型评分、Codex 会话外发、现有服务重新部署、运行路径迁移均未执行。本次结果是独立源码仓库与开发环境的验证，不替代原发布包的 seal 或独立环境锁。
