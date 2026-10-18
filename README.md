# Codex Memory RAG

本地记忆、文档与代码候选检索项目。使用 `google/embeddinggemma-2` 的原生 768D 文本向量、LanceDB 向量与全文索引，以及可选的本地重排；通过单 daemon、Unix domain socket 和只读 MCP 客户端共享检索服务。

这是从已运行系统提取的独立开发仓库。内核来自已部署的 r4，候选 MCP 适配器来自最终 r3。现有安装继续使用原发布目录；创建本仓库没有迁移或重启服务。

候选检索已经用于实际项目聊天，但候选相关性不等于证据充分性。结果保留 `retrieval_role=related_candidate`、`support_status=not_checked` 和“完整支持未检查”。原强证据目标尚未全部达到；不要把此仓库描述为已通过全部质量门槛的自动问答系统，也不要把历史单机测量当成任意机器的性能保证。

## 代码结构

| 路径 | 内容 |
| --- | --- |
| `src/codex_memory_rag/` | 配置、来源安全读取、分块、768D 编码、索引、检索、重排、原文回读、daemon、CLI、兼容 MCP |
| `integrations/candidate_mcp/` | 最终 r3 的精确授权、候选缓存、输出投影、进程收尾与四工具 MCP |
| `integrations/deployment/control/` | 已部署 r4 的守卫 CLI/daemon、环境核验与部署事务源码 |
| `tests/` | 内核与候选模式的合成单元/集成测试 |
| `docs/` | 架构、开发和提取边界；源码来源指纹 |

仓库不包含业务资料、模型权重、索引、凭据、真实配置、资料外发策略、机器环境锁或操作日志。旧失败记录与冻结包继续保留在原位置；原始历史文档没有批量上传到 GitHub。

## 开发

需要 Python 3.12，依赖以 `uv.lock` 为准。优先使用 macOS Apple Silicon；MPS 能力需要实机单独确认。

```sh
uv sync --frozen --group dev
# 完整 AST 回归需要先显式准备公开 Python 语法资源（不是模型）：
uv run --frozen python -c 'from pathlib import Path; from tree_sitter_language_pack import configure, PackConfig, download; configure(PackConfig(cache_dir=str(Path.home() / ".cache/codex-memory-rag/grammars"))); download(["python"])'
uv run --frozen python -m pytest -q
uv run --frozen cmr --help
uv run --frozen cmr --config config.example.toml doctor
```

安装开发依赖会从包仓库下载软件。默认测试使用合成资料，不下载模型，也不读取真实业务源。完整测试依赖事先准备的公开 Python 语法资源；上面的显式命令负责准备，测试自身不下载。树形解析器缺少本地语法资源时，运行检索会使用降级分块，五个依赖 AST 的合成回归会失败；可以用 `CMR_GRAMMAR_CACHE` 指定已准备好的缓存。

## 独立本地实例

复制 `config.example.toml` 到仓库外，设置独立绝对 runtime 路径，再显式登记允许读取的源。示例默认没有源，候选模式默认关闭。`allow_cloud_context` 是检索源准入字段，不代表用户自动批准把该源交给任意云服务。

```sh
cmr --config /absolute/private/config.toml source audit
# 显式准备公开模型；此步骤会下载模型：
cmr --config /absolute/private/config.toml model prepare
cmr --config /absolute/private/config.toml serve
# 在另一个终端：
cmr --config /absolute/private/config.toml index build --project my-project
cmr --config /absolute/private/config.toml search --project my-project --query "配置如何更新"
cmr --config /absolute/private/config.toml context --project my-project --chunk-id CHUNK_ID --max-chars 4000
```

需要依据本地数据与已批准合同填写检索阈值；示例不是可直接部署的已校准配置。重排默认关闭，现有 `BAAI/bge-reranker-v2-m3` 模型文件与资源锁必须由使用者在仓库外显式准备。不要复制另一台机器的环境锁冒充本机核验。

## Codex MCP

日常候选模式使用 `integrations/candidate_mcp/candidate_entry.py`，工具只有 `candidate_status`、`candidate_search_memory`、`candidate_search_code`、`candidate_get_context`。搜索只返回授权候选元数据，原文需在同一连接搜索后回读；授权绑定 chunk、文件 hash、行范围及部署身份，策略变化会清空旧候选。

此适配器保留原试点的 `composing-agent` / `ca-memory` / `ca-docs` / `ca-code` 标识合同。这些标识不是所含业务资料。入口要求专用封存包、本机独立环境锁与明确批准的资料策略；这些文件有意不随源码发布，因此直接从新检出目录运行会拒绝启动。现有已批准接入仍运行原封存版本。详见 [MCP 接入边界](docs/MCP.md)。

`codex_memory_rag.mcp_bridge` 是兼容入口，候选模式会明确拒绝它；它不具备最终 r3 的精确片段外发策略。不要为了连接成功绕开候选守卫。旧 `install` / `rollback` 命令保留兼容性，但不是本仓库推荐的日常接入或真实迁移步骤。

后续开发从此仓库开展；部署到现有实例、迁移实际路径、扩大资料外发范围仍需对应授权。参见 [架构](docs/ARCHITECTURE.md)、[开发说明](docs/DEVELOPMENT.md) 和 [提取说明](docs/EXTRACTION.md)。
