# 开发与验证

Python 3.12，使用仓库 uv.lock 锁定的依赖。`uv sync --frozen --group dev` 创建本项目自己的环境；不要对正在运行服务的虚拟环境执行 editable 安装或修改其依赖。

```sh
python -m pytest -q
python -m codex_memory_rag.cli --help
python -m codex_memory_rag.cli --config config.example.toml doctor
git diff --check
```

完整回归需要公开 Python 语法缓存，先按 README 的显式命令准备，或设置 `CMR_GRAMMAR_CACHE` 指向已有缓存。没有缓存时，运行时降级分块仍可用，但 AST 功能测试不满足前置条件；不要通过跳过这些测试来冒充完整验证。

测试范围包括合成文件、索引、UDS、来源漂移、候选标记、CLI 回读、MCP 投影、策略跨调用失效、取消清理和合成 CAS 恢复。候选适配器中的四工具和原试点项目标识合同保持不变。原环境专属的绑定测试依赖真实锁，没有导入默认测试；本仓库新增缺失锁默认拒绝的独立测试。

默认路径已移除固定用户名：配置默认 `~/.config/codex-memory-rag/config.toml`，独立 runtime 默认 `~/.local/share/codex-memory-rag`，语法缓存默认 `~/.cache/codex-memory-rag/grammars`。实际使用始终建议显式传入 `--config`。

模型下载、真实语料评测、实际 Codex 会话、服务接管和回滚演练不由默认测试触发。提取期间没有追加这些操作；测试通过仅证明合成回归的观察结果。
