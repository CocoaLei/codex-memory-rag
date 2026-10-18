# MCP 源码与现有接入

候选适配器的三个层次：

1. `candidate_adapter.py` 验证策略和候选资格，仅投影允许字段。精确授权包含 source_id、相对路径、file_sha256、chunk_id、line_start、line_end；修改或撤销策略会使旧缓存失效。
2. `candidate_transport.py` 将四个工具连接到守卫 CLI，限制并发、输出、截止和进程清理。
3. `candidate_entry.py` 校验封存包、Python 启动标志、运行身份及独立环境锁后才建立 STDIO 服务。缺少任一绑定时默认拒绝。

适配器固定使用 `example-project` 和 `example-memory` / `example-docs` / `example-code`。这些是通用示例标识；运行配置、批准策略和 CLI 守卫必须一致，不会因重命名而自动批准读取或外发资料。

`capture_environment.py` 要求预先准备的专用部署输入，不是一键安装命令。运行包要求 control、src、输入锁和环境记录匹配。保留的版本化绑定字段（如 `r4_input_hash`）属于文件格式合同，不表示本仓库已完成部署验收。

源码修改会改变封存指纹；不得复用旧 seal 或环境锁，也不得直接将现有 MCP 配置改指本目录。封存、部署、迁移和真实资料验收均需单独授权。
