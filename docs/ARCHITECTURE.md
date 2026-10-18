# 架构与能力边界

登记的本地源 → 安全扫描和状态过滤 → Markdown/AST 分块 → 768D embedding 与 LanceDB/FTS → 候选合并及可选重排 → 终检当前来源 → CLI → 策略投影 MCP → Codex。

daemon 是唯一索引 writer，多个客户端通过私有 Unix domain socket 复用模型。检索与回读需要 project/source/worktree 身份；原文返回前核验文件 hash、行范围和来源状态，漂移返回错误。候选模式提供相关片段，不判断查询中的每一项要求是否由片段完整证明。

长期记忆写入继续由外部 Basic Memory / Obsidian 工作流负责。本系统检索已登记记忆，不替代原系统的写入、回读或审计合同。

合成测试覆盖实现合同，不证明真实语料的答案充分性；模型质量和实际接入需单独验证。
