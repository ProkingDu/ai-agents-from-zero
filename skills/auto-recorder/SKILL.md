---
name: auto-recorder
description: 仅在用户明确手动调用本技能时使用。手动调用时，如果调用内容携带了问题，则将该问题与本次回答记录为 QA；如果未携带问题，则将上一轮问题与其回答记录为 QA，并追加写入指定目录下的 qa-log.md。不要在普通回答结束后自动触发本技能。
---

# Auto-Recorder 问答记录器

## 触发方式

本技能只支持**手动触发**，不得根据普通对话内容自动启用，也不得在每次回答结束后自行检查或写入日志。

当用户明确调用 `auto-recorder`（例如 `/auto-recorder`）时，按照以下规则确定要记录的问题：

- **调用时携带问题**：将用户在本次调用中提供的问题，与本次调用产生的回答记录为一条 QA。
- **调用时未携带问题**：将最近一轮对话中的用户问题与对应回答记录为一条 QA。优先使用紧邻本次调用之前的上一轮完整问答。
- 如果无法找到可对应的问题或回答，不要虚构内容；应向用户说明缺少可记录的问答，并且不要写入日志。

## 记录文件

- **文件位置**：`/Users/Zhuanz1/Document/Project/Knowledge/ai-agents-from-zero/qa-log.md`。
- 文件不存在时创建；文件已存在时直接追加到末尾，不能覆盖或删除历史记录。
- 只记录用户指定的那一组问题和回答，不记录本技能的判断过程、工具调用过程或内部信息。
- 同一次手动调用只追加一条 QA，不得重复写入。

## Markdown 格式规范

每条 QA 必须严格按照以下格式写入：

```markdown
### Q: <问题标题>

<回答正文>
```

**格式要求：**
- 每个问题使用三级标题 `### Q: ...`，这是唯一的标题层级。
- 回答正文紧跟其后，可以包含段落、列表、表格和代码块。
- 不允许使用四级或更深层级标题（如 `####`）。
- 不允许使用日期作为标题。
- 不允许添加一级标题或其他分类结构。

**示例：**

```markdown
### Q: 如何用 Python 创建一个 LangGraph Agent？

可以使用 LangChain 的 `create_react_agent` 函数快速创建一个基于 LLM 的 Agent。以下是基本用法：

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
agent = create_react_agent(llm=llm, tools=[])
executor = AgentExecutor(agent=agent, tools=[], verbose=True)

result = executor.invoke({"input": "你好，请介绍一下自己"})
print(result["output"])
```

关键步骤：
1. 初始化 LLM 模型
2. 定义可用的工具（tools）
3. 创建 Agent 实例
4. 通过 AgentExecutor 执行任务
```

## 写入流程

1. 确认要记录的问题和回答内容。
2. 打开目标文件 `/Users/Zhuanz1/Document/Project/Knowledge/ai-agents-from-zero/qa-log.md`。
3. **不要读取文件的任何内容**，直接追加到文件末尾。
4. 先写入一个空行，然后写入格式化的 QA 条目。
5. 确保每条 QA 之间有适当的空行分隔。
6. 写入完成后无需验证文件内容。

## 注意事项

- 回答正文保持原样，包括解释、步骤、代码、命令、配置、表格和链接。
- 不要把“本次回答已记录”等操作说明写入 QA 的回答正文。
- 如果回答中有四级或更深层级标题，将其降级为加粗文本或列表项。
- 不要自动记录未被用户手动指定的其他问题或回答。
