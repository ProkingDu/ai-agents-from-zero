"""
第 16 章最小 demo：读历史 -> 拼提示 -> 调模型 -> 写回历史
内存版：进程一停，历史就没了
"""

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

# 1) 模型：本地 Ollama（换成 init_chat_model + 云端模型也一样）
model = init_chat_model(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    model_provider="deepseek",
    api_key=os.getenv("deepseek-api"),
    temperature=0.7,
)

# 2) 模板：系统设定 + 历史消息占位 + 当前问题
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个友好的学习助手，回答控制在 50 字以内。"),
        MessagesPlaceholder("history"),  # 历史整块塞进这里
        ("human", "{question}"),
    ]
)

# 3) LCEL 组链：只负责“这一轮怎么答”
chain = prompt | model

# 4) 历史存哪里：按 session_id 一人一份
store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 5) 给整条链包一层历史管理：何时读、何时写
chat = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


def ask(session_id: str, question: str) -> str:
    reply = chat.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},  # 告诉它读哪份历史
    )
    return reply.content


if __name__ == "__main__":
    print("A1:", ask("user-001", "我叫张三，正在学 LangChain。"))
    print("A2:", ask("user-001", "我叫什么？在学什么？"))  # 能答出来
    print("B1:", ask("user-002", "我叫什么？"))  # 换 session_id → 不知道

    # 顺手验证“记忆说到底就是一个消息列表”
    print("\nuser-001 的历史条数:", len(store["user-001"].messages))
    for m in store["user-001"].messages:
        print(" -", m.type, "|", m.content[:30])