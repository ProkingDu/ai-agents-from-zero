import asyncio
import os 
from Practice.functions import *
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
from langchain_core.prompts import PromptTemplate

llm = get_deepseek_model()


msgs = [
    SystemMessage(content="你是一名专业的短视频创作者，擅长于幽默风趣的短视频文案撰写"),
    HumanMessage(content="请根据我提供给你的内容编写一条短视频文案"),
    HumanMessage(content="欢迎收看00后程序员下班后自学Agent，今天打卡第N天")
]

# res = llm.invoke(msgs)
# print(res.content)

msg = PromptTemplate(template="你是一名专业的短视频创作者，擅长于幽默风趣的短视频文案撰写\n请根据我提供给你的内容编写一条短视频文案\n{input}",input_variables=["input"])
str = msg.format(input="欢迎收看00后程序员下班后自学Agent，今天打卡第N天")
# res = llm.invoke(str)
# print(res.content)

template = PromptTemplate.from_template("给你一段文本，请你帮我分析这段文本的情绪，你的回答只能用2-3个字。内容为：{cont}")
msg = template.format(cont="哈哈哈哈哈")
# 人工智障
# res = llm.invoke(msg)
# print(res.content)


# 两小二辩日

a_messages = [
    SystemMessage(content="孔子东游，见两小儿辩日，你是两小儿之一的小儿A，你将参与这个辩论中，并且你不能一次说太多，你必须结合之前辩论的内容思考自己要说什么。注意：禁止直接引用原文内容，发挥你的想象力自己去辩论！"),
    HumanMessage(content="太阳刚出来的时候，离人很远，为什么看起来会很大？"),
]

b_messages = [
    SystemMessage(content="孔子东游，见两小儿辩日，你是两小儿之一的小儿A，你将参与这个辩论中，并且你不能一次说太多，你必须结合之前辩论的内容思考自己要说什么。注意：禁止直接引用原文内容，发挥你的想象力自己去辩论！"),
]

a_llm = get_deepseek_model()
b_llm = get_qwen_model()

for i in range(1,3):
    a_res = a_llm.invoke(a_messages)
    a_messages.append(AIMessage(content=a_res.content))
    b_messages.append(HumanMessage(content=a_res.content))
    b_res = b_llm.invoke(b_messages)
    b_messages.append(AIMessage(content=b_res.content))
    a_messages.append(HumanMessage(content=b_res.content))

    # print("Messages:",a_messages,b_messages)
    # print(f"第{i}轮辩论结束")
    # print(f"小儿A说：{a_res.content}")
    # print(f"小儿B说：{b_res.content}")
    # print("--------------------------------")