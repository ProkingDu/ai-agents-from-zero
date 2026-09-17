import asyncio
import os 
from Practice.functions import *
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
from langchain_core.prompts import PromptTemplate
import time



llm = get_deepseek_model()

msgs = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
]

cors = llm.ainvoke(msgs)
# print(123)

# 此时直接执行携程对象，还是同步的
# asyncio.run(cors)
# 异步,并行
async def call():
    cors1 = llm.ainvoke("你是老大")
    cors2 = llm.ainvoke("你是老二")
    res = await asyncio.gather(cors1, cors2)
    print("老大:", res[0].content)
    print("老二:", res[1].content)

asyncio.run(call())
exit()
# 异步，串行

async def call2():
    cors1 = llm.ainvoke("你是老大")
    cors2 = llm.ainvoke("你是老二")
    res1 =  await asyncio.create_task(cors1)
    res2 =  await asyncio.create_task(cors2)
    print("老大:", res1.content)
    print("老二:", res2.content)

cors1 = llm.ainvoke("你是老大")
cors2 = llm.ainvoke("你是老二")
a = asyncio.gather(cors1, cors2)
print(a)
asyncio.run(a)
