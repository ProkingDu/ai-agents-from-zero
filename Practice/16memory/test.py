import json
from typing import List, TypedDict
from webbrowser import get
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda,RunnableSequence,RunnableBranch,RunnableParallel
from Practice.functions import *


# 先看没有记忆的时候会发生什么
logger = Logger()
llm = get_deepseek_model(model_name="deepseek-chat")

prompt = PromptTemplate.from_template("""
你是一个助手，请根据用户的问题给出回答
用户的问题是：{input}
""")

chain = prompt | llm

# while True:
#     input_user = input("请输入问题：")
#     res = chain.invoke({"input":input_user})
#     logger.log(f"用户提问：{input_user}")
#     logger.log(f"助手回答：{res.content}")
#     logger.log("-"*100)