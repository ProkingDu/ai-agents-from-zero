import json
from typing import List, TypedDict
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda,RunnableSequence,RunnableBranch,RunnableParallel
from Practice.functions import *


# 分支链

llm = get_deepseek_model(model_name="deepseek-chat")

prompt_writer = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="你是一个作家，请根据主题写一个200字的小故事"),
        HumanMessage(content="{input}"),
    ]
)

prompt_en = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="你是一个翻译家，请将输入的文本翻译为英文。"),
        HumanMessage(content="{input}"),
    ]
)

prompt_fr = ChatPromptTemplate.from_messages(
    [
        ("system","你是一个翻译家，将输入的文本翻译为法语"),
        ("human","{input}"),
    ]
)

prompt_de = ChatPromptTemplate.from_messages(
    [
        ("system","你是一个翻译家，将输入的文本翻译为德语"),
        ("human","{input}"),
    ]
)

prompt_summery = ChatPromptTemplate.from_messages(
    [
        ("system","你是一个精通多国语言的翻译家，我将给你一段含有各个语言的句子，请你将它们整理成每一行中文后面都依次接几行英文/德文/法文的句子。只需要写句子就行了，不需要标注语言。"),
        ("human","{input}"),
    ]
)

chain_1 = prompt_writer | llm | StrOutputParser()

chain_en = prompt_en | llm | StrOutputParser()
chain_fr = prompt_fr | llm | StrOutputParser()
chain_de = prompt_de | llm | StrOutputParser()

chain_summery = prompt_summery | llm

chain_res = RunnableParallel({"en": chain_en, "fr": chain_fr, "de": chain_de})

cn_text = ""
def save_cn_text(x):
    global cn_text
    cn_text = x
    return cn_text
def combine_parallel_result(x:dict):
    x["cn"] = cn_text
    return json.dumps(x)

def build_inputs(x):
    return {"input": x}

chain_final = chain_1 | save_cn_text | build_inputs | chain_res | combine_parallel_result | build_inputs | chain_summery

result = chain_final.invoke({"input": "写一个关于程序员的小故事"})

print(result.content)