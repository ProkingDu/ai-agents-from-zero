from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any, TypeVar
from Practice.functions import *


class MyOutputParser(BaseOutputParser[list[str]]):
    def parse(self,cont:str) -> list[str]:
        ls = cont.split("\n")
        for item in ls:
            if item.strip() == "" or len(item) < 10:
                print("丢弃一个空的或者小于10的输出")
                ls.remove(item)
        return ls
    def get_format_instructions(self) -> str:
        return "请严格按照用分隔符来分割每一句话。"

llm = get_deepseek_model()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions and help with tasks."),
    ("system","{format_instructions}"),
    ("human", "{input}"),
])
parser = MyOutputParser()

# prompt = prompt.invoke({"input":"随便输出几句话攒劲的话用换行符隔开","format_instructions":parser.get_format_instructions()})
prompts = prompt.batch([{"input":"随便输出几句话攒劲的话用换行符隔开","format_instructions":parser.get_format_instructions()},{"input":"随便输出几句话悲伤的话用换行符隔开","format_instructions":parser.get_format_instructions()}])

response = llm.batch(prompts)

outputs = parser.batch(response)
print(outputs)