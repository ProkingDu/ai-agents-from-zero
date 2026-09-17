import json
from typing import List, TypedDict
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from Practice.functions import *

llm = get_deepseek_model(model_name="deepseek-chat",max_tokens=99999)

# —— 1️⃣ 定义第二段的 json 结构,交给 with_structured_output ——
class Component(TypedDict):
    position: str
    size: str
    content: str
    style: str
    event: str
    description: str

class ComponentDesign(TypedDict):
    components: List[Component]

llm_json = get_deepseek_model(model_name="deepseek-chat",max_tokens=99999).with_structured_output(ComponentDesign)

prompt_req = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业需求分析师,你把用户简单要求分析成复杂的网页设计方案。"),
    ("human", "{input}"),
])
prompt_dom = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业前端组件设计师,根据方案设计网页组件。以 json 输出每个组件的 position/size/content/style/event/description。"),
    ("human", "{input}"),
])
prompt_niuma = ChatPromptTemplate.from_messages([
    ("system", "你是专业前端工程师,根据组件设计写完整 html,直接输出纯 html,不要用代码围栏。"),
    ("human", "{input}"),
])

class SaveHtml(BaseOutputParser):
    def parse(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):            # 顺手剥掉可能的 html 围栏
            text = text.strip("`").lstrip("html").strip()
        with open("html.html", "w", encoding="utf-8") as f:
            f.write(text)
        return text

chain_req_analysis = prompt_req | llm | StrOutputParser()

# 将 ComponentDesign 对象转换为可读文本传给下一个 prompt
def format_components_for_prompt(design) -> str:
    if design is None:
        return "无法生成组件设计"
    components = design.get("components", []) if isinstance(design, dict) else []
    components_text = []
    for i, comp in enumerate(components, 1):
        components_text.append(
            f"组件{i}: 位置={comp.get('position')}, 尺寸={comp.get('size')}, "
            f"内容={comp.get('content')}, 样式={comp.get('style')}, "
            f"事件={comp.get('event')}, 描述={comp.get('description')}"
        )
    return "\n".join(components_text) if components_text else "没有设计任何组件"

chain_dom_designer = prompt_dom | llm_json | RunnableLambda(format_components_for_prompt)

chain_dom_niuma =  prompt_niuma | llm | SaveHtml()

chain_run = chain_req_analysis | chain_dom_designer | chain_dom_niuma

result = chain_run.invoke("帮我设计一个后端程序员的个人主页,要很装逼很炫酷。")
print(result)