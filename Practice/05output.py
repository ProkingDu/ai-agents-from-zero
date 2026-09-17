from Practice.functions import *
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser,PydanticOutputParser
from typing import List,Dict,Any,Optional,TypedDict,Annotated
from pydantic import BaseModel,Field


llm = get_deepseek_model(model_name="deepseek-chat")

parser = JsonOutputParser()

res = llm.invoke("请随机生成几个包含name,phone,sex,age的用户json数据，以标准JSON格输出")

cont = parser.invoke(res)

print(type(cont))




# class Comment(TypedDict):
#     id:int
#     content:str
#     author:str
#     created_at : str
#     avator: str

# # llm = get_qwen_model()
# structured_llm = llm.with_structured_output(Comment)

# res = structured_llm.invoke("请随机生成十个有趣的关于一篇Agent概述的评论。")

# print(res)

# exit()


class Comment(TypedDict):
    id : Annotated[int,"评论ID"]
    content : Annotated[str,"评论内容"]
    author : Annotated[str,"评论作者"]

class CommentList(TypedDict):
    comments : Annotated[List[Comment],"评论列表"]




# llm = llm.with_structured_output(CommentList)

# res = llm.invoke("请随机生成5条关于Agent概述的评论")

# # 为什么只生成了一个？因为要求输出是一个Comment类型的对象，这个类型只能装一个。所以需要用CommentList包含多个Comment。
# print(res)

