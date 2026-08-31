from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


# 获取环境变量
load_dotenv()

llm = ChatOpenAI(
    base_url=os.getenv("baseURL_ali"),
    model="deepseek-v3.2",
    api_key=os.getenv("aliQwen-api")
)

res = llm.invoke("你好我是丁真")
print(res)
print("=============\n")
print(res.content)