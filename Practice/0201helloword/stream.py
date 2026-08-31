from dotenv import load_dotenv
import os 
from langchain.chat_models import init_chat_model
import json
import time
from datetime import datetime
load_dotenv()

ds = init_chat_model(
        model="qwen3.6-flash",
        model_provider="openai",
        base_url = os.getenv("baseURL_ali"),
        api_key=os.getenv("QWEN_API_KEY"),
        temperature=0.5
    )

a = time.perf_counter()
res = ds.stream("给我讲一个天大的冷笑话")

for chunk in res:
    if chunk.content:
        print(f"首字输出：{time.perf_counter() - a}")
        exit()
    # print(f"总耗时：{time.perf_counter() - a}")