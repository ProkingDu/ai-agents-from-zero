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
        base_url = os.getenv("BASE_URL_QWEN"),
        api_key=os.getenv("aliQwen-api"),
        temperature=1.9,
        max_tokens=400
    )

res = ds.invoke("Hello, world!")
print(res)