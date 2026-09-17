from dotenv import load_dotenv
import os 
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model="qwen3.7-flash",
    model_provider="openai",
    base_url = os.getenv("baseURL_ali"),
    api_key = os.getenv("aliQwen-api")
)

print(model)

res = model.invoke("给我讲个笑话吧。")

print(res.content)

# 存在多个模型
model2 = init_chat_model(
    model="wan2.7-image-pro",
    model_provider="openai",
    base_url = os.getenv("baseURL_ali"),
    api_key = os.getenv("aliQwen-api")
)

print("============================\r\n")
print(model2.invoke("帮我生成一幅山水画图片"))