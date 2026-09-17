from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import datetime
import os
load_dotenv()

def get_qwen_model(model_name="qwen3.6-flash",temperature=1.0,max_tokens=1000):
    return init_chat_model(
        model=model_name,
        model_provider="openai",
        base_url = os.getenv("BASE_URL_QWEN"),
        api_key=os.getenv("aliQwen-api"),
        temperature=temperature,
        max_tokens=max_tokens
    )

def get_deepseek_model(model_name="deepseek-chat",temperature=1.0,max_tokens=1000, thinking: bool = True):
    extra_params = {}
    if not thinking:
        extra_params["thinking"] = {"type": "disabled"}  # 关闭思考模式以支持 structured output
    return init_chat_model(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        model_provider="deepseek",
        api_key=os.getenv("deepseek-api"),
        **extra_params
    )


# Log类
class Logger:
    def __init__(self,file_path:str = "",with_print:bool = True):
        self.file_path = file_path
        self.with_print = with_print
        if self.file_path == "":
            self.file_path = f"/Users/Zhuanz1/Document/Project/Knowledge/ai-agents-from-zero/Practice/log/log_{datetime.datetime.now().strftime('%Y-%m-%d')}.txt"
        if not os.path.exists(self.file_path):
            os.makedirs(os.path.dirname(self.file_path),exist_ok=True)
        self.file = open(self.file_path,"a",encoding="utf-8")
    def log(self,message:str):
        self.file.write(message+"\n")
        if self.with_print:
            print(message)
    def log_with_print(self,message:str):
        self.log(message)
        print(message)
    def close(self):
        self.file.close()
    def __del__(self):
        self.close()