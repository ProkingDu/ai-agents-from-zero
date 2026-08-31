from dotenv import load_dotenv
import os 
from langchain.chat_models import init_chat_model
import json
import time
from datetime import datetime
load_dotenv()

# 双模型对话
def getDtime():
    dtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return dtime

def b(s):
    return time.perf_counter() - s;

def msg(c_list):
    return f"""
        你生活在一个测试环境中，你将和另一个大模型展示10轮对话，我只会提供给你会话消息列表，如果会话消息列表中没有任何内容，说明将由你开启一段对话，请你想一个topic描述接下来的10轮对话你们如何展开？会话列表会注明qwen/deepseek，用来提示你之前的对话是你们谁输出的，你可别忘了自己的身份哦。消息列表为json格式。
        会话列表：
    
        ```
        {json.dumps(c_lists)}
        ```
        """

with open("./demo.log","w",encoding="utf-8") as f:
    def pas(c):
        c = getDtime() + "：" + c
        print(c)
        f.write(c)
    
    pas("两迪奥猫辩AI开始运行")
    
    ty = init_chat_model(
        model_provider="openai",
        model="qwen3.7-flash",
        base_url = os.getenv("baseURL_ali"),
        api_key = os.getenv("aliQwen-api")
    )

    # 此方法中用于转发到厂商包，不会自动下载厂商包，如果没有厂商包就实例化会报错。
    ds = init_chat_model(
        model="deepseek:deepseek-v4-flash",
        api_key=os.getenv("deepseek-api"),
        temperature=0.5
    )
    pas("模型初始化完成。")

    c_lists = []
    
    pas("获取初始topic")
    s = time.perf_counter()
    init = ty.invoke(msg(c_lists))
    
    topic = init.content
    usage = init.usage_metadata
    c_lists.append({"qwen":topic})
    pas(f"初始topc获取完成。内容：\n {topic}\n统计信息\n耗时：{b(s)}秒 输入：{usage.get('input_tokens', 0)} tokens 输出：{usage.get('input_tokens', 0)} 总：{usage.get('input_tokens', 0)} tokens \n\n")

    for i in range(10):
        t = time.perf_counter()
        pas(f"第{i}轮对话开始。")
        ds_c = ds.invoke(msg(c_lists))
        pas(f"deepseek说： {ds_c.content}")
        c_lists.append({"deepseek":ds_c.content})
        ty_c = ty.invoke(msg(c_lists))
        c_lists.append({"qwen":ty_c.content})
        
        pas(f"第{i}轮对话完成，耗时：{b(t)}秒 输入：{ds_c.usage_metadata.get('input_tokens', 0) + ty_c.usage_metadata.get('input_tokens', 0)}tokens 输出：{ds_c.usage_metadata.get('output_tokens', 0) + ty_c.usage_metadata.get('output_tokens', 0)} 总：{ds_c.usage_metadata.get('total_tokens', 0) + ty_c.usage_metadata.get('total_tokens', 0)}")
        pas("\n")
        
        
