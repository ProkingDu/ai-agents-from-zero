from Practice.functions import *

ds = get_deepseek_model()


# 可以通过调用invoke()方法的时候动态指定温度；尽量不要原地修改。
for i in range(1,11):
    ds.temperature = i/10
    print(f"第{i}次调用，温度：{ds.temperature}")
    print("模型输出：",ds.invoke("我困了").content)

