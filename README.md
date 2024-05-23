# Gemma-1.1-7B-Chinese-Chat
Chinese chat model specifically fine-tuned for Chinese through SFT based on the gemma-1.1-7b-it model.

# Updates
- [May 23, 2024] 🔥 support function calling


# Model Summary

[Gemma-1.1-7B-Chinese-Chat](https://huggingface.co/ycjcl868/Gemma-1.1-7B-Chinese-Chat) is an instruction-tuned language model for Chinese & English users built upon the gemma-1.1-7b-it model.([Github](https://github.com/ycjcl868/Gemma-1.1-7B-Chinese-Chat/tree/main))

Developed by: [ycjcl868](https://github.com/ycjcl868)

- License: [Gemma License](https://www.kaggle.com/models/google/gemma/license/consent)
- Base Model: gemma-1.1-7b-it
- Model Size: 8.54B
- Context length: 8K

# Introduction

This is the first model specifically fine-tuned for Chinese & English user through SFT based on the [gemma-1.1-7b-it model](https://huggingface.co/google/gemma-1.1-7b-it).

**Compared to the original [gemma-1.1-7b-it model](https://huggingface.co/google/gemma-1.1-7b-it), our Gemma-1.1-7B-Chinese-Chat model significantly reduces the issues of "Chinese questions with English answers" and the mixing of Chinese and English in responses.**

Training details:

- epochs: 3
- learning rate: 0.0001
- learning rate scheduler type: cosine
- Warmup ratio: 0.1
- cutoff len (i.e. context length): 8192
- global batch size: 2

<br />

# Usage

## Chat Template

The instruction-tuned models use a chat template that must be adhered to for conversational use.
The easiest way to apply it is using the tokenizer's built-in chat template, as shown in the following snippet.

Let's load the model and apply the chat template to a conversation. In this example, we'll start with a single user interaction:

```py
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_id = "ycjcl868/Gemma-1.1-7B-Chinese-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
)

messages = [
    { "role": "user", "content": "写一首诗" },
]
input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=8192,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

# 春风拂柳，春雨轻轻，
# 花香满园，鸟鸣清脆。
# 一朵红梅，红彤彤的，
# 美不胜收，美不胜收。
```

# Examples

```
User: 参加四年级数学竞赛同学中，男同学获奖人数比女同学多2人，女同学获奖人数比男同学人数的一半多2人，男女同学各有多少人获奖?
Assistant:
设男同学的获奖人数
为x，则女同学的获奖人数为y，所以有以下两式：
x=y+2
y=2(x+2)
将第一式代入第二式中，得：
y=2(y+2)+2
y=2y+4+2
y=2y+6
2y=y+6
y=6
因此，女同学获奖人数为6人，男同学获奖人数为8人。
```

```
User: 我的蓝牙耳机坏了，我应该去看耳鼻喉科还是牙科？
Assistant: 蓝牙耳机是属于电子产品的，而不是耳鼻喉科或牙科的范畴。如果你怀疑耳机有故障，建议去电子产品维修店或者原厂维修中心进行检查。如果需要新的耳机，你也可以去电器商店或电子产品市场购买。
```

```
User: 来几句成语
Assistant: “一日三尺”是指过时间一切都会改变。
```

```
User: 列出5种可以改善睡眠质量的方法
Assistant:
1. 保持规律的睡眠时间表
2. 避免下午摄入咖啡因
3. 避免睡前摄入酒精
4. 避免睡前摄入高糖饮食
5. 每天至少进行30分钟的运动
```

```
User: 回答一个开放性的问题：人类存在的意义是什么？
Assistant: 人类存在的意义是一个复杂的哲学和哲学问题，因为人类存在于自然界中，我们作为人类有责任和义务来保护地球和生物，而我们作为人类也有责任来发展自己的精神和思想，来创造更好的未来。
我从哲学角度来看，人类存在的意义是通过我们自己的思想和行动，来实现我们对精神和精神的追求。通过我们自己的努力和创造力，我们可以创造出更多的艺术、文化和思想，来促进人类的进步和发展。
```

```
User: 解释以下缩写： JSON、CEO、PDF、HTML、NASA
Assistant:
JSON（JavaScript Object Notation）：JavaScript对象的表示法，用来在JavaScript和许多其他编程语言之间传递数据。
CEO（Chief Executive Officer）：首席执行官，指企业的高级管理人员，通常是指企业的主席或总裁。
PDF（Portable Document Format）：可移植文档格式，是一种电子文档格式，可以确保文档在不同的操作系统和设备上都能以一致的方式呈现。
HTML（HyperText Markup Language）：超文本标记语言，是网页内容的标记语言，用来定义网页的结构和内容。
NASA（National Aeronautics and Space Administration）：美国国家航空航天局，是美国政府的宇航机构，负责美国在太空和航空方面的研究和发展。
```

## Function call

**User**:
````
以下是您可以使用的工具列表:

```python
def internet_search(query: str):
    \"\"\"
    Returns a list of relevant document snippets for a textual query retrieved from the internet

    Args:
        query (str): Query to search the internet with
    \"\"\"
    pass
```

```python
def directly_answer():
    \"\"\"
    Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history
    \"\"\"
    pass
```

写 'Action:'，后跟要调用的 JSON 中的操作列表，例如.
Action:
```json
[
    {
        "tool_name": "tool name (one of [internet_search, directly_answer])",
        "parameters": "the input to the tool"
    }
]
```

帮我找到今天的新闻有哪些:
````

**Response**:
```
Action:
[
  {
    "tool_name": "internet_search", 
    "parameters": "今天有哪些新闻"
  }
]
```
