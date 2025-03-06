#!/usr/bin/env python3
"""
用于RAG系统的增强脚本。
提供增强用户查询和生成答案的功能。
"""

import os
import dotenv
from openai import OpenAI
from fastapi.responses import StreamingResponse

dotenv.load_dotenv()

# 初始化qwen客户端用于chat completions
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # Replace with your API key if not using env vars
    base_url="https://api.openai.com/v1"
)

qwen_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # Replace with your API key if not using env vars
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def extract_text_from_context(context: list) -> str:
    """
    从上下文对象列表中提取并合并文本。

    Args:
        context (list): 包含'document'属性的对象列表

    Returns:
        str: 从所有上下文文档中提取的合并文本
    """
    extracted_texts = [result.document for result in context]
    combined_text = "\n\n".join(extracted_texts)
    return combined_text

def enhance_query(query: str) -> str:
    """
    增强输入查询以产生更精确的搜索词。

    Args:
        query (str): 用户查询

    Returns:
        str: 增强的查询文本
    """
    completion = client.chat.completions.create(
        model="qwen-plus",  # Refer to https://help.aliyun.com/zh/model-studio/getting-started/models for available models
        messages=[
            {
                'role': 'system',
                'content': (
                    '你是一个专业的RAG信息检索专家，请根据用户的问题进行完善，'
                    '我会把你的回答向量化，在文本数据库中进行相似度识别。'
                    '当用户的问题很模糊的时候，希望你可以给出更加准确和全面的检索词。'
                )
            },
            {'role': 'user', 'content': query}
        ]
    )
    enhanced_query_text = completion.choices[0].message.content
    print("Enhanced query:", enhanced_query_text)
    return enhanced_query_text

def enhance_answer(query: str, context: str):
    """
    通过合并相关上下文来增强答案生成，并以流式方式返回结果。

    Args:
        query (str): 用户查询
        context (str): 与查询相关的上下文信息

    Returns:
        generator: 生成答案的流
    """
    prompt = (
        f"以下是与查询相关的内容：\n\n{context}\n\n"
        f"请根据以上信息回答下面的问题：\n{query}\n回答："
    )
    response = client.chat.completions.create(
        model="gpt-4o", 
        # model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是一个会根据已知文本进行总结的专家，给出简洁的回答。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        stream=True
    )
    
    # 返回生成器
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


