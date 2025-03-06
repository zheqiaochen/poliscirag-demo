#!/usr/bin/env python3
"""
query.py
用于RAG系统的查询脚本。
计算查询嵌入，搜索Qdrant，并重新排序结果。
"""

import os
import json
import sys
import dotenv
import requests
import voyageai
from qdrant_client import QdrantClient, models
from openai import OpenAI

dotenv.load_dotenv()

# 初始化Qdrant客户端
qdrant_client = QdrantClient(
    url="https://78ad3c0f-57d0-4401-8f8a-8823bfa3c6a7.us-east4-0.gcp.cloud.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY"),
    port=None,
)

# 初始化qwen客户端用于计算embedding
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 初始化VoyageAI客户端用于rerank
vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

def query_embedding(text: str) -> list:
    """
    计算查询文本的embedding

    Args:
        text (str): 查询文本

    Returns:
        list: embedding向量
    """
    completion = client.embeddings.create(
        model="text-embedding-v3",
        input=text,
        dimensions=1024,
        encoding_format="float"
    )
    result_json = json.loads(completion.model_dump_json())
    embedding_vector = result_json['data'][0]['embedding']
    return embedding_vector

def search_qdrant_with_rerank(query: str, collection_name: str, top_k: int = 10, rerank_top_k: int = 5) -> list:
    # 计算查询embedding
    query_emb = query_embedding(query)

    # 执行 Qdrant 查询
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_emb,
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
        limit=top_k,
    )

    # 从查询结果中提取文档内容和元数据
    documents = []
    for i in range(len(results.points)):
        content = results.points[i].payload.get('content', '')
        title = results.points[i].payload.get('title', '未知标题')
        author = results.points[i].payload.get('author', '未知作者')
        
        if content:
            documents.append({
                'content': content,
                'title': title,
                'author': author
            })

    # 如果没有找到有效文档，则返回空列表
    if not documents:
        return []

    # 使用Voyage reranker对结果进行重新排序
    # 只传入内容进行重排序
    contents_only = [doc['content'] for doc in documents]
    reranked_indices = voyage_rerank(query, contents_only, top_k=rerank_top_k)

    # 根据重排序结果重组带有元数据的文档
    reranked_documents = []
    for result in reranked_indices:
        original_idx = result.index
        if original_idx < len(documents):
            reranked_documents.append({
                'document': documents[original_idx]['content'],
                'title': documents[original_idx]['title'],
                'author': documents[original_idx]['author']
                # 如果需要得分信息，可以改为使用正确的属性，例如:
                # 'score': getattr(result, 'similarity', None)
            })
    return reranked_documents



def jina_rerank(query: str, documents: list) -> list:
    """
    使用Jina重新排序文档。

    Args:
        query (str): 查询文本
        documents (list): 文档文本列表

    Returns:
        list: 重新排序的文档
    """
    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Content-Type": "application/json",
        "Authorization": os.getenv("JINA_API_KEY")
    }
    data = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": query,
        "top_n": 5,
        "documents": documents
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_json = response.json()
    reranked = [result['document']['text'] for result in response_json.get('results', [])]
    return reranked

def voyage_rerank(query: str, documents: list, model: str = "rerank-2", top_k: int = 5) -> list:
    """
    使用VoyageAI服务重新排序文档。

    Args:
        query (str): 查询文本
        documents (list): 文档文本列表
        model (str, optional): 重新排序模型
        top_k (int, optional): 需要返回的重排序结果数量

    Returns:
        list: 重新排序的文档
    """
    # 检查文档列表是否为空
    if not documents:
        return []
        
    # 过滤掉空字符串
    valid_documents = [doc for doc in documents if doc and isinstance(doc, str)]
    
    # 再次检查过滤后的列表是否为空
    if not valid_documents:
        return []
        
    result = vo.rerank(
        query=query,
        documents=valid_documents,
        model=model,
        top_k=min(top_k, len(valid_documents)),  # 确保top_k不超过文档数量
    )
    return result.results

def get_collection_name():
    """
    从Qdrant服务器提取所有collection名称。

    Returns:
        list: 所有collection名称的列表
    """
    collections = qdrant_client.get_collections()
    return [collection.name for collection in collections.collections]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python query.py <query> <collection_name> [metadata] [top_k]\n"
              "Example: python query.py '什么是耕田队' '中央苏区革命史调查资料汇编'")
    else:
        query_text = sys.argv[1]
        collection = sys.argv[2]
        metadata = sys.argv[3] if len(sys.argv) > 3 else None
        top_k = int(sys.argv[4]) if len(sys.argv) > 4 else 10

        results = search_qdrant_with_rerank(query_text, collection, top_k=top_k)
        print(results)
