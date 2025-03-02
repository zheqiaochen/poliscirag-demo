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

def search_qdrant_with_rerank(query: str, collection_name: str, metadatas=None, top_k: int = 10, rerank_top_k: int = 5) -> list:
    """
    使用Voyage reranker结果搜索Qdrant。

    Args:
        query (str): 用户查询
        collection_name (str): Qdrant集合名称
        metadatas (str or list, optional): 元数据过滤
        top_k (int, optional): 需要检索的top结果数量
        rerank_top_k (int, optional): 重新排序后返回的结果数量

    Returns:
        list: 重新排序的搜索结果
    """
    # 计算查询embedding
    query_emb = query_embedding(query)

    # 根据提供的元数据构造过滤条件
    query_filter = None
    if metadatas:
        if isinstance(metadatas, str):
            metadatas = [metadatas]
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata",
                    match=models.MatchValue(value=metadata)
                )
                for metadata in metadatas
            ]
        )

    # 执行 Qdrant 查询，使用 query_points 方法，并传入 search_params
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_emb,
        query_filter=query_filter,
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
        limit=top_k,
    )

    # 从查询结果中提取文档内容
    documents = []
    for i in range(len(results.points)):
        content = results.points[i].payload.get('content', '')
        if content and isinstance(content, str):  # 确保内容不为空且是字符串
            documents.append(content)
    
    # 检查文档列表是否为空
    if not documents:
        return []  # 如果没有有效文档，返回空列表
        
    # 使用Voyage reranker对结果进行重新排序，传入自定义的rerank_top_k
    reranked_results = voyage_rerank(query, documents, top_k=rerank_top_k)
    return reranked_results


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

        results = search_qdrant_with_rerank(query_text, collection, metadatas=metadata, top_k=top_k)
        print(results)
