#!/usr/bin/env python3
"""
main.py

使用 FastAPI 实现 RAG 系统的 Web 界面接口：
- /index：上传 Markdown 文件进行索引
- /query：输入查询后先从 Qdrant 检索相关文档，再调用 enhance_answer 生成最终回答

依赖：fastapi, uvicorn, python-multipart, jinja2 等
"""

import os

from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, Form, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json

# 导入 index.py 和 query.py 中的核心函数
from query import search_qdrant_with_rerank, get_collection_name

app = FastAPI(title="CommonTale system interface", description="CommonTale indexing and querying interface based on FastAPI", version="1.0")
templates = Jinja2Templates(directory="templates")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加favicon路由
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse('static/favicon.ico')

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    首页，展示索引和查询的表单
    """
    try:
        collection_names = get_collection_name()
    except Exception as e:
        collection_names = []
        print(f"failed to get collection names: {e}")
    
    return templates.TemplateResponse("index.html", {"request": request, "collection_names": collection_names})


@app.get("/query")
async def query_document(
    request: Request,
    query: str,
    collection_name: str,
    top_k: int = 10,
    rerank_top_k: int = 5
):
    """
    根据用户输入的查询内容和collection：
    1. 先调用Qdrant检索相关文档，组合为上下文；
    2. 再调用enhance_answer根据问题和上下文生成最终回答。
    """
    try:
        # 通过Qdrant检索相关文档，作为上下文
        results = search_qdrant_with_rerank(
            query, 
            collection_name, 
            top_k=top_k,
            rerank_top_k=rerank_top_k
        )
        
        # 检查结果是否为空
        if not results:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "query_answer": "抱歉，没有找到相关文档。请尝试其他查询或检查集合是否包含相关内容。",
                "query": query,
                "collection_names": get_collection_name(),
                "documents": []
            })
            
        # 提取文档内容和元数据，用于前端显示
        documents = []
        for result in results:
            documents.append({
                'content': result['document'],
                'title': result['title'],
                'author': result['author']
            })

        # 准备上下文文本（只使用内容部分）
        context_text = "\n\n".join([doc['content'] for doc in documents])

        # 调用enhance_answer生成最终回答
        from enhance import enhance_answer
        answer = enhance_answer(query, context_text)
        
        # 获取所有collection名称，用于下拉框
        collection_names = get_collection_name()
    except Exception as e:
        return JSONResponse(content={"error": f"查询失败: {e}"}, status_code=500)
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "query_answer": answer, 
        "query": query,
        "collection_names": collection_names,
        "documents": documents  # 添加文档列表到模板上下文
    })

@app.get("/get_collections")
async def get_collections():
    """
    获取所有可用的集合名称
    """
    try:
        collection_names = get_collection_name()
        return {"collections": collection_names}
    except Exception as e:
        return {"error": str(e), "collections": []}

@app.get("/stream_query")
async def stream_query(
    request: Request,
    query: str,
    collection_name: str,
    top_k: int = 10,
    rerank_top_k: int = 5
):
    """
    流式查询接口，返回SSE格式的流式响应
    """
    try:
        print(f"收到流式查询请求: {query}, 集合: {collection_name}")
        # 通过Qdrant检索相关文档，作为上下文
        results = search_qdrant_with_rerank(
            query, 
            collection_name, 
            top_k=top_k,
            rerank_top_k=rerank_top_k
        )
        
        # 检查结果是否为空
        if not results:
            # 返回错误消息
            return StreamingResponse(
                iter([json.dumps({"error": "没有找到相关文档"}) + "\n\n"]),
                media_type="text/event-stream"
            )
            
        # 提取文档内容和元数据，用于前端显示
        documents = []
        for result in results:
            documents.append({
                'content': result['document'],
                'title': result['title'],
                'author': result['author']
            })
        
        # 准备上下文文本（只使用内容部分）
        context_text = "\n\n".join([doc['content'] for doc in documents])
        
        # 添加日志
        print(f"找到 {len(results)} 个相关文档")
        
        # 发送文档内容作为第一个事件
        async def event_generator():
            print("开始生成流式响应...")
            # 首先发送文档列表
            yield f"data: {json.dumps({'type': 'documents', 'data': documents})}\n\n"
            
            # 然后发送流式回答
            from enhance import enhance_answer
            for text_chunk in enhance_answer(query, context_text):
                yield f"data: {json.dumps({'type': 'text', 'data': text_chunk})}\n\n"
                
            # 在所有文本块发送完毕后
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
            print("流式响应生成完成")
                
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"流式查询失败: {str(e)}\n{error_details}")
        # 返回错误消息
        return StreamingResponse(
            iter([json.dumps({"error": f"查询失败: {str(e)}"}) + "\n\n"]),
            media_type="text/event-stream"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
