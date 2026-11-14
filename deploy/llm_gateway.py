"""转发卡5的模型列表、生成和token计数；可独立运行于CPU。"""

from contextlib import asynccontextmanager
import os

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response


@asynccontextmanager
async def lifespan(app):
    upstream = os.environ.get("LLM_UPSTREAM", "http://127.0.0.1:8018")
    async with httpx.AsyncClient(base_url=upstream, timeout=90, trust_env=False,
                                 follow_redirects=False) as client:
        app.state.upstream = client
        yield


app = FastAPI(title="本地 LLM 接口转发", lifespan=lifespan)


async def forward(method, path, body=None):
    try:
        response = await app.state.upstream.request(method, path, json=body)
    except httpx.RequestError:
        return JSONResponse(status_code=503, content={"error": {"code": "llm_upstream_unavailable"}})
    # 只传回响应正文、状态和类型，不转发重定向地址及其他上游响应头。
    return Response(content=response.content, status_code=response.status_code,
                    headers={"content-type": response.headers.get("content-type", "application/json")})


@app.get("/v1/models")
async def models():
    return await forward("GET", "/v1/models")


@app.post("/v1/chat/completions")
async def chat(body: dict):
    if body.get("stream", False) is not False:
        return JSONResponse(status_code=400, content={"error": {"code": "stream_not_supported"}})
    return await forward("POST", "/v1/chat/completions", body)


@app.post("/tokenize")
async def tokenize(body: dict):
    return await forward("POST", "/tokenize", body)
