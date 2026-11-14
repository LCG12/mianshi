"""网页、问答状态与原文PDF的HTTP入口。"""

from contextlib import asynccontextmanager
from pathlib import Path as FilePath

from fastapi import FastAPI, HTTPException, Path, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from insurance_qa.errors import AppError
from insurance_qa.schemas import AnswerResult, ChatRequest


def create_app(pipeline=None, settings=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.pipeline is None:
            from insurance_qa.config import load_settings
            from insurance_qa.pipeline import create_pipeline

            app.state.settings = settings or load_settings()
            app.state.pipeline = create_pipeline(app.state.settings)
        try:
            yield
        finally:
            close = getattr(app.state.pipeline, "aclose", None)
            if close is not None:
                await close()

    app = FastAPI(title="保险条款问答", version="0.1.0", lifespan=lifespan)
    app.state.pipeline = pipeline
    app.state.settings = settings

    @app.exception_handler(AppError)
    async def application_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, __: Exception) -> JSONResponse:
        # 内部异常不把模型地址、凭据或原始响应传给浏览器。
        return JSONResponse({"error": {"code": "internal_error", "message": "本次请求未能完成，请稍后重试。"}},
                            status_code=500)

    @app.get("/", include_in_schema=False)
    async def index(request: Request) -> FileResponse:
        configured = request.app.state.settings
        root = FilePath(configured.project_root) if configured else FilePath(__file__).resolve().parents[1]
        page = root / "web" / "index.html"
        if not page.is_file():
            raise HTTPException(status_code=404, detail="演示页面尚未安装。")
        return FileResponse(page, media_type="text/html")

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready(request: Request) -> JSONResponse:
        state = request.app.state.pipeline.readiness()
        return JSONResponse(status_code=200 if state.get("ready") else 503, content=state)

    @app.post("/api/chat", response_model=AnswerResult,
              responses={503: {"model": AnswerResult}, 422: {"description": "请求或结果校验失败"},
                         409: {"description": "会话失效、超限或修订冲突"}})
    async def chat(body: ChatRequest, request: Request, response: Response) -> AnswerResult:
        result = await request.app.state.pipeline.ask(body)
        response.status_code = {"service_error": 503, "validation_error": 422}.get(result.execution_status, 200)
        return result

    @app.get("/sources/{document_id}/{sha256}.pdf", response_class=FileResponse)
    async def source(
        request: Request,
        document_id: str = Path(pattern=r"^[A-Za-z0-9_-]{1,128}$"),
        sha256: str = Path(pattern=r"^[a-f0-9]{64}$"),
    ) -> FileResponse:
        # 文件路径只由已校验的数据清单解析。
        path = request.app.state.pipeline.resolve_source(document_id, sha256)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="未找到该版本的原始资料。")
        return FileResponse(path, media_type="application/pdf", filename=path.name,
                            content_disposition_type="inline", headers={"X-Content-Type-Options": "nosniff"})

    return app
