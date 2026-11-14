"""直接运行：python main.py；ask 单次问答，chat 连续问答。"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from insurance_qa.errors import AppError
from insurance_qa.models.contracts import ModelServiceError

ROOT = Path(__file__).resolve().parent


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def check_index(root, directory):
    from insurance_qa.corpus_validation import validate_corpus
    from insurance_qa.index import load_index
    from insurance_qa.store import PublishedStore

    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return {"status": "not_ready", "code": "build_manifest_missing", "message": "没有找到构建清单。"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] == "corpus_ready":
        validation = validate_corpus(root, directory)
    else:
        snapshot = PublishedStore(root, directory, manifest["build_id"]).snapshot()
        units, vectors, _ = load_index(snapshot)
        validation = {"indexed_units": len(units), "dimensions": vectors.shape[1]}
    return {"status": "ok", "validation": validation, "runtime_ready": False,
            "note": "文件与结构通过，不代表问答效果已验证。"}


async def query(args):
    from insurance_qa.config import load_settings
    from insurance_qa.pipeline import create_pipeline
    from insurance_qa.query_plan import OriginalQuestionPlanner
    from insurance_qa.schemas import ChatRequest

    settings = load_settings(args.project_root)
    request = ChatRequest(question=args.question, product_id=settings.policy.product_id,
                          source_policy=args.source_policy)
    if request.source_policy not in settings.policy.allowed_source_policies:
        raise AppError("source_policy_not_allowed", "请先在pipeline.yaml中启用该来源范围。", 422)
    pipeline = create_pipeline(settings, rerank=getattr(args, "rerank", None))
    try:
        if args.command == "ask":
            return (await pipeline.ask(request)).model_dump()
        ticket = pipeline.sessions.begin(product_id=request.product_id, source_policy=request.source_policy)
        if getattr(args, "rewrite", True):
            plan = await pipeline.plan_query(request, ticket)
        else:
            plan = await OriginalQuestionPlanner().plan(request, ticket)
            plan.execution_status = "ok"
            plan.diagnostics = ["original_only_requested"]
        if args.command == "plan":
            return {"status": plan.execution_status, "query_plan": plan.model_dump()}
        snapshot = pipeline.store.snapshot()
        result = await pipeline.retriever.search_with_trace(plan, snapshot)
    finally:
        await pipeline.aclose()
    stage_statuses = (plan.execution_status, result["diagnostics"]["execution_status"])
    status = "degraded" if "degraded" in stage_statuses else "ok"
    response = {"status": status, "question": request.question, "query_plan": plan.model_dump(),
                "source_policy": request.source_policy, "build_id": snapshot.build_id,
                "diagnostics": result["diagnostics"], "answer_generated": False}
    if args.command == "evidence":
        from insurance_qa.evidence import build_evidence

        pack = build_evidence(plan, result["hits"], snapshot, max_chars=args.max_chars,
                              project_root=settings.project_root, candidates=result.get("candidates", []))
        response["evidence_pack"] = pack.model_dump()
        if pack.execution_status != "ok":
            response["status"] = pack.execution_status
        return response
    units = {u["evidence_unit_id"]: u for u in rows(snapshot.corpus_dir / "evidence_units.jsonl")}
    spans = {s["span_id"]: s for s in rows(snapshot.corpus_dir / "source_spans.jsonl")}
    matches = []
    for hit in result["hits"]:
        unit = units[hit.evidence_unit_id]
        selected_spans = [spans[span_id] for span_id in unit["source_span_ids"]]
        matches.append({
            **hit.model_dump(),
            "section": unit["section"],
            "title": unit["title"],
            "source_type": unit["source_type"],
            "pdf_pages": sorted({span["pdf_page"] for span in selected_spans}),
            "source_url": selected_spans[0]["source_url"],
            "excerpt": unit["text"][:300],
        })
    return {**response, "matches": matches}


async def run(args):
    from insurance_qa.config import load_settings

    root = args.project_root.expanduser().resolve()
    if args.command in {"ask", "plan", "search", "evidence"}:
        return await query(args)
    if args.command == "build-corpus":
        from insurance_qa.corpus import build_corpus

        return build_corpus(root, args.pages_dir)
    if args.command == "build-index" and args.check and args.build_dir:
        return check_index(root, (root / args.build_dir).resolve())
    settings = load_settings(root)
    if args.command == "doctor":
        from insurance_qa.pipeline import create_pipeline

        pipeline = create_pipeline(settings)
        try:
            state = pipeline.readiness()
            return {"status": "ok" if state["ready"] else "not_ready", "readiness": state}
        finally:
            await pipeline.aclose()
    if args.check:
        return check_index(root, settings.build_dir)
    from insurance_qa.index import build_index

    directory = args.build_dir
    if not directory:
        raise ValueError("请用 --build-dir 指定 data/corpus 下的语料目录。")
    return await build_index(settings, (root / directory).resolve())


async def chat(args):
    """只保存服务返回的会话标识；每轮复用同一Pipeline，不在客户端拼接旧答案。"""
    from insurance_qa.config import load_settings
    from insurance_qa.pipeline import create_pipeline
    from insurance_qa.schemas import ChatRequest

    settings = load_settings(args.project_root)
    if args.source_policy not in settings.policy.allowed_source_policies:
        raise AppError("source_policy_not_allowed", "当前应用未启用该来源范围。", 422)
    pipeline = create_pipeline(settings)
    session_id, revision = None, None
    print("输入问题或补充信息；/new 新建会话，exit 退出。")
    try:
        while True:
            question = input("你：")
            command = question.strip().lower()
            if command in {"exit", "quit"}:
                break
            if command == "/new":
                session_id, revision = None, None
                print("已新建会话。")
                continue
            if not command:
                continue
            try:
                result = await pipeline.ask(ChatRequest(question=question, product_id=settings.policy.product_id,
                    source_policy=args.source_policy, session_id=session_id, expected_revision=revision))
            except AppError as exc:
                message = exc.message
                if exc.status_code == 409:
                    message += " 请用 /new 新建会话。"
                print(message)
                continue
            for block in result.answer_blocks:
                text = block.text
                if block.citation_ids:
                    text += " [" + "][".join(block.citation_ids) + "]"
                print(text)
            for gap in result.gaps:
                print(gap.message)
            for question in result.clarification_questions:
                print("请补充：" + question)
            for citation in result.citations:
                pages = "、".join(map(str, citation.pdf_pages))
                print(f"[{citation.citation_id}] {citation.section}，PDF第{pages}页：{citation.source_url}")
            if result.execution_status in {"ok", "degraded"} and result.session_id and result.revision is not None:
                session_id, revision = result.session_id, result.revision
    except (EOFError, KeyboardInterrupt):
        print("\n会话结束。")
    finally:
        await pipeline.aclose()


def parser():
    result = argparse.ArgumentParser(description="保险问答：启动、规划、检索与构建")
    commands = result.add_subparsers(dest="command", required=True)
    for name, help_text in {
        "serve": "启动网页和API", "ask": "单次问答并显示引用", "chat": "连续问答，/new新建会话",
        "plan": "查看意图和查询计划", "search": "检索并显示原文出处",
        "evidence": "补齐条件、定义与出处，查看证据包",
        "build-corpus": "构建并校验证据库", "build-index": "构建索引；--check只校验",
        "doctor": "检查问答各阶段是否已接入",
    }.items():
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--project-root", type=Path, default=ROOT)
        if name == "serve":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8000)
        elif name in {"ask", "chat", "plan", "search", "evidence"}:
            if name != "chat":
                command.add_argument("question")
            command.add_argument("--source-policy", choices=["terms_only", "official_materials"], default="terms_only")
            if name in {"search", "evidence"}:
                command.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
                command.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
            if name == "evidence":
                command.add_argument("--max-chars", type=positive_int, default=6000, help="证据上下文字符预算，非token数")
        elif name == "build-corpus":
            command.add_argument("--pages-dir", default="data/pages")
        elif name == "build-index":
            command.add_argument("--build-dir", help="构建时指定语料目录；--check 时可指定待校验目录")
            command.add_argument("--check", action="store_true")
    return result


def main(argv=None):
    arguments = list(argv) if argv is not None else sys.argv[1:]
    args = parser().parse_args(arguments or ["serve"])
    try:
        if args.command == "chat":
            asyncio.run(chat(args))
            return 0
        if args.command == "serve":
            import uvicorn
            from insurance_qa.api import create_app
            from insurance_qa.config import load_settings

            uvicorn.run(create_app(settings=load_settings(args.project_root)), host=args.host, port=args.port)
            return 0
        result = asyncio.run(run(args))
    except (AppError, ModelServiceError) as exc:
        result = {"status": "service_error", "code": exc.code, "message": str(exc)}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"status": "validation_error", "code": "input_invalid", "message": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    status = result.get("execution_status", result.get("status"))
    return 0 if status in {"ok", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
