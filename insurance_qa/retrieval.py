"""BM25 + 精确向量检索，一次分组RRF融合，可选重排。"""

from __future__ import annotations

import math
import time

import numpy as np

from .errors import AppError
from .index import load_index, tokenize
from .models.contracts import ModelServiceError
from .query_plan import validate_plan
from .schemas import ChatRequest, QueryPlan, RecallHit
from .store import BuildSnapshot


def bm25_scores(query_tokens, documents, *, k1=1.2, b=0.75):
    """正IDF；DF/平均长度只在固定来源视图内计算，查询词不重复计票。"""
    lengths = np.array([sum(row.values()) for row in documents], dtype=float)
    scores = np.zeros(len(documents))
    if not len(documents) or not lengths.sum():
        return scores
    for token in set(query_tokens):
        tf = np.array([row.get(token, 0) for row in documents], dtype=float)
        df = np.count_nonzero(tf)
        if df:
            idf = math.log1p((len(documents) - df + 0.5) / (df + 0.5))
            scores += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths / lengths.mean()))
    return scores


def ranked(scores, ids, limit, *, positive_only=False):
    """先在完整视图计算同分平均名次，再保留Top-K边界的整个同分组。"""
    order = sorted((i for i, score in enumerate(scores) if not positive_only or score > 0),
                   key=lambda i: (-float(scores[i]), ids[i]))
    rows, start = [], 0
    while start < min(limit, len(order)):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        rows.extend({"evidence_unit_id": ids[i], "score": float(scores[i]), "rank": rank}
                    for i in order[start:end])
        start = end
    return rows


def _failure(error, stage):
    # 地址/超时可降级；响应身份、向量/ID等契约错误不能伪装为成功检索。
    code = error.code if isinstance(error, ModelServiceError) else "service_timeout"
    if code in {"model_identity_mismatch", "auxiliary_invalid_response", "service_invalid_response"}:
        raise AppError("retrieval_contract_error", f"{stage}服务响应契约不符。", 422) from error
    return {"stage": stage, "code": code}


class HybridRetriever:
    ready = True

    def __init__(self, embedding, reranker=None, *, rerank=False, top_k=8,
                 per_channel_k=20, candidate_k=40, rrf_k=60):
        if (any(type(x) is not int or x < 1 for x in (top_k, per_channel_k, candidate_k, rrf_k))
                or top_k > candidate_k or (rerank and reranker is None)):
            raise ValueError("Invalid retrieval budget or missing reranker")
        self.embedding, self.reranker, self.rerank = embedding, reranker, rerank
        self.top_k, self.per_channel_k, self.candidate_k, self.rrf_k = top_k, per_channel_k, candidate_k, rrf_k
        self._cache_key, self._index = None, None

    async def aclose(self):
        for service in (self.embedding, self.reranker):
            if service is not None and hasattr(service, "aclose"):
                await service.aclose()

    async def search(self, plan: QueryPlan, snapshot: BuildSnapshot) -> list[RecallHit]:
        return (await self.search_with_trace(plan, snapshot))["hits"]

    async def search_with_trace(self, plan: QueryPlan, snapshot: BuildSnapshot) -> dict:
        started = time.perf_counter()
        validate_plan(plan, ChatRequest(question=plan.original_question, product_id=plan.product_id,
                                        source_policy=plan.source_policy))
        if (set(snapshot.index_versions) != {"bm25", "dense", "hybrid"}
                or dict(snapshot.index_versions) != dict(snapshot.manifest.get("index_versions", {}))):
            raise AppError("index_version_mismatch", "检索索引版本与快照不一致。", 422)
        profile = snapshot.manifest.get("retrieval", {})
        model = getattr(getattr(self.embedding, "config", None), "model", None)
        if not model or model != profile.get("embedding_model"):
            raise AppError("embedding_model_mismatch", "查询模型与文档向量模型不一致。", 422)
        cache_key = (snapshot.build_id, str(snapshot.build_dir), tuple(snapshot.index_versions.items()))
        try:
            if self._cache_key != cache_key:
                self._index = load_index(snapshot)
                self._cache_key = cache_key
            units, vectors, lexical = self._index
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise AppError("invalid_retrieval_index", "检索索引的内容或ID映射无效。", 422) from exc
        allowed = {"terms"} if plan.source_policy == "terms_only" else {
            "terms", "product_description", "sum_assured_table", "cash_value_table"}
        selected = [i for i, unit in enumerate(units)
                    if unit["product_id"] == plan.product_id and unit["source_type"] in allowed]
        view = [units[i] for i in selected]
        ids = [u["evidence_unit_id"] for u in view]
        queries, seen = [], set()
        for query in plan.queries:
            key = query.text.strip().casefold()
            if key and key not in seen:
                seen.add(key)
                queries.append(query)
        if len(queries) == 1:
            weights = [1.0]
        else:
            generated_count = len(queries) - 1
            weights = [0.5] + [0.5 / generated_count] * generated_count
        diagnostics = {"execution_status": "ok", "source_policy": plan.source_policy, "view_size": len(view),
                       "queries": [{**q.model_dump(), "weight": w} for q, w in zip(queries, weights)],
                       "parameters": {"top_k": self.top_k, "per_channel_k": self.per_channel_k,
                                      "candidate_k": self.candidate_k, "rrf_k": self.rrf_k,
                                      "rerank_requested": self.rerank, "k1": profile["k1"], "b": profile["b"]},
                       "channels": {}, "errors": [], "latency_ms": {}, "reservations": {}}
        if not view:
            diagnostics["latency_ms"]["total"] = round((time.perf_counter() - started) * 1000, 3)
            return {"hits": [], "candidates": [], "diagnostics": diagnostics}

        t0 = time.perf_counter()
        counts = [lexical["token_counts"][i] for i in selected]
        channels = {"bm25": []}
        for query in queries:
            tokens = tokenize(query.text, lexical["terms"])
            scores = bm25_scores(tokens, counts, k1=profile["k1"], b=profile["b"])
            channels["bm25"].append(ranked(scores, ids, self.per_channel_k, positive_only=True))
        diagnostics["latency_ms"]["bm25"] = round((time.perf_counter() - t0) * 1000, 3)
        t0 = time.perf_counter()
        try:
            values = await self.embedding.embed([q.text for q in queries], "query")
        except (ModelServiceError, TimeoutError) as exc:
            diagnostics["errors"].append(_failure(exc, "dense"))
        else:
            try:
                query_vectors = np.asarray(values, dtype=np.float32)
                if query_vectors.ndim != 2:
                    raise ValueError
                norms = np.linalg.norm(query_vectors, axis=1)
                if (query_vectors.shape != (len(queries), profile["dimensions"])
                        or not np.isfinite(query_vectors).all() or not np.isfinite(norms).all() or np.any(norms <= 0)):
                    raise ValueError
            except (ValueError, TypeError) as exc:
                raise AppError("embedding_shape_mismatch", "查询向量维度、数量或数值无效。", 422) from exc
            similarities = (query_vectors / norms[:, None]) @ vectors[selected].T
            channels["dense"] = [ranked(scores, ids, self.per_channel_k) for scores in similarities]
        diagnostics["latency_ms"]["dense"] = round((time.perf_counter() - t0) * 1000, 3)

        candidates = {}
        channel_count = len(channels)
        for channel, results in channels.items():
            diagnostics["channels"][channel] = {}
            for query, weight, rows in zip(queries, weights, results):
                diagnostics["channels"][channel][query.query_id] = {
                    "weight": 1 / channel_count, "rows": rows,
                    "all_tied": len(rows) == len(view) and bool(rows) and rows[0]["score"] == rows[-1]["score"]}
                for row in rows:
                    unit_id = row["evidence_unit_id"]
                    contribution = weight / channel_count / (self.rrf_k + row["rank"])
                    item = candidates.setdefault(unit_id, {"evidence_unit_id": unit_id, "rrf_score": 0.0,
                                                           "supports": [], "retrieved_for_aspect_ids": []})
                    item["rrf_score"] += contribution
                    item["supports"].append({**row, "query_id": query.query_id, "channel": channel,
                                             "query_weight": weight, "channel_weight": 1 / channel_count,
                                             "contribution": contribution, "index_version": snapshot.index_versions[channel]})
                    item["retrieved_for_aspect_ids"] = sorted(set(item["retrieved_for_aspect_ids"] + query.aspect_ids))
        fused = sorted(candidates.values(), key=lambda row: (-row["rrf_score"], row["evidence_unit_id"]))
        by_id = {u["evidence_unit_id"]: u for u in view}
        for position, item in enumerate(fused, 1):
            unit = by_id[item["evidence_unit_id"]]
            item.update(rrf_position=position, section=unit["section"], title=unit["title"], source_type=unit["source_type"])
        # 预算内保留各方面实际命中的代表；这只表示检索来源，不代表证据支持充分。
        reserved = []
        diagnostics["reservation_query_ids"] = {}
        for aspect in plan.aspects:
            # 原问常关联全部方面，不能据此给所有方面都保留全局第一名。
            query_ids = {q.query_id for q in queries[1:] if aspect.aspect_id in q.aspect_ids}
            query_ids = query_ids or {queries[0].query_id}
            relevant = []
            for row in fused:
                score = sum(s["contribution"] for s in row["supports"] if s["query_id"] in query_ids)
                if score > 0:
                    relevant.append((score, row))
            if not relevant:
                continue
            match = max(relevant, key=lambda pair: (pair[0], pair[1]["rrf_score"]))[1]
            if match not in reserved:
                if len(reserved) >= self.candidate_k:
                    continue
                reserved.append(match)
            diagnostics["reservations"][aspect.aspect_id] = match["evidence_unit_id"]
            diagnostics["reservation_query_ids"][aspect.aspect_id] = sorted(query_ids)
        chosen = {x["evidence_unit_id"] for x in reserved}
        for item in fused:
            if len(chosen) >= self.candidate_k:
                break
            chosen.add(item["evidence_unit_id"])
        final = [item for item in fused if item["evidence_unit_id"] in chosen]
        diagnostics["candidate_boundary_tie_truncated"] = (len(fused) > self.candidate_k
            and fused[self.candidate_k - 1]["rrf_score"] == fused[self.candidate_k]["rrf_score"])
        diagnostics["unreserved_aspect_ids"] = [a.aspect_id for a in plan.aspects if a.aspect_id not in diagnostics["reservations"]]

        t0 = time.perf_counter()
        reranked = False
        if self.rerank and final:
            payload = [{"evidence_unit_id": row["evidence_unit_id"],
                        "text": by_id[row["evidence_unit_id"]]["retrieval_text"]} for row in final]
            # 多轮短答可能只有“改成20天”；重排同样需要查询规划补回的问题上下文。
            rerank_query = "\n".join(q.text for q in queries)
            diagnostics["rerank_query"] = rerank_query
            try:
                ranked_rows = await self.reranker.rerank(rerank_query, payload)
            except (ModelServiceError, TimeoutError) as exc:
                diagnostics["errors"].append(_failure(exc, "rerank"))
            else:
                try:
                    scores = {row["evidence_unit_id"]: row["rerank_score"] for row in ranked_rows}
                    if (len(scores) != len(ranked_rows) or set(scores) != chosen
                            or any(type(s) not in (float, int) or not math.isfinite(s) or not 0 <= s <= 1 for s in scores.values())):
                        raise ValueError
                except (ValueError, KeyError, TypeError) as exc:
                    raise AppError("rerank_identity_mismatch", "重排返回的ID或分数无效。", 422) from exc
                for row in final:
                    row["rerank_score"] = float(scores[row["evidence_unit_id"]])
                final.sort(key=lambda row: (-row["rerank_score"], -row["rrf_score"], row["evidence_unit_id"]))
                reranked = True
        diagnostics["latency_ms"]["rerank"] = round((time.perf_counter() - t0) * 1000, 3)
        diagnostics["rerank_applied"] = reranked
        diagnostics["execution_status"] = "degraded" if diagnostics["errors"] else "ok"
        diagnostics["latency_ms"]["total"] = round((time.perf_counter() - started) * 1000, 3)
        channel = "hybrid_rerank" if reranked else "hybrid"
        score_key = "rerank_score" if reranked else "rrf_score"
        hits = []
        for row in final[:self.top_k]:
            strongest_support = max(row["supports"], key=lambda support: support["contribution"])
            hits.append(RecallHit(evidence_unit_id=row["evidence_unit_id"],
                                  query_id=strongest_support["query_id"], channel=channel,
                                  score=row[score_key], index_version=snapshot.index_versions["hybrid"]))
        return {"hits": hits, "candidates": final, "diagnostics": diagnostics}
