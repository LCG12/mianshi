"""卡 2 的三个薄适配器：发请求、检查响应、保持输入输出对应关系。"""

import math

from .contracts import ModelServiceError
from .client import PhysicalServiceClient

ROUTES = {"insurance", "assistant_info", "chitchat", "out_of_scope", "unknown"}
TASKS = {"rule_qa", "case_qa", "amount_query", "action_request", "unknown"}
TOPICS = {"eligibility", "coverage", "exclusions", "premium", "policy_value", "dividend", "claims", "contract_admin"}
BATCH_SIZE = 8
DIMENSIONS = 1024


class _LocalClient(PhysicalServiceClient):
    async def post_json(self, payload):
        raw = await super().post_json(payload)
        if raw.get("model") != self.config.model:
            raise ModelServiceError("model_identity_mismatch", self.config.name)
        return raw

    def _invalid(self):
        return ModelServiceError("auxiliary_invalid_response", self.config.name)


class LocalIntentClient(_LocalClient):
    async def classify(self, question, context):
        raw = await self.post_json({"question": question, "context": dict(context)})
        raw.setdefault("route", "unknown")  # 旧服务没有路由字段，交给查询规划继续判断。
        try:
            route, task, topics, status = raw["route"], raw["task"], raw["topics"], raw["execution_status"]
            if (not isinstance(route, str) or route not in ROUTES
                    or not isinstance(task, str) or task not in TASKS or not isinstance(topics, list)
                    or any(not isinstance(topic, str) or topic not in TOPICS for topic in topics)
                    or status not in {"ok", "degraded"} or raw.get("confidence_status") != "unavailable"):
                raise ValueError
            if route in {"assistant_info", "chitchat", "out_of_scope"} and (task != "unknown" or topics):
                raise ValueError
            if status == "degraded" and (route != "unknown" or task != "unknown" or topics):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise self._invalid() from None
        # 保留 degraded 和 reason，由业务层决定如何回退，不伪造分类置信度。
        return raw


class LocalEmbeddingClient(_LocalClient):
    async def embed(self, texts, purpose):
        if purpose not in {"query", "document"}:
            raise ValueError("purpose must be query or document")
        vectors = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = list(texts[start:start + BATCH_SIZE])
            raw = await self.post_json({"texts": batch, "purpose": purpose})
            try:
                values = raw["vectors"]
                if (raw.get("truncated") is not False or raw.get("dimensions") != DIMENSIONS
                        or not isinstance(values, list) or len(values) != len(batch)):
                    raise ValueError
                for vector in values:
                    if (not isinstance(vector, list) or len(vector) != DIMENSIONS
                            or not all(type(v) in (int, float) and math.isfinite(v) for v in vector)):
                        raise ValueError
                    norm = math.hypot(*vector)
                    if not math.isfinite(norm) or norm <= 0:
                        raise ValueError
            except (KeyError, TypeError, ValueError, OverflowError):
                raise self._invalid() from None
            vectors.extend(values)
        return vectors


class LocalRerankClient(_LocalClient):
    async def rerank(self, question, candidates):
        ranked = []
        for start in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[start:start + BATCH_SIZE]
            raw = await self.post_json({"query": question, "documents": [row["text"] for row in batch]})
            try:
                rows = raw["results"]
                if (raw.get("truncated") is not False or raw.get("score_type") != "yes_no_softmax"
                        or not isinstance(rows, list) or len(rows) != len(batch)):
                    raise ValueError
                indexes = [row["index"] for row in rows]
                if any(type(index) is not int for index in indexes) or set(indexes) != set(range(len(batch))):
                    raise ValueError
                for row in rows:
                    score = row["score"]
                    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError
                    ranked.append({**batch[row["index"]], "rerank_score": float(score)})
            except (KeyError, TypeError, ValueError, OverflowError):
                raise self._invalid() from None
        # 服务逐对算 yes/no 分数，不做批内归一化，因此可以合并批次排序。
        return sorted(ranked, key=lambda row: row["rerank_score"], reverse=True)
