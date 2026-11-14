"""按条款关系补齐证据；完整保留前提，引用公式不继承另一分支的门槛。"""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from .corpus_validation import validate_corpus
from .errors import AppError
from .schemas import EvidenceItem, EvidencePack, Gap
from .store import _checked_bytes

# 会影响给付条件/计算口径的定义随规则补齐；其他定义在原问题提及时加入。
CORE_TERMS = {"合同生效日对应日", "周岁", "意外伤害", "全残", "已支付的保险费", "到达年龄",
              "医疗机构", "鉴定机构"}


def _load(snapshot, root):
    """复用离线原文校验，并按本次快照重新核对读取内容，防止缓存后文件被替换。"""
    try:
        validate_corpus(root, snapshot.corpus_dir)
        _checked_bytes(snapshot.corpus_dir, "manifest.json", snapshot.manifest["corpus"]["manifest_sha256"])
        tables = []
        for name in ("evidence_units.jsonl", "source_spans.jsonl", "relations.jsonl"):
            raw = _checked_bytes(snapshot.corpus_dir, name, snapshot.corpus_manifest["files"][name])
            tables.append([json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()])
        units, spans, relations = tables
        edges = defaultdict(list)
        for edge in relations:
            if edge["kind"] not in {"requires_condition", "references", "defines"}:
                raise ValueError("unknown relation kind")
            if edge["kind"] != "defines" and edge["review_status"] != "source_checked":
                raise ValueError("unchecked relation")
            edges[edge["from_id"]].append(edge)
        units = {u["evidence_unit_id"]: u for u in units}
        active, done = set(), set()

        def check_conditions(uid):
            if uid in active:
                raise ValueError("condition cycle")
            if uid in done:
                return
            active.add(uid)
            for edge in edges[uid]:
                if edge["kind"] == "requires_condition":
                    if edge["trigger"]["mode"] != "always" or not edge["propagate_target_conditions"]:
                        raise ValueError("invalid condition relation")
                    check_conditions(edge["to_id"])
            active.remove(uid)
            done.add(uid)

        for uid in units:
            check_conditions(uid)
        return units, {s["span_id"]: s for s in spans}, edges
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
        raise AppError("invalid_evidence_data", "证据版本、原文位置或条款关系校验失败。", 422) from exc


def _closure(seed, question, units, edges, allowed):
    """full递归带前提；reference只借用正文。visited允许合法的交叉引用环。"""
    roles, links, skipped = defaultdict(set), defaultdict(set), set()
    start = (seed, "full")
    roles[start].add("root")
    pending, visited = [start], set()
    while pending:
        key = pending.pop(0)
        if key in visited:
            continue
        visited.add(key)
        uid, mode = key
        if not allowed(units[uid]):
            return None
        for edge in edges[uid]:
            kind, target = edge["kind"], edge["to_id"]
            if kind == "requires_condition" and mode != "full":
                continue
            if kind == "defines":
                terms = edge["trigger"]["terms"]
                if not any(term in question or term in CORE_TERMS for term in terms):
                    skipped.add(target)
                    continue
            target_key = (target, "full" if edge["propagate_target_conditions"] else "reference")
            roles[target_key].add({"requires_condition": "condition", "references": "linked_rule",
                                   "defines": "definition"}[kind])
            links[key].add(target_key)
            pending.append(target_key)
    return roles, links, skipped


def render_context(items):
    """只压缩空白；原始引文另存。相同单元的两种用途共享正文，不能共享前提。"""
    blocks, shown = [], {}
    for item in items:
        header = f"[{item.evidence_id}] {item.section} {item.title} 用途={','.join(item.roles)}"
        if item.required_evidence_ids:
            header += " 必须同引=" + ",".join(item.required_evidence_ids)
        if item.reference_for:
            header += " 仅随其中一个主条款引用=" + ",".join(item.reference_for)
        if item.formula_layout_risk:
            header += " 公式排版待核对，不用于精确计算"
        if "catalog" in item.roles:
            header += " 仅表目录及选表规则，未启用个人金额查询"
        uid = item.unit_ids[0]
        if uid in shown:
            body = "正文同[" + shown[uid] + "]；适用条件以本项用途和依赖为准。"
        else:
            body = re.sub(r"\s+", " ", item.text).strip()
        shown.setdefault(uid, item.evidence_id)
        blocks.append(header + "\n" + body)
    return "\n\n".join(blocks)


def build_evidence(plan, hits, snapshot, *, max_chars=6000, project_root=None, candidates=()):
    """按排名接纳完整证据组；预算是展示字符数，后续生成必须核算整请求token。"""
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    if (dict(snapshot.index_versions) != dict(snapshot.manifest["index_versions"])
            or snapshot.corpus_version != snapshot.corpus_manifest["corpus_version"]
            or any(hit.index_version != snapshot.index_versions["hybrid"] for hit in hits)):
        raise AppError("evidence_version_mismatch", "召回结果与证据快照版本不一致。", 422)
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    units, spans, edges = _load(snapshot, root)
    identifiers = {uid: f"E{i}" for i, uid in enumerate(sorted(units), 1)}
    allowed_types = {"terms"} if plan.source_policy == "terms_only" else {
        "terms", "product_description", "sum_assured_table", "cash_value_table"}

    def allowed(unit):
        return unit["product_id"] == plan.product_id and unit["source_type"] in allowed_types

    def evidence_id(key):
        unit_id, mode = key
        return identifiers[unit_id] + ("R" if mode == "reference" else "")

    selected, selection, gaps, skipped = {}, [], [], set()
    query_aspects = {q.query_id: q.aspect_ids for q in plan.queries}
    retrieved_for = defaultdict(set)
    for hit in hits:
        if hit.evidence_unit_id not in units or hit.query_id not in query_aspects:
            raise AppError("unknown_evidence_hit", "召回单元或查询标识不属于本次输入。", 422)
        retrieved_for[hit.evidence_unit_id].update(query_aspects[hit.query_id])
    for candidate in candidates:
        uid = candidate["evidence_unit_id"]
        if uid in retrieved_for:
            retrieved_for[uid].update(candidate["retrieved_for_aspect_ids"])
    aspect_ids = {aspect.aspect_id for aspect in plan.aspects}
    if any(not ids <= aspect_ids for ids in retrieved_for.values()):
        raise AppError("unknown_evidence_aspect", "召回方面标识不属于本次查询计划。", 422)
    aspect_roots = defaultdict(list)
    seen = set()
    for hit in hits:
        uid = hit.evidence_unit_id
        if uid in seen:
            continue
        seen.add(uid)
        root_id = evidence_id((uid, "full"))
        record = {"unit_id": uid, "evidence_id": root_id, "section": units[uid]["section"]}
        closure = _closure(uid, plan.original_question, units, edges, allowed)
        if closure is None:
            record.update(status="source_policy_blocked", reason="必要证据不在当前产品或来源范围内。")
        else:
            roles, links, omitted_definitions = closure
            proposed = {key: item.model_copy(deep=True) for key, item in selected.items()}
            for key, usage in roles.items():
                unit = units[key[0]]
                item = proposed.setdefault(key, EvidenceItem(evidence_id=evidence_id(key), unit_ids=[key[0]],
                    source_span_ids=unit["source_span_ids"], section=unit["section"], title=unit["title"],
                    formula_layout_risk=unit["formula_layout_risk"], text=unit["text"]))
                item_roles = set(item.roles) | usage
                if key[1] == "reference":
                    item_roles.add("reference")
                if unit["unit_kind"] == "table_catalog":
                    item_roles.add("catalog")
                item.roles = sorted(item_roles)
                if key[1] == "reference":
                    item.reference_for = sorted(set(item.reference_for) | {root_id})
                else:
                    pending, dependencies = list(links[key]), set()
                    while pending:
                        target = pending.pop()
                        if target != key and target not in dependencies:
                            dependencies.add(target)
                            pending.extend(links[target])
                    required_ids = {evidence_id(dependency) for dependency in dependencies}
                    item.required_evidence_ids = sorted(set(item.required_evidence_ids) | required_ids)
            if len(render_context(proposed.values())) > max_chars:
                record.update(status="budget_exceeded", reason="完整条款及依赖超出字符预算，整组不加入。")
            else:
                selected = proposed
                skipped.update(omitted_definitions)
                record.update(status="included", reason="已补齐本次触发的条件、引用和定义。")
                for aid in sorted(retrieved_for[uid]):
                    aspect_roots[aid].append(root_id)
        selection.append(record)
        if record["status"] != "included":
            code = "context_budget_exceeded" if record["status"] == "budget_exceeded" else "source_policy_blocked"
            gaps.append(Gap(code=code, message=f"{record['section']}：{record['reason']}",
                            aspect_ids=sorted(retrieved_for[uid])))
    items = list(selected.values())
    if not items:
        gaps.append(Gap(code="no_evidence", message="本次没有可用的完整证据组。", aspect_ids=[a.aspect_id for a in plan.aspects]))
    used_spans = dict.fromkeys(sid for item in items for sid in item.source_span_ids)
    sources = []
    for span_id in used_spans:
        span = spans[span_id]
        url = f"/sources/{span['document_id']}/{span['document_sha256']}.pdf#page={span['pdf_page']}"
        sources.append({**span, "local_url": url})
    context = render_context(items)
    identity = json.dumps([plan.model_dump(), snapshot.build_id, selection, context], ensure_ascii=False, sort_keys=True)
    return EvidencePack(pack_id="pack-" + hashlib.sha256(identity.encode()).hexdigest()[:16],
        corpus_version=snapshot.corpus_version, index_versions=dict(snapshot.index_versions), source_policy=plan.source_policy,
        evidence_items=items, source_spans=sources, context=context, selection=selection, gaps=gaps,
        execution_status="degraded" if gaps else "ok",
        aspect_coverage=[{"aspect_id": a.aspect_id, "retrieved_for_evidence_ids": aspect_roots[a.aspect_id],
                          "semantic_status": "not_run"} for a in plan.aspects],
        budget={"max_chars": max_chars, "used_chars": len(context), "token_count_verified": False,
                "definition_policy": "core_terms_or_question",
                "unexpanded_definition_unit_ids": sorted(skipped - {key[0] for key in selected})},
        versions={"evidence": "evidence/1.0", "build_id": snapshot.build_id})
