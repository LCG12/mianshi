"""把已核对的产品结构转为证据、原文位置和关系；此阶段不发布索引。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from insurance_qa.corpus_layout import terms_units
from insurance_qa.corpus_validation import validate_corpus

VERSION = "corpus/1.0"
EXTRACTION = "pdftotext-layout/1.0"
PRODUCT = "sshy_s_hengying"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text).replace("\uf0cd", "×")


def page_ranges(pages: list[str], start: list[int], end: list[int]) -> list[list[int]]:
    """将跨页起止位置拆成每页的非空区间，页码从1开始。"""
    ranges = []
    for page in range(start[0], end[0] + 1):
        left = start[1] if page == start[0] else 0
        right = end[1] if page == end[0] else len(pages[page - 1])
        if left < right:
            ranges.append([page, left, right])
    return ranges


def slice_spans(spans: list[list[int]], start: int, end: int) -> list[list[int]]:
    """拼接文本的区间映射回各页；不修改原文坐标。"""
    result, offset = [], 0
    for page, left, right in spans:
        a, b = max(start, offset), min(end, offset + right - left)
        if a < b:
            result.append([page, left + a - offset, left + b - offset])
        offset += right - left
    return result


def description_units(pages: list[str], config: dict) -> list[dict]:
    """说明书单独标记来源；脚注归回产品特色，投保示例不进入正文。"""
    sections = config["description_sections"]
    units = [{"section": "intro", "title": "产品说明书名称", "unit_kind": "metadata",
              "spans": [[1, 0, sections[0]["start"][1]]]}]
    for i, section in enumerate(sections):
        last_page = config["description_excluded_sections"][0]["pages"][0] - 1
        end = sections[i + 1]["start"] if i + 1 < len(sections) else [last_page, len(pages[last_page - 1])]
        spans = page_ranges(pages, section["start"], end)
        if section["section"] == "三":
            spans[0][2] = config["description_footnotes"][0][1]
        if section["section"] == config["description_footnote_section"]:
            spans += config["description_footnotes"]
        splits = config.get("description_splits", {}).get(section["section"])
        if not splits:
            units.append({**section, "unit_kind": "clause", "spans": spans})
            continue
        text = "".join(pages[p - 1][a:b] for p, a, b in spans)
        positions = []
        for part in splits:
            if text.count(part["start"]) != 1:
                raise ValueError(f"说明书 {section['section']}: 分段锚点不唯一")
            positions.append(text.index(part["start"]))
        if positions != sorted(positions):
            raise ValueError("说明书分段锚点顺序错误")
        positions[0] = 0  # 章标题随第一个子段保存，不丢文字。
        for part, start, end in zip(splits, positions, positions[1:] + [len(text)]):
            units.append({"section": section["section"] + "." + part["key"],
                          "title": section["title"] + "：" + part["title"], "unit_kind": "clause",
                          "spans": slice_spans(spans, start, end), "requires": part["requires"],
                          "references": part.get("references", [])})
    return units


def merge_ranges(ranges: list[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def build_corpus(root: Path, pages_directory: str = "data/pages") -> dict:
    root = root.resolve()
    pages_dir = (root / pages_directory).resolve()
    if not pages_dir.is_relative_to(root):
        raise ValueError("逐页文件必须位于项目目录内")
    config_paths = [root / "configs/corpus_terms.json", root / "configs/corpus_catalog.json"]
    terms_config, catalog = [json.loads(p.read_text()) for p in config_paths]
    expected = {**terms_config["input_sha256"], **catalog["input_sha256"]}
    sources, documents, layouts = [], {}, []
    for source in json.loads((root / "data/raw/sources.json").read_text())["documents"]:
        doc_id = source["id"]
        pdf = (root / source["local_pdf"]).resolve()
        file = pages_dir / f"{doc_id}.pages.jsonl"
        if not pdf.is_relative_to(root):
            raise ValueError("PDF必须位于项目目录内")
        hashes = {"pdf": digest(pdf.read_bytes()), "page_text": digest(file.read_bytes())}
        if hashes != expected.get(doc_id):
            raise ValueError(f"{doc_id}: 原件或逐页文本已变化，请重新核对结构配置")
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        if len(rows) != source["expected_pages"]:
            raise ValueError(f"{doc_id}: 页数不符")
        for number, row in enumerate(rows, 1):
            if (row["page"] != number or row["document_id"] != doc_id
                    or row["document_type"] != source["document_type"] or row["source_url"] != source["url"]
                    or row["extraction_method"] != "pdftotext -layout -enc UTF-8" or not row["text"].strip()):
                raise ValueError(f"{doc_id}: 第{number}页元信息不符")
        pages = [row["text"] for row in rows]
        metadata = {"document_id": doc_id, "sha256": hashes["pdf"], "local_pdf": source["local_pdf"],
                    "source_url": source["url"], "source_type": source["document_type"],
                    "page_text_path": str(file.relative_to(root)), "page_text_sha256": hashes["page_text"]}
        sources.append(metadata)
        documents[doc_id] = (metadata, pages)
        if source["document_type"] == "terms":
            parts = terms_units(pages, terms_config)
        elif source["document_type"] == "product_description":
            parts = description_units(pages, catalog)
        else:
            parts = [{**table, "unit_kind": "table_catalog"} for table in catalog["tables"]
                     if table["document_id"] == doc_id]
        layouts.extend({**part, "document_id": doc_id} for part in parts)

    # 输入、产品配置和构建代码都绑定版本；相同输入重跑复用同一目录。
    inputs = {str(p.relative_to(root)): digest(p.read_bytes()) for p in config_paths}
    inputs["data/raw/sources.json"] = digest((root / "data/raw/sources.json").read_bytes())
    for name in ("corpus.py", "corpus_layout.py", "corpus_validation.py"):
        inputs[f"insurance_qa/{name}"] = digest(Path(__file__).with_name(name).read_bytes())
    build_id = "corpus-" + digest(encoded([VERSION, sources, inputs]))[:16]
    units, spans, by_key = [], {}, {}
    for part in layouts:
        doc_id = part["document_id"]
        metadata, pages = documents[doc_id]
        span_ids = []
        for page, start, end in part["spans"]:
            raw = pages[page - 1]
            if not 0 <= start < end <= len(raw):
                raise ValueError(f"{doc_id}/{part['section']}: 原文跨度越界")
            page_hash = digest(raw.encode())
            span_id = "sp_" + digest(encoded([metadata["sha256"], EXTRACTION, page_hash, page, start, end]))[:20]
            spans[span_id] = {
                "span_id": span_id, "document_id": doc_id, "document_sha256": metadata["sha256"],
                "source_type": metadata["source_type"], "extraction_version": EXTRACTION,
                "page_text_sha256": page_hash, "pdf_page": page, "start_char": start, "end_char": end,
                "offset_unit": "unicode_codepoint", "exact": raw[start:end],
                "prefix": raw[max(0, start - 30):start], "suffix": raw[end:end + 30],
                "source_url": metadata["source_url"], "local_pdf": metadata["local_pdf"],
                "page_text_path": metadata["page_text_path"],
                "representation_note": "pdftotext布局原文；上标、表格及私用字形须结合原PDF核对，不用于自动金额计算。",
            }
            span_ids.append(span_id)
        text = "\n".join(spans[s]["exact"] for s in span_ids)
        unit_id = f"{doc_id}:{part['section']}:" + digest(encoded([VERSION, inputs, span_ids]))[:12]
        unit = {"evidence_unit_id": unit_id, "document_id": doc_id, "product_id": PRODUCT,
                "source_type": metadata["source_type"], "section": part["section"], "title": part["title"],
                "unit_kind": part["unit_kind"], "source_span_ids": span_ids, "text": text,
                "retrieval_text": part["title"] + "\n" + compact(part.get("retrieval_text", text)),
                "retrievable": part["unit_kind"] not in {"heading", "metadata"},
                "formula_layout_risk": bool(re.search(r"[（(]n\s*-\s*1[）)]", text)),
                "lookup_supported": False}
        if part["unit_kind"] == "table_catalog":
            unit["catalog"] = {k: v for k, v in part.items() if k not in
                               {"spans", "document_id", "section", "title", "unit_kind"}}
            unit["retrieval_text"] = "\n".join([part["title"], part["unit"], part["applicability"]])
        key = (doc_id, part["section"])
        if key in by_key:
            raise ValueError(f"duplicate section: {key}")
        by_key[key] = unit
        units.append(unit)

    relations, seen, unresolved = [], set(), []

    def relate(unit: dict, target: dict, kind: str, terms: list[str] | None = None) -> None:
        key = (unit["evidence_unit_id"], target["evidence_unit_id"], kind)
        if key in seen or key[0] == key[1]:
            return
        seen.add(key)
        relations.append({"from_id": key[0], "to_id": key[1], "kind": kind,
                          "trigger": {"mode": "term_used" if terms else "always", "terms": terms or []},
                          "basis_span_ids": list(dict.fromkeys(unit["source_span_ids"] + target["source_span_ids"])),
                          "review_status": "source_checked" if kind != "defines" else "lexical_link",
                          "propagate_target_conditions": kind == "requires_condition"})

    for part, unit in zip(layouts, units):
        for field, kind in (("requires", "requires_condition"), ("references", "references")):
            for section in part.get(field, []):
                target_doc = PRODUCT + "_terms" if section[0].isdigit() else unit["document_id"]
                target = by_key.get((target_doc, section))
                if target is None:
                    raise ValueError(f"{unit['section']}: 未找到依赖 {section}")
                relate(unit, target, kind)
        # 只识别引号内的条款号，避免把1.75%、表内数字或12.10误作12.1。
        for section in re.findall(r'“(\d+\.\d+(?:\.\d+)?)(?=\s|[\u4e00-\u9fff])', unit["text"]):
            target = by_key.get((PRODUCT + "_terms", section))
            if target is not None:
                relate(unit, target, "references")
            elif not any(ref.startswith(section + ".") for ref in part.get("references", [])):
                unresolved.append({"evidence_unit_id": unit["evidence_unit_id"], "target_section": section})
    definitions = [u for u in units if u["source_type"] == "terms" and u["section"].startswith("12.")]
    for unit in units:
        if unit["source_type"] != "terms" or not unit["retrievable"]:
            continue
        for definition in definitions:
            term = definition["title"]
            if term in compact(unit["text"]):
                relate(unit, definition, "defines", [term])

    # 检索/重排表示也带必要前提；引用计算公式不会继承目标分支的相反条件。
    required = defaultdict(list)
    for relation in relations:
        if relation["kind"] == "requires_condition":
            required[relation["from_id"]].append(relation["to_id"])

    def conditions(unit_id: str, visiting: tuple = ()) -> list[str]:
        if unit_id in visiting:
            raise ValueError("必要条件关系成环")
        ordered = []
        for parent in required[unit_id]:
            ordered.extend(conditions(parent, (*visiting, unit_id)))
            ordered.append(parent)
        return list(dict.fromkeys(ordered))

    search_texts = {u["evidence_unit_id"]: u["retrieval_text"] for u in units}
    for unit in units:
        context = conditions(unit["evidence_unit_id"])
        unit["retrieval_context_ids"] = context
        unit["retrieval_text"] = "\n".join([search_texts[x] for x in context] + [unit["retrieval_text"]])

    coverage = []
    included = defaultdict(list)
    for span in spans.values():
        included[(span["document_id"], span["pdf_page"])].append([span["start_char"], span["end_char"]])
    for doc_id, (metadata, pages) in documents.items():
        for page, raw in enumerate(pages, 1):
            ranges = merge_ranges(included[(doc_id, page)])
            excluded = []
            source_type = metadata["source_type"]
            if source_type == "terms" and page < terms_config["body_start_page"]:
                reason = "阅读指引及目录，不作为合同正文"
            elif source_type == "product_description" and any(
                    section["pages"][0] <= page <= section["pages"][1]
                    for section in catalog["description_excluded_sections"]):
                reason = "投保示例及假设，不作为通用产品规则"
            elif source_type.endswith("_table"):
                reason = "数表数据行及未选表头；本阶段仅编制目录，未启用数值查询"
            else:
                reason = None
            if reason:
                cursor = 0
                for start, end in ranges + [[len(raw), len(raw)]]:
                    if cursor < start:
                        excluded.append({"start_char": cursor, "end_char": start, "reason": reason})
                    cursor = end
            coverage.append({"document_id": doc_id, "pdf_page": page, "included": ranges, "excluded": excluded})

    artifacts = {"evidence_units.jsonl": units, "source_spans.jsonl": list(spans.values()),
                 "relations.jsonl": relations, "coverage.jsonl": coverage}
    manifest = {"schema_version": "build-manifest/1.0", "build_id": build_id, "status": "corpus_ready",
                "corpus_version": VERSION, "index_versions": {}, "sources": sources, "inputs": inputs,
                "files": {}, "unresolved_references": unresolved, "semantic_review": "not_run"}
    parent = root / "data/corpus"
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / build_id
    with TemporaryDirectory(prefix=".corpus-", dir=parent) as temporary:
        staging = Path(temporary) / "build"
        staging.mkdir()
        for name, records in artifacts.items():
            data = b"".join(encoded(record) for record in records)
            (staging / name).write_bytes(data)
            manifest["files"][name] = digest(data)
        (staging / "manifest.json").write_bytes(encoded(manifest))
        validation = validate_corpus(root, staging)
        if directory.exists():
            if (directory / "manifest.json").read_bytes() != (staging / "manifest.json").read_bytes():
                raise ValueError("相同构建ID已有不同产物，拒绝覆盖")
            validate_corpus(root, directory)
        else:
            staging.rename(directory)
    result = {"status": "ok", "build_id": build_id, "build_directory": str(directory.relative_to(root)),
              "build_status": "corpus_ready", "validation": validation,
              "inventory": {doc_id: {"units": sum(u["document_id"] == doc_id for u in units),
                                      "retrievable": sum(u["document_id"] == doc_id and u["retrievable"] for u in units)}
                            for doc_id in documents},
              "unresolved_references": unresolved, "index_built": False, "runtime_ready": False}
    return result
