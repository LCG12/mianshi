"""Validate a corpus before indexing; this does not review insurance semantics."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(all(isinstance(row, dict) for row in rows), f"{path.name}: expected JSON objects")
    return rows


def _checked_file(base: Path, name: str, expected: str) -> Path:
    _require(isinstance(name, str) and bool(name), "missing file path")
    path = (base / name).resolve()
    _require(path.is_relative_to(base) and path.is_file(), f"invalid file location: {name}")
    _require(isinstance(expected, str) and bool(re.fullmatch(r"[a-f0-9]{64}", expected)),
             f"invalid SHA256: {name}")
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"file hash mismatch: {name}")
    return path


def _unique(rows: list[dict], field: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        value = row.get(field)
        _require(isinstance(value, str) and bool(value.strip()), f"missing {field}")
        _require(value not in result, f"duplicate {field}: {value}")
        result[value] = row
    return result


def _mark(mask: bytearray, start: int, end: int) -> None:
    _require(type(start) is int and type(end) is int and 0 <= start < end <= len(mask),
             f"invalid character interval: {start}:{end}")
    mask[start:end] = b"\x01" * (end - start)


def validate_corpus(root: Path, directory: Path) -> dict:
    """Check hashes, exact offsets, references and full-page coverage; never write."""
    root, directory = root.resolve(), directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    _require(isinstance(manifest, dict), "manifest must be an object")
    _require(manifest.get("schema_version") == "build-manifest/1.0"
             and manifest.get("status") == "corpus_ready" and manifest.get("index_versions") == {},
             "expected a corpus_ready manifest without indexes")
    for field in ("build_id", "corpus_version"):
        _require(isinstance(manifest.get(field), str) and bool(manifest[field].strip()), f"missing {field}")
    files = manifest.get("files")
    required = {"evidence_units.jsonl", "source_spans.jsonl", "relations.jsonl", "coverage.jsonl"}
    _require(isinstance(files, dict) and required <= files.keys(), "missing corpus files")
    paths = {name: _checked_file(directory, name, digest) for name, digest in files.items()}

    source_rows = manifest.get("sources")
    _require(isinstance(source_rows, list) and bool(source_rows)
             and all(isinstance(row, dict) for row in source_rows), "missing sources")
    sources = _unique(source_rows, "document_id")
    pages = {}
    for document_id, source in sources.items():
        _checked_file(root, source.get("local_pdf"), source.get("sha256"))
        page_file = _checked_file(root, source.get("page_text_path"), source.get("page_text_sha256"))
        rows = _rows(page_file)
        _require(bool(rows), f"no pages: {document_id}")
        for number, page in enumerate(rows, 1):
            _require(page.get("document_id") == document_id and type(page.get("page")) is int
                     and page["page"] == number and isinstance(page.get("text"), str),
                     f"invalid or nonconsecutive page: {document_id}:{number}")
            pages[document_id, number] = page

    spans = _unique(_rows(paths["source_spans.jsonl"]), "span_id")
    units = _unique(_rows(paths["evidence_units.jsonl"]), "evidence_unit_id")
    _require(bool(spans) and bool(units), "empty corpus")
    schema = json.loads((root / "schemas/source_span.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    intervals = defaultdict(list)
    for span_id, span in spans.items():
        errors = list(validator.iter_errors(span))
        _require(not errors, f"invalid source span {span_id}: {errors[0].message if errors else ''}")
        key = span["document_id"], span["pdf_page"]
        _require(key in pages, f"unknown source page: {span_id}")
        page, source = pages[key], sources[span["document_id"]]
        text, start, end = page["text"], span["start_char"], span["end_char"]
        _require(0 <= start < end <= len(text) and text[start:end] == span["exact"],
                 f"source span exact mismatch: {span_id}")
        _require(span["page_text_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest(),
                 f"page text hash mismatch: {span_id}")
        _require(span["document_sha256"] == source["sha256"]
                 and span["local_pdf"] == source["local_pdf"]
                 and span["page_text_path"] == source["page_text_path"]
                 and span["source_type"] == page.get("document_type")
                 and span["source_url"] == page.get("source_url"), f"span source mismatch: {span_id}")
        _require(text[:start].endswith(span["prefix"]) and text[end:].startswith(span["suffix"]),
                 f"span context mismatch: {span_id}")
        intervals[key].append((start, end))

    used_spans = set()
    for unit_id, unit in units.items():
        for field in ("document_id", "source_type", "section", "title", "unit_kind", "text", "retrieval_text"):
            _require(isinstance(unit.get(field), str) and bool(unit[field].strip()), f"{unit_id}: missing {field}")
        ids = unit.get("source_span_ids")
        _require(isinstance(ids, list) and bool(ids) and all(isinstance(value, str) for value in ids),
                 f"{unit_id}: missing source spans")
        _require(len(ids) == len(set(ids)) and all(value in spans for value in ids),
                 f"{unit_id}: duplicate or unknown source span")
        selected = [spans[value] for value in ids]
        _require(all(span["document_id"] == unit["document_id"] and span["source_type"] == unit["source_type"]
                     for span in selected), f"{unit_id}: mixed sources")
        positions = [(span["pdf_page"], span["start_char"], span["end_char"]) for span in selected]
        _require(positions == sorted(positions), f"{unit_id}: source spans out of order")
        _require(unit["text"] == "\n".join(span["exact"] for span in selected), f"{unit_id}: text mismatch")
        used_spans.update(ids)
    _require(used_spans == spans.keys(), "orphan source spans")

    relations = _rows(paths["relations.jsonl"])
    seen_relations = set()
    for relation in relations:
        _require(all(isinstance(relation.get(field), str) and relation[field] for field in
                     ("from_id", "to_id", "kind", "review_status")) and "trigger" in relation,
                 "incomplete relation")
        _require(relation["from_id"] in units and relation["to_id"] in units, "unknown relation endpoint")
        basis = relation.get("basis_span_ids")
        _require(isinstance(basis, list) and bool(basis)
                 and all(isinstance(value, str) and value in spans for value in basis), "unknown relation basis")
        key = (relation["from_id"], relation["to_id"], relation["kind"], json.dumps(relation["trigger"], sort_keys=True))
        _require(key not in seen_relations, "duplicate relation")
        seen_relations.add(key)

    coverage = _rows(paths["coverage.jsonl"])
    covered_pages = set()
    for row in coverage:
        key = row.get("document_id"), row.get("pdf_page")
        _require(isinstance(key[0], str) and type(key[1]) is int and key in pages
                 and key not in covered_pages, "unknown or duplicate coverage page")
        covered_pages.add(key)
        text = pages[key]["text"]
        actual, included, excluded = (bytearray(len(text)) for _ in range(3))
        for start, end in intervals[key]:
            _mark(actual, start, end)
        _require(isinstance(row.get("included"), list) and isinstance(row.get("excluded"), list),
                 f"missing coverage intervals: {key}")
        for interval in row["included"]:
            _require(isinstance(interval, list) and len(interval) == 2, f"invalid included interval: {key}")
            _mark(included, *interval)
        _require(included == actual, f"included coverage differs from source spans: {key}")
        for interval in row["excluded"]:
            _require(isinstance(interval, dict) and isinstance(interval.get("reason"), str)
                     and bool(interval["reason"].strip()), f"missing exclusion reason: {key}")
            _mark(excluded, interval.get("start_char"), interval.get("end_char"))
        # 数表大多整页排除；前置校验已确认区间，直接按字节检查即可。
        if 1 not in included and 0 not in excluded:
            continue
        for offset, char in enumerate(text):
            if not char.isspace():
                _require(not (included[offset] and excluded[offset]), f"included/excluded overlap: {key}:{offset}")
                _require(bool(included[offset] or excluded[offset]), f"uncovered non-whitespace: {key}:{offset}")
    _require(covered_pages == pages.keys(), "missing coverage pages")
    return {
        "counts": {"documents": len(sources), "pages": len(pages), "evidence_units": len(units),
                   "source_spans": len(spans), "relations": len(relations)},
        "uncovered_non_whitespace": 0,
        "semantic_review": "not_run",
    }
