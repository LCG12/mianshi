"""小语料索引：中文词频 + 归一化向量，校验后发布新版本。"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import jieba
import numpy as np

from .corpus_validation import validate_corpus
from .models.factory import create_auxiliary_client
from .models.contracts import ServiceUnavailable
from .store import PublishedStore

TOKENIZER_VERSION = "jieba-search/1.0"
DIMENSIONS = 1024


def _json(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@lru_cache(maxsize=4)
def _tokenizer(terms: tuple[str, ...]):
    jieba.setLogLevel(logging.WARNING)
    tokenizer = jieba.Tokenizer()
    tokenizer.initialize()
    for term in terms:
        tokenizer.add_word(term, freq=100000)
    return tokenizer


def tokenize(text: str, terms=()) -> Counter:
    """同一词在同一位置只计一次；保留否定词、数字、百分比和条款编号。"""
    text = text.lower()
    positions = set(_tokenizer(tuple(terms)).tokenize(text, mode="search", HMM=False))
    positions.update((m.group(), m.start(), m.end()) for m in re.finditer(r"\d+(?:\.\d+)*(?:[%％])?", text))
    positions.update((m.group(), m.start(), m.end()) for m in re.finditer(r"不含|不|未|内|后|含|且|或", text))
    return Counter(word for word, _, _ in positions if re.search(r"\w", word))


def load_index(snapshot):
    """载入登记且哈希一致的索引，再检查词频、ID及矩阵的对应关系。"""
    contents = {}
    for directory, files, name in (
        (snapshot.build_dir, snapshot.manifest["files"], "lexical.json"),
        (snapshot.build_dir, snapshot.manifest["files"], "vectors.npy"),
        (snapshot.corpus_dir, snapshot.corpus_manifest["files"], "evidence_units.jsonl"),
    ):
        if name not in files:
            raise ValueError("index files missing from manifest")
        contents[name] = (directory / name).read_bytes()
        if _sha(contents[name]) != files[name]:
            raise ValueError(f"index file hash mismatch: {name}")
    lexical = json.loads(contents["lexical.json"])
    rows = [json.loads(line) for line in contents["evidence_units.jsonl"].splitlines() if line.strip()]
    units = sorted((u for u in rows if u.get("retrievable") is True),
                   key=lambda u: u["evidence_unit_id"])
    ids = [u["evidence_unit_id"] for u in units]
    if (not ids or len(set(ids)) != len(ids) or lexical.get("unit_ids") != ids
            or lexical.get("jieba_version") != jieba.__version__
            or lexical.get("tokenizer_version") != TOKENIZER_VERSION):
        raise ValueError("index identity or tokenizer version mismatch")
    terms, counts = lexical.get("terms"), lexical.get("token_counts")
    if (not isinstance(terms, list) or any(not isinstance(x, str) or not x for x in terms)
            or terms != sorted(set(terms)) or not isinstance(counts, list) or len(counts) != len(ids)):
        raise ValueError("invalid lexical index")
    for unit, count in zip(units, counts):
        if (not isinstance(count, dict) or not count
                or any(not isinstance(k, str) or type(v) is not int or v < 1 for k, v in count.items())
                or count != tokenize(unit["retrieval_text"], terms)):
            raise ValueError("token counts differ from indexed source")
    vectors = np.load(BytesIO(contents["vectors.npy"]), allow_pickle=False)
    dimensions = snapshot.manifest["retrieval"]["dimensions"]
    if (dimensions != DIMENSIONS or vectors.dtype != np.float32 or vectors.shape != (len(ids), dimensions)
            or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
        raise ValueError("invalid embedding matrix")
    vectors.setflags(write=False)
    return units, vectors, lexical


async def build_index(settings, corpus_dir: Path) -> dict:
    root, corpus_dir = settings.project_root.resolve(), corpus_dir.resolve()
    if not corpus_dir.is_relative_to(root):
        raise ValueError("语料必须位于项目目录内")
    validation = validate_corpus(root, corpus_dir)
    corpus_bytes = (corpus_dir / "manifest.json").read_bytes()
    corpus = json.loads(corpus_bytes)
    if corpus.get("unresolved_references"):
        raise ValueError("语料仍有未解析引用")
    units = sorted((u for u in _rows(corpus_dir / "evidence_units.jsonl") if u.get("retrievable") is True),
                   key=lambda u: u["evidence_unit_id"])
    if not units:
        raise ValueError("没有可检索证据")
    terms = sorted({part for u in units for part in re.split(r"[（）()：、，\s]+", u["title"])
                    if 2 <= len(part) <= 16 and re.fullmatch(r"[\u4e00-\u9fff]+", part)})
    lexical = {"unit_ids": [u["evidence_unit_id"] for u in units], "terms": terms,
               "token_counts": [dict(tokenize(u["retrieval_text"], terms)) for u in units],
               "jieba_version": jieba.__version__, "tokenizer_version": TOKENIZER_VERSION}
    service = settings.model_roles.get("embedding", {}).get("service", "embedding")
    raw_service = settings.model_services.get(service, {})
    if not raw_service.get("enabled"):
        raise ServiceUnavailable(service)
    client = create_auxiliary_client(raw_service, service)
    try:
        retrieval = {"embedding_model": client.config.model, "dimensions": DIMENSIONS,
                     "embedding_profile": "local_embedding/1.0", "document_instruction": "none",
                     "query_instruction_owner": "model_server", "normalization": "l2",
                     "k1": 1.2, "b": 0.75, "rrf_k": 60, "per_channel_k": 20,
                     "candidate_k": 40, "top_k": 8, "rerank_default": False}
        reference = {"path": str(corpus_dir.relative_to(root)), "manifest_sha256": _sha(corpus_bytes)}
        inputs = {"corpus": reference, "builder_sha256": _sha(Path(__file__).read_bytes()),
                  "lexical_sha256": _sha(_json(lexical)), "retrieval": retrieval}
        for name in ("deploy/models.lock.json", "deploy/model_server.py", "insurance_qa/retrieval.py"):
            if (root / name).is_file():
                inputs[name] = _sha((root / name).read_bytes())
        suffix = _sha(_json(inputs))[:16]
        build_id = "index-" + suffix
        parent = root / "data/index"
        directory = parent / build_id

        def checked(path):
            snapshot = PublishedStore(root, path, build_id).snapshot()
            if dict(snapshot.manifest["index_inputs"]) != inputs:
                # Nested frozen containers compare by value with the original dictionaries.
                raise ValueError("existing index inputs differ")
            load_index(snapshot)
            return snapshot

        reused = directory.exists()
        if reused:
            snapshot = checked(directory)
        else:
            vectors = np.asarray(await client.embed([u["retrieval_text"] for u in units], purpose="document"), dtype=np.float32)
            if vectors.shape != (len(units), DIMENSIONS) or not np.isfinite(vectors).all():
                raise ValueError("invalid document embeddings")
            norms = np.linalg.norm(vectors.astype(np.float64), axis=1, keepdims=True)
            if not np.isfinite(norms).all() or np.any(norms <= 0):
                raise ValueError("zero or invalid document embedding")
            vectors = (vectors / norms).astype(np.float32)
            parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".index-", dir=parent) as temporary:
                staging = Path(temporary) / "build"
                staging.mkdir()
                (staging / "lexical.json").write_bytes(_json(lexical))
                np.save(staging / "vectors.npy", vectors, allow_pickle=False)
                files = {p.name: _sha(p.read_bytes()) for p in staging.iterdir()}
                manifest = {**corpus, "build_id": build_id, "status": "published", "files": files,
                            "corpus": reference,
                            "corpus_build_id": corpus["build_id"], "retrieval": retrieval, "index_inputs": inputs,
                            "index_versions": {name: name + "-" + suffix for name in ("bm25", "dense", "hybrid")}}
                (staging / "manifest.json").write_bytes(_json(manifest))
                checked(staging)
                staging.rename(directory)
            snapshot = checked(directory)
        result = {"status": "ok", "build_id": build_id, "build_directory": str(directory.relative_to(root)),
                  "corpus_build_id": corpus["build_id"], "indexed_units": len(units), "dimensions": DIMENSIONS,
                  "index_versions": dict(snapshot.index_versions), "embedding_model": retrieval["embedding_model"],
                  "counts": {"indexed_units": len(units), "dimensions": DIMENSIONS,
                             "by_source": dict(Counter(u["source_type"] for u in units))},
                  "corpus_validation": validation, "reused": reused, "index_built": True, "runtime_ready": False}
        return result
    finally:
        await client.aclose()
