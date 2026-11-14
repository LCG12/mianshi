"""Read-only build snapshot loading. No design fixture is published implicitly."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .errors import AppError


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _checked_bytes(base: Path, name: str, expected: str) -> bytes:
    path = (base / name).resolve()
    if (not path.is_relative_to(base) or not path.is_file()
            or not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected)):
        raise ValueError("file location or hash")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("file integrity")
    return content


@dataclass(frozen=True)
class BuildSnapshot:
    build_id: str
    corpus_version: str
    index_versions: Mapping[str, str]
    build_dir: Path
    manifest: Mapping[str, Any]
    corpus_dir: Path
    corpus_manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index_versions", _freeze(dict(self.index_versions)))
        object.__setattr__(self, "manifest", _freeze(dict(self.manifest)))
        object.__setattr__(self, "corpus_manifest", _freeze(dict(self.corpus_manifest)))


class PublishedStore:
    """Verify a publication manifest and file integrity, not contract semantics."""

    def __init__(self, project_root: Path, build_dir: Path, build_id: str | None):
        self.project_root = project_root.resolve()
        self.build_dir = build_dir.resolve()
        self.build_id = build_id
        self._snapshot: BuildSnapshot | None = None

    def snapshot(self) -> BuildSnapshot:
        if self._snapshot is not None:
            return self._snapshot
        if not self.build_id:
            raise AppError("build_unpublished", "尚未发布正式证据库与索引。")
        manifest_path = self.build_dir / "manifest.json"
        if not manifest_path.is_file():
            raise AppError("build_manifest_missing", "指定构建没有发布清单。")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest["schema_version"] != "build-manifest/1.0"
                    or manifest["status"] != "published"
                    or manifest["build_id"] != self.build_id
                    or not isinstance(manifest["corpus_version"], str)
                    or not manifest["corpus_version"]
                    or not isinstance(manifest["index_versions"], dict)
                    or not manifest["index_versions"]
                    or not all(isinstance(v, str) and v for v in manifest["index_versions"].values())):
                raise ValueError("manifest identity")
            reference = manifest["corpus"]
            corpus_dir = (self.project_root / reference["path"]).resolve()
            if Path(reference["path"]).is_absolute() or not corpus_dir.is_relative_to(self.project_root):
                raise ValueError("corpus location")
            corpus = json.loads(_checked_bytes(corpus_dir, "manifest.json", reference["manifest_sha256"]))
            if (corpus["schema_version"] != "build-manifest/1.0" or corpus["status"] != "corpus_ready"
                    or corpus["index_versions"] != {} or corpus["build_id"] != manifest["corpus_build_id"]
                    or corpus["corpus_version"] != manifest["corpus_version"]
                    or corpus["sources"] != manifest["sources"]):
                raise ValueError("corpus identity")
            for directory, files, required in (
                (self.build_dir, manifest["files"], {"lexical.json", "vectors.npy"}),
                (corpus_dir, corpus["files"], {"evidence_units.jsonl", "source_spans.jsonl", "relations.jsonl", "coverage.jsonl"}),
            ):
                if not isinstance(files, dict) or not required <= files.keys():
                    raise ValueError("manifest files")
                for name, expected in files.items():
                    _checked_bytes(directory, name, expected)
            sources = corpus["sources"]
            if not isinstance(sources, list) or not sources:
                raise ValueError("source list")
            seen = set()
            for source in sources:
                identity = (source["document_id"], source["sha256"])
                if (identity in seen or not re.fullmatch(r"[A-Za-z0-9_-]+", identity[0])
                        or not re.fullmatch(r"[a-f0-9]{64}", identity[1])):
                    raise ValueError("source identity")
                seen.add(identity)
                if Path(source["local_pdf"]).suffix.lower() != ".pdf":
                    raise ValueError("source file")
                _checked_bytes(self.project_root, source["local_pdf"], source["sha256"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            raise AppError("invalid_build", "构建清单、产物或原文版本校验失败。") from None
        self._snapshot = BuildSnapshot(self.build_id, manifest["corpus_version"],
                                       dict(manifest["index_versions"]), self.build_dir, manifest,
                                       corpus_dir, corpus)
        return self._snapshot

    def readiness(self) -> dict[str, Any]:
        try:
            snapshot = self.snapshot()
        except AppError as exc:
            return {"ready": False, "build_id": self.build_id, "reasons": [{"code": exc.code, "message": exc.message}]}
        return {"ready": True, "build_id": snapshot.build_id, "reasons": [],
                "validation_scope": "manifest_and_file_integrity_only"}

    def resolve_source(self, document_id: str, sha256: str) -> Path | None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", document_id) or not re.fullmatch(r"[a-f0-9]{64}", sha256):
            return None
        try:
            snapshot = self.snapshot()
        except AppError:
            return None
        for source in snapshot.manifest.get("sources", []):
            if (source["document_id"], source["sha256"]) == (document_id, sha256):
                path = (self.project_root / source["local_pdf"]).resolve()
                try:
                    _checked_bytes(self.project_root, source["local_pdf"], sha256)
                    return path
                except (OSError, ValueError):
                    return None
        return None
