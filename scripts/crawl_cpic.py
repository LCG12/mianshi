#!/usr/bin/env python3
"""Download the official PDFs listed in data/raw/sources.json; no third-party packages.

Run from any directory: python3 scripts/crawl_cpic.py
Use --only ID[,ID] to select documents or --refresh to replace cached PDFs.
PDF text extraction uses the optional system command pdftotext (Poppler).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "CPIC-Interview-DataCollector/1.0 (public product documents)"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def check_official_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        host == "cpic.com.cn" or host.endswith(".cpic.com.cn")
    ):
        raise ValueError(f"Only HTTPS cpic.com.cn sources are accepted: {url}")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError(f"Unexpected URL credentials or port: {url}")


class OfficialRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_official_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def local_path(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError(f"local_pdf must be relative: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Output escapes project directory: {relative}")
    return path


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def is_pdf(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return stream.read(5) == b"%PDF-"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path, timeout: float, retries: int) -> dict:
    check_official_url(url)
    opener = urllib.request.build_opener(OfficialRedirectHandler())
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    last_error = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf"}
            )
            with opener.open(request, timeout=timeout) as response:
                check_official_url(response.geturl())
                expected_length = response.headers.get("Content-Length")
                with temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
                if expected_length is not None and temporary.stat().st_size != int(expected_length):
                    raise ValueError("Incomplete response: Content-Length does not match bytes received")
                if not is_pdf(temporary):
                    raise ValueError("Response does not start with %PDF-; refusing to save as PDF")
                metadata = {
                    "fetched_at": now(),
                    "final_url": response.geturl(),
                    "http_status": response.status,
                    "content_type": response.headers.get("Content-Type"),
                    "attempts": attempt + 1,
                }
            temporary.replace(target)
            return metadata
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
        finally:
            temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Download failed after {retries + 1} attempt(s): {last_error}")


def extract_text(root: Path, document: dict, pdf: Path) -> dict:
    executable = shutil.which("pdftotext")
    if not executable:
        return {"status": "not_available", "reason": "pdftotext is not installed"}
    completed = subprocess.run(
        [executable, "-layout", "-enc", "UTF-8", str(pdf), "-"],
        capture_output=True, check=True, timeout=180,
    )
    text = completed.stdout.decode("utf-8")
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    if not pages or not any(page.strip() for page in pages):
        raise ValueError("No extractable PDF text; OCR or manual review is required")
    destination = root / "data" / "pages"
    pages_path = destination / f"{document['id']}.pages.jsonl"
    records = [{
        "document_id": document["id"],
        "page": index,
        "source_url": document["url"],
        "product_name": document["product_name"],
        "document_type": document["document_type"],
        "extraction_method": "pdftotext -layout -enc UTF-8",
        "text": page.strip("\n"),
    } for index, page in enumerate(pages, 1)]
    atomic_write(pages_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    expected = document.get("expected_pages")
    matches = expected is None or len(pages) == expected
    return {
        "status": "ok" if matches else "page_count_mismatch",
        "pages": len(pages),
        "expected_pages": expected,
        "pages_path": str(pages_path.relative_to(root)),
        "warnings": completed.stderr.decode("utf-8", errors="replace").strip(),
    }


def process_document(root: Path, document: dict, args, previous: dict) -> dict:
    record = {**document, "checked_at": now(), "status": "failed", "download_status": "failed"}
    try:
        identifier = document["id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError("Document id must contain only ASCII letters, digits, '-' and '_'")
        check_official_url(document["url"])
        path = local_path(root, document["local_pdf"])
        if path.suffix.lower() != ".pdf":
            raise ValueError("local_pdf must end in .pdf")
        if is_pdf(path) and not args.refresh:
            record["download_status"] = "cached"
            # A pre-existing file is not evidence that it was fetched in this run.
            same_file = previous.get("sha256") == file_digest(path)
            record["fetched_at"] = previous.get("fetched_at") if same_file else None
            if same_file:
                for field in ("final_url", "http_status", "content_type"):
                    if field in previous:
                        record[field] = previous[field]
        else:
            record.update(download(document["url"], path, args.timeout, args.retries))
            record["download_status"] = "downloaded"
        record["sha256"] = file_digest(path)
        record["bytes"] = path.stat().st_size
        record["extraction"] = extract_text(root, document, path)
        if record["extraction"]["status"] == "page_count_mismatch":
            record["status"] = "partial"
            record["error"] = "Extracted page count differs from expected_pages; review required"
        else:
            record["status"] = "ok"
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
        if record["download_status"] in ("downloaded", "cached"):
            record["status"] = "partial"
    return record


def main(argv=None, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=root / "data" / "raw" / "sources.json")
    parser.add_argument("--only", action="append", default=[], metavar="ID[,ID]",
                        help="Select document ids; option may be repeated")
    parser.add_argument("--refresh", action="store_true", help="Download again even if a PDF exists")
    parser.add_argument("--timeout", type=float, default=30, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retries after the first HTTP attempt")
    parser.add_argument("--delay", type=float, default=1, help="Seconds between documents")
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.retries < 0 or args.delay < 0:
        parser.error("timeout must be positive; retries and delay must be nonnegative")
    try:
        source_data = json.loads(args.sources.read_text(encoding="utf-8"))
        documents = source_data["documents"]
        ids = [document["id"] for document in documents]
        if not documents or len(ids) != len(set(ids)):
            raise ValueError("documents must be nonempty and have unique ids")
        selected = {item.strip() for group in args.only for item in group.split(",") if item.strip()}
        if selected - set(ids):
            raise ValueError(f"Unknown document ids: {sorted(selected - set(ids))}")
        manifest_path = root / "data" / "raw" / "download_manifest.json"
        existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        by_id = {item["id"]: item for item in existing.get("documents", [])}
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    pending = [document for document in documents if not selected or document["id"] in selected]
    failed = 0
    started_at = now()
    for index, document in enumerate(pending):
        if index:
            time.sleep(args.delay)
        record = process_document(root, document, args, by_id.get(document["id"], {}))
        by_id[document["id"]] = record
        failed += record["status"] != "ok"
        manifest = {
            "schema_version": 1,
            "run_started_at": started_at,
            "updated_at": now(),
            "source_manifest": str(args.sources),
            "selected_ids": [item["id"] for item in pending],
            "documents": [by_id[identifier] for identifier in ids if identifier in by_id],
        }
        atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print(f"{record['id']}: {record['status']} ({record['download_status']})", flush=True)
        if "error" in record:
            print(f"  {record['error']}", file=sys.stderr, flush=True)
    print(f"Processed {len(pending)} document(s); {failed} failed or require review.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
