from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO = "AIgenius1/persian-pretraining-dataset"
REVISION = "a6906d5585c981b74442f4be186ab9098e26b363"
EXPECTED_SHARDS = 46
EXPECTED_BYTES = 2_241_208_057
DATASET_URL = f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}"
TREE_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true&expand=true&limit=100"
UPSTREAM_DUMP_URL = "https://dumps.wikimedia.org/fawiki/20261001/"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
WIKIPEDIA_RIGHTS_URL = "https://en.wikipedia.org/wiki/Wikipedia:Copyrights"

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "wikipedia_source"
SHARD_MANIFEST = SOURCE_DIR / "shard_manifest.json"
TRAIN = ROOT / "data" / "train.txt"
VALIDATION = ROOT / "data" / "validation.txt"
TEST = ROOT / "data" / "test.txt"
CORPUS_MANIFEST = ROOT / "data" / "corpus_manifest.json"
ATTRIBUTION = ROOT / "data" / "article_attribution.tsv"
PRIVACY_REPORT = ROOT / "data" / "privacy_scan.json"

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩"
ASCII_DIGITS = "01234567890123456789"
DIGIT_TRANSLATION = str.maketrans(PERSIAN_DIGITS, ASCII_DIGITS)

PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "email",
        re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])", re.I),
    ),
    (
        "iran_iban",
        re.compile(r"(?<![A-Z])IR\d{24}(?!\d)", re.I),
    ),
    (
        "labelled_card_number",
        re.compile(r"(?:شماره\s*کارت|کارت\s*بانکی|card(?:\s*(?:number|no))?)\s*[:：]?\s*\d[\d -]{12,23}\d", re.I),
    ),
    (
        "labelled_national_id",
        re.compile(r"(?:کد\s*ملی|national[\s-]*id)\s*[:：]?\s*[\d -]{9,12}\d", re.I),
    ),
    (
        "labelled_postal_code",
        re.compile(r"(?:کد\s*پستی|postal[\s-]*code)\s*[:：]?\s*\d{10}", re.I),
    ),
    (
        "labelled_contact_number",
        re.compile(
            r"(?:شماره[\s‌]*(?:تلفن|تماس|همراه|موبایل)|تلفن(?:\s+همراه)?|موبایل|شماره\s+تماس|"
            r"phone|telephone|mobile|fax|tel)\s*[:：]?\s*[+\d() .-]{7,30}",
            re.I,
        ),
    ),
    (
        "iran_mobile_number",
        re.compile(r"(?<!\d)(?:(?:\+|00)?98[- ]?)?0?9(?:[- ]?\d){9}(?!\d)"),
    ),
)


def _request(url: str, *, timeout: int = 90) -> urllib.request.addinfourl:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Monster-1-corpus-preparation/1.0 (reproducible research; contact: repo owner)"},
    )
    response = urllib.request.urlopen(request, timeout=timeout)
    if not response.geturl().startswith("https://"):
        response.close()
        raise RuntimeError("Refusing a non-HTTPS source or redirect")
    return response


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_shard_metadata() -> list[dict[str, Any]]:
    with _request(TREE_URL) as response:
        listing = json.load(response)
    shards = sorted(
        (row for row in listing if re.fullmatch(r"persian_pretraining_articles_\d{4}\.jsonl", row.get("path", ""))),
        key=lambda row: row["path"],
    )
    if len(shards) != EXPECTED_SHARDS or sum(int(row.get("size", 0)) for row in shards) != EXPECTED_BYTES:
        raise RuntimeError(
            "Pinned Hugging Face tree differs from the reviewed inventory: "
            f"{len(shards)} shards, {sum(int(row.get('size', 0)) for row in shards)} bytes"
        )
    normalized = []
    for row in shards:
        lfs = row.get("lfs") or {}
        digest = str(lfs.get("oid", "")).removeprefix("sha256:").lower()
        size = int(row["size"])
        if len(digest) != 64 or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"No valid SHA-256 in source metadata for {row['path']}")
        if int(lfs.get("size", size)) != size:
            raise RuntimeError(f"LFS size mismatch in source metadata for {row['path']}")
        normalized.append({"path": row["path"], "size": size, "sha256": digest})
    return normalized


def _download_one(item: dict[str, Any]) -> dict[str, Any]:
    path = SOURCE_DIR / item["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(item["size"])
    expected_sha = item["sha256"]
    if path.is_file() and path.stat().st_size == expected_size and sha256_file(path) == expected_sha:
        return {**item, "status": "verified_existing"}

    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{item['path']}?download=true"
    last_error: Exception | None = None
    for attempt in range(1, 4):
        partial = path.with_suffix(path.suffix + f".part-{os.getpid()}")
        digest = hashlib.sha256()
        size = 0
        try:
            with _request(url, timeout=180) as response, partial.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if size != expected_size or digest.hexdigest() != expected_sha:
                raise RuntimeError(
                    f"Integrity failure for {item['path']}: {size} bytes / {digest.hexdigest()}"
                )
            os.replace(partial, path)
            return {**item, "status": "downloaded_verified"}
        except Exception as exc:
            last_error = exc
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt < 3:
                time.sleep(attempt * 2)
    raise RuntimeError(f"Failed to download {item['path']} after retries: {last_error}")


def download_corpus(workers: int = 3) -> dict[str, Any]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    items = fetch_shard_metadata()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, item): item for item in items}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['status']}: {result['path']} ({result['size']:,} bytes)", flush=True)
    results.sort(key=lambda row: row["path"])
    manifest = {
        "repository": REPO,
        "revision": REVISION,
        "dataset_url": DATASET_URL,
        "upstream_dump_url": UPSTREAM_DUMP_URL,
        "license": "CC BY-SA 4.0 as stated by the dataset publisher; Wikipedia reuse terms also apply",
        "license_url": LICENSE_URL,
        "expected_shards": EXPECTED_SHARDS,
        "expected_bytes": EXPECTED_BYTES,
        "files": results,
    }
    SHARD_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_downloaded_shards() -> dict[str, Any]:
    if not SHARD_MANIFEST.is_file():
        raise FileNotFoundError(f"Missing source shard manifest: {SHARD_MANIFEST}; run download first")
    manifest = json.loads(SHARD_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("repository") != REPO or manifest.get("revision") != REVISION:
        raise RuntimeError("Source shard manifest does not match the pinned repository revision")
    if len(manifest.get("files", [])) != EXPECTED_SHARDS:
        raise RuntimeError("Source shard manifest has the wrong number of files")
    for item in manifest["files"]:
        path = SOURCE_DIR / item["path"]
        if not path.is_file() or path.stat().st_size != int(item["size"]):
            raise RuntimeError(f"Missing or wrong-sized shard: {path}")
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for source shard: {path}")
    return manifest


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def _matches(text: str) -> list[tuple[int, int, str]]:
    scan_text = text.translate(DIGIT_TRANSLATION)
    found: list[tuple[int, int, str]] = []
    for label, pattern in PII_PATTERNS:
        found.extend((match.start(), match.end(), label) for match in pattern.finditer(scan_text))
    return found


def redact_sensitive(text: str) -> tuple[str, dict[str, int]]:
    found = _matches(text)
    counts: dict[str, int] = {}
    for _, _, label in found:
        counts[label] = counts.get(label, 0) + 1
    if not found:
        return text, counts

    spans = sorted((start, end) for start, end, _ in found)
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    for start, end in reversed(merged):
        text = text[:start] + "[REDACTED-PERSONAL-INFO]" + text[end:]
    return text, counts


def _split_for(article_url: str) -> str:
    bucket = int(hashlib.sha256(article_url.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def _hash_file(path: Path) -> str:
    return sha256_file(path)


def prepare_corpus() -> dict[str, Any]:
    source = verify_downloaded_shards()
    for path in (TRAIN, VALIDATION, TEST, ATTRIBUTION):
        path.parent.mkdir(parents=True, exist_ok=True)

    split_paths = {"train": TRAIN, "validation": VALIDATION, "test": TEST}
    stats: dict[str, Any] = {
        "records_read": 0,
        "records_written": 0,
        "skipped_empty_or_short": 0,
        "skipped_missing_attribution": 0,
        "duplicate_urls": 0,
        "duplicate_texts": 0,
        "documents_per_split": {key: 0 for key in split_paths},
        "redaction_detections_by_pattern": {},
        "residual_detections_by_pattern": {},
    }
    seen_urls: set[str] = set()
    seen_text_hashes: set[str] = set()

    with (
        TRAIN.open("w", encoding="utf-8", newline="\n") as train_stream,
        VALIDATION.open("w", encoding="utf-8", newline="\n") as validation_stream,
        TEST.open("w", encoding="utf-8", newline="\n") as test_stream,
        ATTRIBUTION.open("w", encoding="utf-8", newline="\n") as attribution_stream,
    ):
        streams = {"train": train_stream, "validation": validation_stream, "test": test_stream}
        attribution_stream.write("split\tarticle_url\ttitle\n")
        for item in source["files"]:
            path = SOURCE_DIR / item["path"]
            with path.open("r", encoding="utf-8", errors="strict") as shard:
                for line_number, line in enumerate(shard, start=1):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"Invalid JSON in {path.name}:{line_number}: {exc}") from exc
                    stats["records_read"] += 1
                    body = row.get("text")
                    title = normalize_text(row.get("title", "")) if isinstance(row.get("title", ""), str) else ""
                    url = row.get("article_url")
                    if not isinstance(body, str) or not isinstance(url, str) or not url.startswith("https://fa.wikipedia.org/wiki/"):
                        stats["skipped_missing_attribution"] += 1
                        continue
                    body = normalize_text(body)
                    if len(body) < 100:
                        stats["skipped_empty_or_short"] += 1
                        continue
                    if url in seen_urls:
                        stats["duplicate_urls"] += 1
                        continue
                    seen_urls.add(url)
                    cleaned_title, title_detections = redact_sensitive(title)
                    cleaned_body, detections = redact_sensitive(body)
                    for label in title_detections.keys() | detections.keys():
                        count = title_detections.get(label, 0) + detections.get(label, 0)
                        stats["redaction_detections_by_pattern"][label] = stats["redaction_detections_by_pattern"].get(label, 0) + count
                    if len(cleaned_body) < 100:
                        stats["skipped_empty_or_short"] += 1
                        continue
                    body_digest = hashlib.sha256(cleaned_body.encode("utf-8")).hexdigest()
                    if body_digest in seen_text_hashes:
                        stats["duplicate_texts"] += 1
                        continue
                    seen_text_hashes.add(body_digest)
                    document = normalize_text((f"عنوان: {cleaned_title}; " if cleaned_title else "") + cleaned_body)
                    residual = _matches(document)
                    for _, _, label in residual:
                        stats["residual_detections_by_pattern"][label] = stats["residual_detections_by_pattern"].get(label, 0) + 1
                    split = _split_for(url)
                    streams[split].write(document + "\n")
                    attribution_stream.write(f"{split}\t{url}\t{cleaned_title.replace(chr(9), ' ')}\n")
                    stats["documents_per_split"][split] += 1
                    stats["records_written"] += 1

    if not all(stats["documents_per_split"].values()):
        raise RuntimeError(f"At least one document split is empty: {stats['documents_per_split']}")
    if stats["residual_detections_by_pattern"]:
        raise RuntimeError(f"Privacy scan found unredacted direct identifiers: {stats['residual_detections_by_pattern']}")

    shard_hashes = "\n".join(f"{item['path']}\t{item['sha256']}" for item in source["files"])
    shard_set_sha256 = hashlib.sha256(shard_hashes.encode("utf-8")).hexdigest()
    privacy = {
        "automated_review": "Scan normalized article text for email addresses, Iranian mobile numbers, Iranian IBANs, and clearly labelled card, national-ID, postal-code, and contact-number fields; redact matches before splitting.",
        "residual_matches_after_redaction": stats["residual_detections_by_pattern"],
        "redaction_detections_by_pattern": stats["redaction_detections_by_pattern"],
        "limitations": "This does not remove every form of personal or sensitive information. Public biographical facts and identifiers in unrecognized formats may remain; do not publish model weights or the derived corpus without a separate rights/privacy review.",
    }
    manifest = {
        "source_name": "Persian Wikipedia namespace-0 articles, via the pinned AIgenius1 clean packaging",
        "source_url": DATASET_URL,
        "upstream_source_url": UPSTREAM_DUMP_URL,
        "upstream_dump_date": "2026-10-01",
        "dataset_revision": REVISION,
        "dataset_shard_set_sha256": shard_set_sha256,
        "dataset_shard_count": EXPECTED_SHARDS,
        "dataset_bytes": EXPECTED_BYTES,
        "license": "CC BY-SA 4.0 as stated by the publisher; Wikipedia text reuse terms and attribution/share-alike obligations apply",
        "license_url": LICENSE_URL,
        "wikipedia_rights_url": WIKIPEDIA_RIGHTS_URL,
        "language": "fa",
        "collection_date": "2026-10-01",
        "rights_to_train": True,
        "privacy_reviewed": True,
        "attribution": "Wikipedia contributors. The article URL for each retained document is preserved in data/article_attribution.tsv. Preserve attribution and applicable ShareAlike terms when redistributing source text; legal review is required before publishing derived weights.",
        "sha256_train": _hash_file(TRAIN),
        "sha256_validation": _hash_file(VALIDATION),
        "sha256_test": _hash_file(TEST),
        "real_data": True,
        "synthetic": False,
        "is_synthetic": False,
        "pretrained_model_used": False,
        "external_model_api_used": False,
        "split_method": "Stable document-level split: SHA-256(article URL) modulo 100; 90% train, 5% validation, 5% test; duplicate cleaned bodies excluded globally.",
        "preprocessing": "NFC normalization; whitespace collapse; short documents under 100 characters skipped; only https://fa.wikipedia.org/wiki/ article URLs accepted; title is prepended; direct identifiers listed in privacy_scan.json redacted.",
        "privacy": privacy,
        "documents_per_split": stats["documents_per_split"],
        "source_records_read": stats["records_read"],
        "records_written": stats["records_written"],
    }
    CORPUS_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PRIVACY_REPORT.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"manifest": manifest, "privacy_scan": stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare a pinned Persian Wikipedia corpus")
    parser.add_argument("mode", choices=("all", "download", "prepare"), nargs="?", default="all")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.mode in ("all", "download"):
        download_corpus(workers=max(1, min(args.workers, 6)))
    if args.mode in ("all", "prepare"):
        result = prepare_corpus()
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
