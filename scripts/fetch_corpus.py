from __future__ import annotations

import argparse
import hashlib
import re
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


ALLOWED_FILES = {"train.txt", "validation.txt", "test.txt", "corpus_manifest.json"}
MAX_ARCHIVE_BYTES = 5 * 1024**3
MAX_EXTRACTED_BYTES = 4 * 1024**3


def _safe_member_name(name: str) -> str | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member path: {name}")
    if path.name in ALLOWED_FILES:
        return path.name
    return None


def _write_member(name: str, source, destination: Path) -> None:
    target_name = _safe_member_name(name)
    if target_name is None:
        return
    target = destination / target_name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _extract(archive_path: Path, destination: Path) -> set[str]:
    written: set[str] = set()
    extracted_bytes = 0
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                name = _safe_member_name(info.filename)
                if not name or info.is_dir():
                    continue
                if name in written:
                    raise ValueError(f"Duplicate corpus archive member: {name}")
                extracted_bytes += info.file_size
                if extracted_bytes > MAX_EXTRACTED_BYTES:
                    raise ValueError(f"Extracted corpus exceeds the {MAX_EXTRACTED_BYTES} byte safety limit")
                with archive.open(info) as member:
                    _write_member(info.filename, member, destination)
                written.add(name)
        return written

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                name = _safe_member_name(member.name)
                if not name:
                    continue
                if name in written:
                    raise ValueError(f"Duplicate corpus archive member: {name}")
                if not member.isfile():
                    raise ValueError(f"Refusing non-regular archive member: {member.name}")
                extracted_bytes += member.size
                if extracted_bytes > MAX_EXTRACTED_BYTES:
                    raise ValueError(f"Extracted corpus exceeds the {MAX_EXTRACTED_BYTES} byte safety limit")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Could not read archive member: {member.name}")
                with stream:
                    _write_member(member.name, stream, destination)
                written.add(name)
        return written
    raise ValueError("Dataset URL must point to a ZIP or tar archive")


def download_corpus(url: str, expected_sha256: str, destination: Path) -> dict[str, object]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Dataset must be a direct HTTPS URL without embedded credentials")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("Expected SHA-256 must be 64 hexadecimal characters")

    destination.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    request = urllib.request.Request(url, headers={"User-Agent": "Monster-1-corpus-fetch/1.0"})
    with tempfile.NamedTemporaryFile(prefix="monster1-corpus-", suffix=".archive") as temporary:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError(f"Dataset download returned HTTP {response.status}")
            final_url = urllib.parse.urlparse(response.geturl())
            if final_url.scheme != "https" or not final_url.hostname:
                raise ValueError("Dataset download redirected away from HTTPS")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_ARCHIVE_BYTES:
                    raise ValueError(f"Dataset archive exceeds the {MAX_ARCHIVE_BYTES} byte safety limit")
                digest.update(chunk)
                temporary.write(chunk)
        temporary.flush()
        actual_sha256 = digest.hexdigest()
        if actual_sha256.lower() != expected_sha256.lower():
            raise ValueError(f"Archive SHA-256 mismatch: got {actual_sha256}")
        written = _extract(Path(temporary.name), destination)

    missing = sorted(ALLOWED_FILES - written)
    if missing:
        raise ValueError("Archive is missing required files: " + ", ".join(missing))
    return {"archive_bytes": size, "archive_sha256": actual_sha256, "files": sorted(written)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a verified static corpus archive; no API calls")
    parser.add_argument("--url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--destination", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(download_corpus(args.url, args.sha256, args.destination))


if __name__ == "__main__":
    main()
