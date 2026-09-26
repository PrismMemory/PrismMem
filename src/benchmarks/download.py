"""Fetch public benchmark data from its upstream repository, with provenance.

No files or network requests are made unless --execute is given. Examples:
  python -m benchmarks.download beam --output ./data/beam --dry-run
  python -m benchmarks.download beam --output ./data/beam --execute
  python -m benchmarks.download locomo --output ./data/locomo --revision COMMIT --execute

BEAM uses the official repository's already converted JSON files; it downloads
neither other splits nor the external project's code. Choose a data directory
outside your submission staging directory. No dotenv or model client is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


SOURCES = {
    "locomo": {
        "provider": "github", "repository": "snap-research/locomo",
        "documentation": "https://github.com/snap-research/LoCoMo",
        "files": [("data/locomo10.json", "locomo10.json")],
    },
    "longmemeval": {
        "provider": "huggingface", "repository": "xiaowu0162/longmemeval-cleaned",
        "documentation": "https://github.com/xiaowu0162/LongMemEval",
        "files": [("longmemeval_s_cleaned.json", "longmemeval_s_cleaned.json")],
    },
    "personamem": {
        "provider": "huggingface", "repository": "bowen-upenn/PersonaMem-v1",
        "documentation": "https://github.com/bowen-upenn/PersonaMem/blob/main/README.md",
        "files": [(name, name) for name in ("questions_32k.csv", "shared_contexts_32k.jsonl")],
    },
    "beam": {
        "provider": "github", "repository": "mohammadtavakoli78/BEAM",
        "documentation": "https://github.com/mohammadtavakoli78/BEAM/tree/main/chats/1M",
        "files": [(f"chats/1M/{number}/{name}", f"chats/1M/{number}/{name}")
                  for number in range(1, 36)
                  for name in ("chat.json", "probing_questions/probing_questions.json")],
    },
}
_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
_MANIFEST = "download_manifest.json"


def _request(url: str):
    return urlopen(Request(url, headers={"User-Agent": "PrismMem-public-data/1.0"}), timeout=60)


def _revision(value: str) -> str:
    if not value or value.startswith("-") or any(c.isspace() for c in value) or ".." in value:
        raise ValueError("revision must be a public branch, tag, or full 40-character commit SHA")
    return value


def _git_revision(repository: str, revision: str) -> str:
    """Resolve public refs when the unauthenticated GitHub API is rate limited."""
    env = os.environ.copy()
    env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull})
    refs = [revision] if revision.startswith("refs/") else [f"refs/heads/{revision}", f"refs/tags/{revision}"]
    result = subprocess.run(
        ["git", "ls-remote", f"https://github.com/{repository}.git", *refs, *(ref + "^{}" for ref in refs)],
        check=True, capture_output=True, text=True, timeout=60, env=env,
    )
    found = {}
    for line in result.stdout.splitlines():
        sha, _, ref = line.partition("\t")
        if _COMMIT.fullmatch(sha):
            found[ref] = sha.lower()
    # An annotated tag's peeled commit is preferable to the tag-object SHA.
    commits = {found.get(ref + "^{}", found[ref]) for ref in refs if ref in found}
    if len(commits) != 1:
        raise ValueError("revision is missing or ambiguous; supply an exact ref or full commit SHA")
    return commits.pop()


def resolve_revision(source: dict, revision: str) -> str:
    revision = _revision(revision)
    if _COMMIT.fullmatch(revision):
        return revision.lower()
    repo = source["repository"]
    encoded = quote(revision, safe="")
    if source["provider"] == "github":
        url = f"https://api.github.com/repos/{repo}/commits/{encoded}"
    else:
        url = f"https://huggingface.co/api/datasets/{repo}/revision/{encoded}"
    try:
        with _request(url) as response:
            payload = json.load(response)
    except HTTPError as exc:
        if source["provider"] == "github" and exc.code in {403, 429}:
            return _git_revision(repo, revision)
        raise
    sha = payload.get("sha") if isinstance(payload, dict) else None
    if not isinstance(sha, str) or not _COMMIT.fullmatch(sha):
        raise ValueError("upstream revision response did not contain a full commit SHA")
    return sha.lower()


def plan(benchmark: str, revision: str = "main") -> dict:
    """Construct an offline plan. Source paths and the BEAM 1M scope are fixed."""
    source = SOURCES[benchmark]
    revision = _revision(revision)
    repo = source["repository"]
    prefix = (f"https://raw.githubusercontent.com/{repo}/{quote(revision, safe='')}"
              if source["provider"] == "github"
              else f"https://huggingface.co/datasets/{repo}/resolve/{quote(revision, safe='')}")
    return {
        "benchmark": benchmark, "provider": source["provider"], "repository": repo,
        "documentation": source["documentation"], "requested_revision": revision,
        "file_count": len(source["files"]),
        "files": [{"source_path": remote, "path": local, "url": f"{prefix}/{remote}"}
                  for remote, local in source["files"]],
        "note": "Plan only. --execute resolves this revision and downloads the complete selected split; data can be large.",
    }


def _destination(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("unsafe relative download path")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("download path escapes the chosen output directory")
    return target


def _fingerprint(path: Path) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _install(temporary: Path, target: Path) -> None:
    """Atomically create, never replace; equal existing bytes are a safe resume."""
    try:
        os.link(temporary, target)
    except FileExistsError:
        if target.is_symlink() or not target.is_file() or _fingerprint(target) != _fingerprint(temporary):
            raise FileExistsError(f"Refusing to overwrite different existing file: {target.name}")


def _download(root: Path, entry: dict) -> dict:
    target = _destination(root, entry["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with _request(entry["url"]) as response, tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".download-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            digest, size = hashlib.sha256(), 0
            prefix = b""
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                if len(prefix) < 200:
                    prefix += chunk[:200 - len(prefix)]
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            expected = response.headers.get("Content-Length")
            if expected is not None and size != int(expected):
                raise ValueError("download Content-Length mismatch; incomplete response")
            if not size or prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError("upstream returned empty data or an unresolved Git LFS pointer")
            if prefix.lstrip().lower().startswith((b"<!doctype html", b"<html")):
                raise ValueError("upstream returned an HTML page instead of dataset bytes")
            etag = response.headers.get("ETag")
            linked_etag = response.headers.get("X-Linked-Etag")
            handle.flush()
            os.fsync(handle.fileno())
        _install(temporary, target)
        return {**entry, "sha256": digest.hexdigest(), "size_bytes": size,
                "etag": etag, "linked_etag": linked_etag}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download(benchmark: str, output: Path, *, revision: str = "main") -> dict:
    """Execute a pinned download; a completed manifest is verified and reused."""
    source = SOURCES[benchmark]
    resolved = resolve_revision(source, revision)
    pinned = plan(benchmark, resolved)
    root = Path(output).expanduser().resolve()
    identity = {"schema_version": 1, "benchmark": benchmark,
                "provider": source["provider"], "repository": source["repository"],
                "requested_revision": revision, "resolved_revision": resolved}
    manifest_path = _destination(root, _MANIFEST)
    if manifest_path.exists():
        if manifest_path.is_symlink():
            raise ValueError("download manifest must not be a symlink")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(existing.get(k) != v for k, v in identity.items()) or existing.get("status") != "complete":
            raise ValueError("output has a different download manifest; choose a new directory")
        rows = existing.get("files", [])
        if len(rows) != len(pinned["files"]):
            raise ValueError("existing manifest has an incomplete file list")
        for row, expected in zip(rows, pinned["files"]):
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError("existing manifest source does not match the pinned plan")
            target = _destination(root, row["path"])
            if target.is_symlink() or not target.is_file() or _fingerprint(target) != (row["sha256"], row["size_bytes"]):
                raise ValueError("existing download is missing or modified; choose a new directory")
        return existing
    root.mkdir(parents=True, exist_ok=True)
    rows = [_download(root, entry) for entry in pinned["files"]]
    manifest = {**identity, "status": "complete", "documentation": source["documentation"],
                "file_count": len(rows), "files": rows}
    raw = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=root, prefix=".manifest-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
    try:
        _install(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark", choices=tuple(SOURCES))
    parser.add_argument("--output", type=Path, required=True, help="data directory, preferably outside the submission tree")
    parser.add_argument("--revision", default="main", help="upstream branch, tag, or full commit SHA (default: main)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="print the offline plan; this is the default")
    action.add_argument("--execute", action="store_true", help="download the complete selected benchmark from public upstream")
    args = parser.parse_args(argv)
    try:
        result = (download(args.benchmark, args.output, revision=args.revision) if args.execute
                  else {**plan(args.benchmark, args.revision), "output": str(args.output)})
    except (OSError, ValueError, URLError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Download failed: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
