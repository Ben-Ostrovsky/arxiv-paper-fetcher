#!/usr/bin/env python3
"""Fetch arXiv papers, extract text, and zip the results."""

from __future__ import annotations

import argparse
import re
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote


REQUEST_TIMEOUT = 30
ARXIV_BATCH_SIZE = 100
ARXIV_MULTI_BATCH_DELAY_SECONDS = 3
USER_AGENT = "arxiv-paper-fetcher/0.1 (local research tool)"
KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

feedparser: Any = None
fitz: Any = None
requests: Any = None
yaml: Any = None
tqdm: Any = None


@dataclass
class Paper:
    key: str
    arxiv: str


@dataclass
class ArxivMetadata:
    arxiv: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    published: str = ""
    updated: str = ""
    abstract: str = ""

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{quote(self.arxiv, safe='/')}.pdf"


@dataclass
class RunStats:
    total: int = 0
    pdf_downloaded: int = 0
    pdf_reused: int = 0
    text_extracted: int = 0
    text_reused: int = 0
    failures: dict[str, str] = field(default_factory=dict)


def ensure_dependencies() -> None:
    """Import third-party packages only when the command actually runs."""

    global feedparser, fitz, requests, yaml, tqdm

    missing: list[str] = []
    try:
        import requests as requests_module
    except ImportError:
        missing.append("requests")
    else:
        requests = requests_module

    try:
        import yaml as yaml_module
    except ImportError:
        missing.append("pyyaml")
    else:
        yaml = yaml_module

    try:
        import fitz as fitz_module
    except ImportError:
        missing.append("pymupdf")
    else:
        fitz = fitz_module

    try:
        import feedparser as feedparser_module
    except ImportError:
        missing.append("feedparser")
    else:
        feedparser = feedparser_module

    try:
        from tqdm import tqdm as tqdm_class
    except ImportError:
        missing.append("tqdm")
    else:
        tqdm = tqdm_class

    if missing:
        raise RuntimeError(
            "Missing dependencies: "
            + ", ".join(missing)
            + ". Install them with: pip install -r requirements.txt"
        )


def clean_text(value: str | None) -> str:
    """Collapse whitespace in arXiv text fields."""

    return " ".join((value or "").split())


def validate_key(key: Any) -> str:
    """Return a filename-safe key or raise ValueError."""

    if not isinstance(key, str) or not key:
        raise ValueError("each paper needs a nonempty string key")
    if not KEY_RE.fullmatch(key):
        raise ValueError(
            f"unsafe key {key!r}; use only letters, numbers, underscore, hyphen, and dot"
        )
    return key


def normalize_arxiv_id(arxiv_id: Any) -> str:
    """Normalize an arXiv ID for API matching."""

    if not isinstance(arxiv_id, str) or not arxiv_id.strip():
        raise ValueError("each paper needs an arxiv ID")
    value = arxiv_id.strip()
    value = value.removeprefix("arXiv:").removeprefix("arxiv:")
    value = value.rstrip("/")
    if "/" in value and value.startswith(("http://", "https://")):
        value = value.rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", value)


def is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def load_manifest(path: Path) -> list[Paper]:
    """Load an arXiv-only manifest."""

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict) or not isinstance(data.get("papers"), list):
        raise ValueError("manifest must contain a 'papers' list")

    papers: list[Paper] = []
    seen_keys: set[str] = set()
    for index, item in enumerate(data["papers"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"paper entry {index} must be a mapping")
        key = validate_key(item.get("key"))
        if key in seen_keys:
            raise ValueError(f"duplicate key {key!r}")
        seen_keys.add(key)
        papers.append(Paper(key=key, arxiv=normalize_arxiv_id(item.get("arxiv"))))

    return papers


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def fetch_arxiv_metadata(session: Any, arxiv_ids: list[str]) -> dict[str, ArxivMetadata]:
    """Fetch metadata from the official arXiv API in batches."""

    metadata: dict[str, ArxivMetadata] = {}
    id_batches = chunks(arxiv_ids, ARXIV_BATCH_SIZE)

    for batch_index, id_batch in enumerate(id_batches):
        if batch_index > 0:
            time.sleep(ARXIV_MULTI_BATCH_DELAY_SECONDS)

        response = session.get(
            "https://export.arxiv.org/api/query",
            params={"id_list": ",".join(id_batch)},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        parsed = feedparser.parse(response.content)

        for entry in parsed.entries:
            entry_id = normalize_arxiv_id(getattr(entry, "id", "").rsplit("/", 1)[-1])
            authors = [
                author.get("name", "")
                for author in getattr(entry, "authors", [])
                if author.get("name")
            ]
            metadata[entry_id] = ArxivMetadata(
                arxiv=entry_id,
                title=clean_text(getattr(entry, "title", "")),
                authors=authors,
                published=getattr(entry, "published", ""),
                updated=getattr(entry, "updated", ""),
                abstract=clean_text(getattr(entry, "summary", "")),
            )

    return metadata


def download_pdf(session: Any, url: str, target: Path) -> None:
    """Download a PDF, retrying once on failure."""

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".part")
    last_error: Exception | None = None

    for _ in range(2):
        tmp_path.unlink(missing_ok=True)
        try:
            with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            handle.write(chunk)
            with tmp_path.open("rb") as handle:
                if b"%PDF-" not in handle.read(1024):
                    raise ValueError("downloaded file did not look like a PDF")
            tmp_path.replace(target)
            return
        except Exception as exc:
            last_error = exc

    tmp_path.unlink(missing_ok=True)
    raise RuntimeError(f"PDF download failed: {last_error}")


def text_header(paper: Paper, metadata: ArxivMetadata) -> str:
    """Build the metadata block placed at the top of each text file."""

    return (
        f"key: {paper.key}\n"
        f"arXiv ID: {paper.arxiv}\n"
        f"title: {metadata.title}\n"
        f"authors: {', '.join(metadata.authors)}\n"
        f"published date: {metadata.published}\n"
        f"updated date: {metadata.updated}\n"
        f"abstract: {metadata.abstract}\n"
        f"PDF URL: {metadata.pdf_url}\n"
    )


def extract_text(pdf_path: Path, text_path: Path, paper: Paper, metadata: ArxivMetadata) -> None:
    """Extract text from a PDF using PyMuPDF."""

    text_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = text_path.with_suffix(text_path.suffix + ".part")
    tmp_path.unlink(missing_ok=True)

    try:
        with fitz.open(pdf_path) as document, tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(text_header(paper, metadata))
            for page_number, page in enumerate(document, start=1):
                handle.write(f"\n\n===== PAGE {page_number} =====\n\n")
                handle.write(page.get_text("text"))
        tmp_path.replace(text_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def create_zip(project_dir: Path, manifest_path: Path) -> Path:
    """Create papers.zip with the manifest, PDFs, and text files."""

    zip_path = project_dir / "papers.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest_path, arcname="arxiv_manifest.yaml")
        for folder in (project_dir / "papers" / "pdf", project_dir / "papers" / "text"):
            for path in sorted(folder.glob("*")):
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(project_dir))
    return zip_path


def process_papers(project_dir: Path, papers: list[Paper], force: bool) -> RunStats:
    """Download PDFs and extract text for every manifest entry."""

    stats = RunStats(total=len(papers))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    try:
        metadata_by_id = fetch_arxiv_metadata(session, [paper.arxiv for paper in papers])
    except Exception as exc:
        for paper in papers:
            stats.failures[paper.key] = f"arXiv API failed: {exc}"
        return stats

    for paper in tqdm(papers, desc="Papers", unit="paper"):
        try:
            metadata = metadata_by_id.get(paper.arxiv)
            if metadata is None:
                raise RuntimeError("arXiv API returned no metadata for this ID")

            pdf_path = project_dir / "papers" / "pdf" / f"{paper.key}.pdf"
            text_path = project_dir / "papers" / "text" / f"{paper.key}.txt"

            if is_nonempty_file(pdf_path) and not force:
                stats.pdf_reused += 1
            else:
                download_pdf(session, metadata.pdf_url, pdf_path)
                stats.pdf_downloaded += 1

            if is_nonempty_file(text_path) and not force:
                stats.text_reused += 1
            else:
                extract_text(pdf_path, text_path, paper, metadata)
                stats.text_extracted += 1
        except Exception as exc:
            stats.failures[paper.key] = str(exc)

    return stats


def print_summary(stats: RunStats, zip_path: Path | None) -> None:
    """Print a concise run summary."""

    print("\nSummary")
    print(f"  papers in manifest: {stats.total}")
    print(f"  PDFs downloaded: {stats.pdf_downloaded}")
    print(f"  PDFs reused: {stats.pdf_reused}")
    print(f"  text files extracted: {stats.text_extracted}")
    print(f"  text files reused: {stats.text_reused}")
    print(f"  failures: {len(stats.failures)}")
    if zip_path:
        print(f"  zip: {zip_path}")

    if stats.failures:
        print("\nFailed keys:")
        for key, reason in stats.failures.items():
            print(f"  - {key}: {reason}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="arxiv_manifest.yaml",
        help="Path to the arXiv manifest. Defaults to arxiv_manifest.yaml.",
    )
    parser.add_argument("--force", action="store_true", help="Redownload PDFs and re-extract text.")
    parser.add_argument("--no-zip", action="store_true", help="Skip creating papers.zip.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    project_dir = Path(__file__).resolve().parent
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = project_dir / manifest_path

    try:
        ensure_dependencies()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        papers = load_manifest(manifest_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Could not read manifest: {exc}", file=sys.stderr)
        return 2

    stats = process_papers(project_dir, papers, args.force)
    zip_path = None if args.no_zip else create_zip(project_dir, manifest_path)
    print_summary(stats, zip_path)
    return 1 if stats.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
