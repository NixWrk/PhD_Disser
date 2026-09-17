from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import fitz


COLLECTION_KEY = "7X4U4E6U"
API = "http://127.0.0.1:23119/api/users/0"
ROOT = Path(__file__).resolve().parent
TEXT_DIR = ROOT / "corpus_text"


def api_json(path: str):
    with urllib.request.urlopen(API + path, timeout=30) as response:
        return json.load(response)


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-zа-яё0-9]+", "", (value or "").casefold())


def source_id(item: dict) -> str:
    data = item["data"]
    creators = data.get("creators") or []
    author = creators[0].get("lastName", "source") if creators else "source"
    year_match = re.search(r"(?:19|20)\d{2}", data.get("date", ""))
    year = year_match.group(0) if year_match else "nd"
    base = re.sub(r"[^A-Za-zА-Яа-яЁё0-9]+", "_", author).strip("_") or "source"
    return f"{base}_{year}_{item['key']}"


def file_path_from_attachment(attachment: dict) -> Path | None:
    enclosure = attachment.get("links", {}).get("enclosure", {})
    href = enclosure.get("href", "")
    if not href.startswith("file:///"):
        return None
    decoded = urllib.parse.unquote(urllib.parse.urlparse(href).path)
    if re.match(r"^/[A-Za-z]:/", decoded):
        decoded = decoded[1:]
    return Path(decoded)


def pdf_record(path: Path, sid: str) -> dict:
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    ocr_path = ROOT / "ocr_pdf" / f"{sid}.pdf"
    extraction_path = ocr_path if ocr_path.exists() else path
    pages = []
    extraction_error = None
    try:
        with fitz.open(extraction_path) as doc:
            for page_number, page in enumerate(doc, start=1):
                text = page.get_text("text").replace("\x00", "").strip()
                pages.append(f"=== PAGE {page_number} ===\n{text}")
            page_count = doc.page_count
    except Exception as exc:
        page_count = 0
        extraction_error = f"{type(exc).__name__}: {exc}"
    text = "\n\n".join(pages)
    text_path = TEXT_DIR / f"{sid}.txt"
    text_path.write_text(text, encoding="utf-8")
    return {
        "pdf_path": str(path),
        "ocr_path": str(ocr_path) if ocr_path.exists() else None,
        "content_hash": f"sha256:{sha256}",
        "page_count": page_count,
        "text_chars": len(text),
        "text_path": str(text_path),
        "extraction_error": extraction_error,
    }


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    items = api_json(f"/collections/{COLLECTION_KEY}/items/top?limit=100")

    grouped: dict[str, list[dict]] = {}
    for item in items:
        data = item.get("data", {})
        doi = (data.get("DOI") or "").strip().lower()
        identity = f"doi:{doi}" if doi else f"title:{normalize_title(data.get('title', ''))}"
        grouped.setdefault(identity, []).append(item)

    sources = []
    duplicates = []
    for identity, candidates in grouped.items():
        ranked = []
        for candidate in candidates:
            children = api_json(f"/items/{candidate['key']}/children")
            pdf_children = [
                child
                for child in children
                if child.get("data", {}).get("contentType") == "application/pdf"
            ]
            ranked.append((1 if pdf_children else 0, candidate, pdf_children))
        ranked.sort(key=lambda row: row[0], reverse=True)
        _, item, pdf_children = ranked[0]
        for _, duplicate, duplicate_pdfs in ranked[1:]:
            duplicates.append(
                {
                    "identity": identity,
                    "kept_key": item["key"],
                    "duplicate_key": duplicate["key"],
                    "duplicate_has_pdf": bool(duplicate_pdfs),
                }
            )

        data = item["data"]
        sid = source_id(item)
        record = {
            "source_id": sid,
            "zotero_key": item["key"],
            "title": data.get("title", ""),
            "authors": data.get("creators", []),
            "date": data.get("date", ""),
            "doi": data.get("DOI", ""),
            "abstract": data.get("abstractNote", ""),
            "representation": "metadata",
            "local_ref": f"zotero:{item['key']}",
            "content_hash": None,
            "pdf_attachment_key": None,
            "pdf_path": None,
            "page_count": 0,
            "text_chars": 0,
            "text_path": None,
            "extraction_error": None,
        }
        if pdf_children:
            primary = pdf_children[0]
            path = file_path_from_attachment(primary)
            record["pdf_attachment_key"] = primary["key"]
            if path and path.exists():
                record.update(pdf_record(path, sid))
                record["representation"] = "pdf"
                record["local_ref"] = f"zotero:{item['key']}/attachment:{primary['key']}"
            else:
                record["extraction_error"] = "Local PDF path unavailable"
        sources.append(record)

    manifest = {
        "collection_key": COLLECTION_KEY,
        "collection_name": "Обзор аналогов и существующих методик мониторинга гемодинамики",
        "source_count": len(sources),
        "pdf_source_count": sum(1 for source in sources if source["representation"] == "pdf"),
        "metadata_only_count": sum(1 for source in sources if source["representation"] != "pdf"),
        "duplicates_excluded": duplicates,
        "sources": sorted(sources, key=lambda source: (source["date"], source["title"])),
    }
    (ROOT / "corpus_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "source_count": manifest["source_count"],
        "pdf_source_count": manifest["pdf_source_count"],
        "metadata_only_count": manifest["metadata_only_count"],
        "duplicates_excluded": duplicates,
        "low_text_sources": [
            {"source_id": source["source_id"], "chars": source["text_chars"], "pages": source["page_count"]}
            for source in sources
            if source["representation"] == "pdf" and source["text_chars"] < 1000
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
