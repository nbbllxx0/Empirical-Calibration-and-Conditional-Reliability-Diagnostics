from __future__ import annotations

import hashlib
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from bearing_dt.table import Rows, read_rows_csv, write_rows_csv
from bearing_dt.utils import ensure_dir, write_json


PHME_ZENODO_RECORD = "10868257"
PHME_ZENODO_API = f"https://zenodo.org/api/records/{PHME_ZENODO_RECORD}"
PHME_ZENODO_DOI = "10.5281/zenodo.10868257"
PHME_ZENODO_TITLE = "Run-to-failure data set of ball bearings subjected to time-varying load and speed conditions"

XJTU_SOURCE_LINKS = [
    "https://biaowang.tech/xjtu-sy-bearing-datasets/",
    "https://drive.google.com/open?id=1_ycmG46PARiykt82ShfnFfyQsaXv3_VK",
    "https://www.dropbox.com/sh/qka3b73wuvn5l7a/AADr6oXKbafhOlrBLCNgonzua?dl=0",
    "http://www.mediafire.com/folder/m3sij67rizpb4/XJTU-SY_Bearing_Datasets",
    "https://mega.nz/#F!H7pnGKBK!PR8qUShaLlJjwrPf3SlBjw",
    "https://pan.baidu.com/s/1OaY82azTXHBwjiCjA_jRcw",
]


def _url_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, path: Path, chunk_size: int = 2**20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "bearing-dt/0.1"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"
    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if downloaded else "wb"
    try:
        with urllib.request.urlopen(req, timeout=120) as response, partial.open(mode) as f:
            shutil.copyfileobj(response, f, length=chunk_size)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and partial.exists():
            partial.replace(path)
            return
        raise
    partial.replace(path)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zenodo_files(record: dict[str, Any]) -> Rows:
    rows: Rows = []
    for item in record.get("files", []):
        links = item.get("links", {})
        checksum = item.get("checksum") or ""
        rows.append(
            {
                "filename": item.get("key") or item.get("filename"),
                "size": int(item.get("size", 0)),
                "checksum": checksum.replace("md5:", ""),
                "download_url": links.get("self") or links.get("download"),
            }
        )
    return rows


def fetch_phme_tvoc(
    out: str | Path,
    files: list[str] | None = None,
    manifest_only: bool = False,
    extract: bool = False,
    verify_md5: bool = True,
    max_gb: float | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(out)
    record = None
    try:
        record = _url_json(PHME_ZENODO_API)
        rows = _zenodo_files(record)
    except Exception:
        manifest_path = out_dir / "file_manifest.csv"
        if not manifest_path.exists():
            raise
        rows = read_rows_csv(manifest_path)
    if not rows:
        raise RuntimeError("Zenodo API returned no files for PHME record")
    write_rows_csv(out_dir / "file_manifest.csv", rows)
    source_manifest = {
        "dataset": "phme_tvoc",
        "record": PHME_ZENODO_RECORD,
        "api": PHME_ZENODO_API,
        "doi": record.get("doi") if record else PHME_ZENODO_DOI,
        "title": record.get("metadata", {}).get("title") if record else PHME_ZENODO_TITLE,
        "total_files": len(rows),
        "total_size_gb": sum(float(row["size"]) for row in rows) / 1e9,
        "notes": "Full Zenodo record is very large; download selected files first.",
        "metadata_source": "zenodo_api" if record else "local_file_manifest",
    }
    write_json(out_dir / "source_manifest.json", source_manifest)
    if manifest_only:
        return {"downloaded": [], "manifest": str(out_dir / "file_manifest.csv")}

    selected = rows
    if files:
        wanted = set(files)
        selected = [row for row in rows if row["filename"] in wanted]
        missing = sorted(wanted - {row["filename"] for row in selected})
        if missing:
            raise FileNotFoundError(f"Requested PHME files not found in Zenodo manifest: {missing}")

    total_gb = sum(float(row["size"]) for row in selected) / 1e9
    if max_gb is not None and total_gb > max_gb:
        raise ValueError(f"Requested download is {total_gb:.2f} GB, above --max-gb {max_gb}.")

    downloaded = []
    for row in selected:
        filename = str(row["filename"])
        url = str(row["download_url"])
        if not url:
            raise RuntimeError(f"No download URL for {filename}")
        target = out_dir / filename
        if target.exists() and target.stat().st_size == int(row["size"]):
            pass
        else:
            print(f"Downloading {filename} ({int(row['size']) / 1e9:.2f} GB)")
            _download(url, target)
        if verify_md5 and row.get("checksum"):
            actual = _md5(target)
            if actual.lower() != str(row["checksum"]).lower():
                raise ValueError(f"MD5 mismatch for {target}: expected {row['checksum']}, got {actual}")
        downloaded.append(str(target))
        if extract and target.suffix.lower() == ".zip":
            extract_dir = ensure_dir(out_dir / target.stem)
            with zipfile.ZipFile(target) as zf:
                zf.extractall(extract_dir)
    return {"downloaded": downloaded, "manifest": str(out_dir / "file_manifest.csv")}


def fetch_xjtu_sy(out: str | Path) -> dict[str, Any]:
    out_dir = ensure_dir(out)
    write_json(
        out_dir / "source_manifest.json",
        {
            "dataset": "xjtu_sy",
            "status": "manual_download_required",
            "reason": "Official page provides browser/file-share links rather than a stable direct archive URL.",
            "source_links": XJTU_SOURCE_LINKS,
            "expected_archive": "XJTU-SY_Bearing_Datasets.zip",
        },
    )
    return {"downloaded": [], "manifest": str(out_dir / "source_manifest.json")}
