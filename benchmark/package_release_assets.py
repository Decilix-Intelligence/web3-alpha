#!/usr/bin/env python3
"""Build release zip files and SHA256 manifests for Web-Rec artifacts."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "release_assets"

CLASSIC_PATHS = [
    ROOT / "benchmark" / "data",
    ROOT / "benchmark" / "recbole_data" / "webrec" / "webrec.inter",
    ROOT / "benchmark" / "results" / "benchmark_results.csv",
    ROOT / "benchmark" / "results" / "recbole_results.csv",
    ROOT / "benchmark" / "results" / "leaderboard_long.csv",
    ROOT / "benchmark" / "results" / "leaderboard_wide.csv",
    ROOT / "benchmark" / "results" / "leaderboard_summary.md",
    ROOT / "benchmark" / "results" / "llm_results.csv",
    ROOT / "docs" / "DATA.md",
    ROOT / "docs" / "REPRODUCIBILITY.md",
]

LLM_PATHS = [
    ROOT / "benchmark" / "newsrec" / "artifacts" / "webrec_mind_v2" / "prompt4nr_data",
]


def iter_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
        else:
            files.extend(p for p in path.rglob("*") if p.is_file())
    return sorted(files)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_zip(zip_path: Path, files: list[Path]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(ROOT).as_posix())


def write_manifest(manifest_path: Path, files: list[Path], archives: list[Path]) -> None:
    lines = []
    for path in sorted(files + archives):
        if path.exists():
            rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
            lines.append(f"{sha256(path)}  {rel}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package Web-Rec release assets.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="Also package Prompt4NR-compatible LLM artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    classic_zip = output_dir / "webrec-classic-v1.zip"
    classic_files = iter_files(CLASSIC_PATHS)
    write_zip(classic_zip, classic_files)

    archives = [classic_zip]
    manifest_files = classic_files.copy()

    if args.include_llm:
        llm_zip = output_dir / "webrec-llm-artifacts-v1.zip"
        llm_files = iter_files(LLM_PATHS)
        write_zip(llm_zip, llm_files)
        archives.append(llm_zip)
        manifest_files.extend(llm_files)

    manifest_path = output_dir / "SHA256SUMS"
    write_manifest(manifest_path, manifest_files, archives)

    print(f"Wrote {classic_zip}")
    if args.include_llm:
        print(f"Wrote {output_dir / 'webrec-llm-artifacts-v1.zip'}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
