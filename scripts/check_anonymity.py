#!/usr/bin/env python3
"""Scan repository text for identity leaks and accidentally committed secrets.

The command intentionally uses only the Python standard library so it can run
before the project's optional dependencies are installed.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent

# These directories contain generated, vendored, or editor-owned files rather
# than repository source.  The same exclusions are applied in Git and fallback
# discovery modes so a local virtual environment cannot create noisy findings.
EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        ".vscode",
        "__pycache__",
        "build",
        "cache",
        "checkpoints",
        "datasets",
        "dist",
        "logs",
        "node_modules",
        "output",
        "outputs",
        "runs",
        "scratch",
        "site-packages",
        "splits",
        "tmp",
        "venv",
    }
)
EXCLUDED_FILES = frozenset({".DS_Store"})
OBVIOUS_BINARY_SUFFIXES = frozenset(
    {
        ".7z",
        ".avi",
        ".bmp",
        ".bz2",
        ".class",
        ".dylib",
        ".eot",
        ".feather",
        ".gif",
        ".gz",
        ".ico",
        ".jar",
        ".jpeg",
        ".jpg",
        ".mov",
        ".mp3",
        ".mp4",
        ".o",
        ".otf",
        ".parquet",
        ".pdf",
        ".png",
        ".pyc",
        ".pyo",
        ".so",
        ".tar",
        ".tiff",
        ".ttf",
        ".wav",
        ".webm",
        ".webp",
        ".woff",
        ".woff2",
        ".xls",
        ".xlsx",
        ".xz",
        ".zip",
    }
)

EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9][A-Za-z0-9._%+-]{0,63}@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,62}\.)+[A-Za-z]{2,63}"
    r"(?![A-Za-z0-9._%+-])"
)
LOCAL_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])/(?:Users|home)/[^\s'\"<>`]*")
GITHUB_URL_RE = re.compile(
    r"https?://(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))"
    r"(?P<remainder>/[^\s<>\"')\]]*)?",
    re.IGNORECASE,
)
GITHUB_HANDLE_RE = re.compile(
    r"\bgithub(?:[ _-]*(?:user(?:name)?|handle|account|profile))?"
    r"\s*[:=]\s*(?P<quote>['\"]?)(?P<at>@?)"
    r"(?P<handle>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))",
    re.IGNORECASE,
)
AUTHOR_HANDLE_RE = re.compile(
    r"\b(?:author|maintainer)\s*[:=]\s*['\"]?@" r"(?P<handle>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))",
    re.IGNORECASE,
)
OPENAI_KEY_RE = re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{16,}")
GITHUB_KEY_RE = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")
AWS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
PRIVATE_KEY_RE = re.compile(
    r"-{5}BEGIN[ \t]+(?:RSA[ \t]+|EC[ \t]+|OPENSSH[ \t]+|DSA[ \t]+)?" r"PRIVATE[ \t]+KEY-{5}"
)
API_KEY_ASSIGNMENT_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"['\"]?(?P<name>(?:[A-Za-z0-9]+[_-])*api[_-]?key)['\"]?"
    r"\s*(?:=|:)\s*"
    r"(?P<value>\"[^\"\n]*\"|'[^'\n]*'|[^\s#,;]+)",
    re.IGNORECASE,
)

CODE_SUFFIXES = frozenset(
    {".c", ".cc", ".cpp", ".go", ".java", ".js", ".jsx", ".py", ".rb", ".rs", ".ts", ".tsx"}
)
PLACEHOLDER_HANDLES = frozenset(
    {
        "anon",
        "anonymous",
        "example",
        "example-user",
        "user",
        "username",
        "your-handle",
        "your-username",
    }
)
PLACEHOLDER_WORDS = frozenset(
    {
        "changeme",
        "dummy",
        "empty",
        "example",
        "fake",
        "none",
        "null",
        "placeholder",
        "redacted",
        "replace-me",
        "replace_me",
        "sample",
        "test",
        "your-api-key",
        "your-key",
        "your_api_key",
        "your_key",
    }
)
SSH_GIT_ADDRESS = "git" + "@github.com"


@dataclass(frozen=True)
class Finding:
    """One actionable match.  The matched value is deliberately not retained."""

    path: Path
    line: int
    column: int
    rule: str
    message: str


@dataclass(frozen=True)
class ScanResult:
    root: Path
    discovery: str
    files_scanned: int
    files_skipped: int
    findings: tuple[Finding, ...]


class ScanError(RuntimeError):
    """Raised when the requested root cannot be scanned reliably."""


def _is_excluded(relative_path: Path) -> bool:
    return relative_path.name in EXCLUDED_FILES or any(
        part in EXCLUDED_DIRS for part in relative_path.parts[:-1]
    )


def _git_tracked_files(root: Path) -> list[Path] | None:
    """Return tracked paths, or ``None`` when *root* is not in a Git worktree."""

    try:
        probe = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        return None
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return None

    listed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--", "."],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if listed.returncode != 0:
        detail = listed.stderr.decode("utf-8", errors="replace").strip()
        raise ScanError(f"git ls-files failed: {detail or 'unknown Git error'}")

    paths: list[Path] = []
    for raw_path in listed.stdout.split(b"\0"):
        if not raw_path:
            continue
        try:
            relative = Path(os.fsdecode(raw_path))
        except UnicodeError:
            continue
        if relative.is_absolute() or ".." in relative.parts or _is_excluded(relative):
            continue
        candidate = root / relative
        if candidate.is_file() and not candidate.is_symlink():
            paths.append(candidate)
    return sorted(set(paths), key=lambda path: path.as_posix())


def _filesystem_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name not in EXCLUDED_DIRS)
        current_path = Path(current)
        for filename in sorted(filenames):
            candidate = current_path / filename
            relative = candidate.relative_to(root)
            if _is_excluded(relative) or candidate.is_symlink() or not candidate.is_file():
                continue
            paths.append(candidate)
    return paths


def discover_files(root: Path) -> tuple[list[Path], str]:
    """Discover tracked files, falling back to a filtered tree walk without Git."""

    tracked = _git_tracked_files(root)
    if tracked is not None:
        return tracked, "git tracked files"
    return _filesystem_files(root), "filesystem fallback (no Git worktree)"


def _read_text(path: Path) -> str | None:
    if path.suffix.lower() in OBVIOUS_BINARY_SUFFIXES:
        return None
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ScanError(f"cannot read {path}: {exc}") from exc
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def _is_allowed_email(value: str, line: str, end: int) -> bool:
    if value.casefold() == "anonymous@example.com":
        return True
    # An SSH Git remote is not an email address.  Project URLs are allowed; a
    # GitHub profile URL is handled separately by the URL rule.
    return value.casefold() == SSH_GIT_ADDRESS and line[end : end + 1] == ":"


def _is_placeholder_handle(value: str) -> bool:
    return value.casefold() in PLACEHOLDER_HANDLES


def _is_placeholder_value(value: str) -> bool:
    value = value.strip().strip("'\"").strip()
    lowered = value.casefold()
    if not value or lowered in PLACEHOLDER_WORDS:
        return True
    if lowered.startswith(("${", "$", "{{", "<")) or lowered.endswith(("}}", ">")):
        return True
    if lowered.startswith(("!env", "env(", "env:", "os.getenv(", "os.environ[", "process.env.")):
        return True
    if any(word in lowered for word in ("placeholder", "example", "redacted")):
        return True
    if lowered.startswith(("your-", "your_")) or any(
        marker in lowered for marker in ("your-key", "your_key", "your-api", "your_api")
    ):
        return True
    if re.search(r"(?:^|[-_])x{8,}(?:$|[-_])", lowered):
        return True
    compact = re.sub(r"[^a-z0-9]", "", lowered)
    return bool(compact) and set(compact) <= {"x"}


def _api_assignment_is_literal(match: re.Match[str], path: Path) -> bool:
    raw = match.group("value")
    if _is_placeholder_value(raw):
        return False
    quoted = len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\""
    if quoted:
        return True
    if path.suffix.casefold() in CODE_SUFFIXES:
        # In source code, unquoted values are expressions such as api_key=key
        # or api_key=config.get(...), not embedded credentials.
        return False
    return raw.casefold() not in {"str", "string", "optional[str]"}


def _line_findings(path: Path, line_number: int, line: str) -> Iterable[Finding]:
    for match in EMAIL_RE.finditer(line):
        value = match.group(0)
        if not _is_allowed_email(value, line, match.end()):
            yield Finding(path, line_number, match.start() + 1, "email", "possible email address")

    for match in LOCAL_PATH_RE.finditer(line):
        if "re.compile(" in line[: match.start()]:
            # Detection code often spells these prefixes inside a regular
            # expression; that is not a workstation path or an identity leak.
            continue
        yield Finding(
            path,
            line_number,
            match.start() + 1,
            "local-path",
            "local home-directory path",
        )

    for match in GITHUB_URL_RE.finditer(line):
        remainder = match.group("remainder")
        # One path component is a user/organization profile.  Two or more path
        # components identify a project and are allowed for third-party links.
        if not remainder or remainder == "/" or remainder.startswith(("/?", "/#")):
            yield Finding(
                path,
                line_number,
                match.start() + 1,
                "github-profile",
                "GitHub profile URL",
            )

    for match in GITHUB_HANDLE_RE.finditer(line):
        handle = match.group("handle")
        dynamic_value = handle.casefold() in {"env", "git", "http", "https", "os", "ssh"}
        unquoted_code_value = (
            not match.group("at")
            and not match.group("quote")
            and path.suffix.casefold() in CODE_SUFFIXES
        )
        if not (_is_placeholder_handle(handle) or dynamic_value or unquoted_code_value):
            yield Finding(
                path,
                line_number,
                match.start() + 1,
                "github-handle",
                "GitHub or author handle",
            )

    for match in AUTHOR_HANDLE_RE.finditer(line):
        if not _is_placeholder_handle(match.group("handle")):
            yield Finding(
                path,
                line_number,
                match.start() + 1,
                "github-handle",
                "GitHub or author handle",
            )

    token_rules = (
        (OPENAI_KEY_RE, "openai-key", "possible OpenAI API key"),
        (GITHUB_KEY_RE, "github-key", "possible GitHub access token"),
        (AWS_KEY_RE, "aws-key", "possible AWS access key"),
        (PRIVATE_KEY_RE, "private-key", "private-key header"),
    )
    for pattern, rule, message in token_rules:
        for match in pattern.finditer(line):
            if rule != "private-key" and _is_placeholder_value(match.group(0)):
                continue
            yield Finding(path, line_number, match.start() + 1, rule, message)

    for match in API_KEY_ASSIGNMENT_RE.finditer(line):
        if _api_assignment_is_literal(match, path):
            yield Finding(
                path,
                line_number,
                match.start() + 1,
                "literal-api-key",
                "literal value assigned to an API-key setting",
            )


def scan_repository(root: Path) -> ScanResult:
    root = root.expanduser().resolve()
    if not root.exists():
        raise ScanError(f"root does not exist: {root}")
    if not root.is_dir():
        raise ScanError(f"root is not a directory: {root}")

    files, discovery = discover_files(root)
    findings: list[Finding] = []
    scanned = 0
    skipped = 0
    for path in files:
        text = _read_text(path)
        if text is None:
            skipped += 1
            continue
        scanned += 1
        relative = path.relative_to(root)
        for line_number, line in enumerate(text.splitlines(), start=1):
            findings.extend(_line_findings(relative, line_number, line))

    findings.sort(key=lambda item: (item.path.as_posix(), item.line, item.column, item.rule))
    return ScanResult(root, discovery, scanned, skipped, tuple(findings))


def _print_report(result: ScanResult) -> None:
    status = "FAIL" if result.findings else "PASS"
    print(f"Anonymity scan: {status}")
    print(f"Discovery: {result.discovery}")
    print(f"Scanned: {result.files_scanned} text file(s); skipped: {result.files_skipped}")
    if not result.findings:
        print("No identity or secret patterns found.")
        return
    print(f"Findings: {len(result.findings)}")
    for finding in result.findings:
        print(
            f"  {finding.path.as_posix()}:{finding.line}:{finding.column} "
            f"[{finding.rule}] {finding.message}"
        )
    print("Remove the findings or replace values with anonymous/environment placeholders.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan tracked repository text for identity and secret leaks.",
        epilog="Exit status: 0 = clean, 1 = findings, 2 = usage or scan error.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root to scan (default: the parent directory of this script)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = scan_repository(args.root)
    except ScanError as exc:
        print(f"Anonymity scan: ERROR\n{exc}", file=sys.stderr)
        return 2
    _print_report(result)
    return 1 if result.findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
