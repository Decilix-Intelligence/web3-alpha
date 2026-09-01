"""Offline tests for the pre-publication anonymity scanner."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_anonymity.py"
SPEC = importlib.util.spec_from_file_location("check_anonymity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
check_anonymity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_anonymity
SPEC.loader.exec_module(check_anonymity)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_detects_identity_paths_and_each_secret_family(tmp_path):
    email = "paper.author" + "@institution.test"
    profile_url = "https://github.com/" + "specific-author-98765"
    handle_line = "GitHub: " + "@specific-author-98765"
    mac_path = "/" + "Users/specific-author/project/file.py"
    linux_path = "/" + "home/specific-author/project/file.py"
    openai_key = "sk" + "-proj-" + ("A" * 28)
    github_key = "gh" + "p_" + ("b" * 36)
    aws_key = "AK" + "IA" + ("7" * 16)
    private_key = ("-" * 5) + "BEGIN PRIVATE KEY" + ("-" * 5)
    literal_setting = "api_" + 'key = "embedded-value"'
    json_literal_setting = '{"primary_service_api_' + 'key": "embedded-json-value"}'
    _write(
        tmp_path / "notes.md",
        "\n".join(
            [
                email,
                profile_url,
                handle_line,
                mac_path,
                linux_path,
                openai_key,
                github_key,
                aws_key,
                private_key,
                literal_setting,
                json_literal_setting,
            ]
        ),
    )

    result = check_anonymity.scan_repository(tmp_path)

    assert {finding.rule for finding in result.findings} == {
        "aws-key",
        "email",
        "github-handle",
        "github-key",
        "github-profile",
        "literal-api-key",
        "local-path",
        "openai-key",
        "private-key",
    }
    assert sum(finding.rule == "local-path" for finding in result.findings) == 2
    assert sum(finding.rule == "literal-api-key" for finding in result.findings) == 2


def test_allows_anonymous_examples_environment_values_and_project_urls(tmp_path):
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "Contact: anonymous@example.com",
                "Documentation: https://example.com/docs/quickstart",
                "Upstream project: https://github.com/python/cpython",
                "GitHub: https://github.com/python/cpython",
                "SSH upstream: git@github.com:python/cpython.git",
                "GitHub: @anonymous",
                'api_key = "${OPENAI_API_KEY}"',
                "OPENAI_API_KEY=",
            ]
        ),
    )
    _write(
        tmp_path / "client.py",
        (
            'api_key = os.getenv("OPENAI_API_KEY")\n'
            'github_handle = os.getenv("PROJECT_HANDLE")\n'
            "client(api_key=api_key)\n"
        ),
    )
    detector_source = 'LOCAL_RE = re.compile(r"(?:/' + 'Users/|/home/)[^\\s]+")\n'
    _write(tmp_path / "detector.py", detector_source)

    result = check_anonymity.scan_repository(tmp_path)

    assert result.findings == ()


def test_filesystem_fallback_respects_generated_directory_exclusions(tmp_path):
    leaked_email = "named.person" + "@research.test"
    _write(tmp_path / "README.md", leaked_email)
    _write(tmp_path / ".venv" / "cached.txt", "another.person" + "@research.test")
    _write(tmp_path / "node_modules" / "package.txt", "third.person" + "@research.test")
    _write(tmp_path / "output" / "run.json", "runtime.person" + "@research.test")

    result = check_anonymity.scan_repository(tmp_path)

    assert result.discovery == "filesystem fallback (no Git worktree)"
    assert [finding.path.as_posix() for finding in result.findings] == ["README.md"]


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is not installed")
def test_git_mode_scans_only_tracked_files(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    _write(tmp_path / "tracked.md", "tracked.person" + "@research.test")
    _write(tmp_path / "untracked.md", "untracked.person" + "@research.test")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.md"], check=True)

    result = check_anonymity.scan_repository(tmp_path)

    assert result.discovery == "git tracked files"
    assert [finding.path.as_posix() for finding in result.findings] == ["tracked.md"]


def test_cli_has_distinct_pass_fail_and_error_exit_codes(tmp_path):
    _write(tmp_path / "README.md", "Anonymous research artifact.\n")
    clean = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert clean.returncode == 0
    assert "Anonymity scan: PASS" in clean.stdout

    _write(tmp_path / "README.md", "author.contact" + "@research.test\n")
    leaking = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert leaking.returncode == 1
    assert "Anonymity scan: FAIL" in leaking.stdout
    assert "README.md:1:" in leaking.stdout
    assert "[email]" in leaking.stdout
    assert "author.contact" not in leaking.stdout

    missing = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path / "missing")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing.returncode == 2
    assert "Anonymity scan: ERROR" in missing.stderr
