#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Check that all source files have correct SPDX license headers."""

import os
import re
import sys

# --- Configuration ---

EXCLUDED_DIRS = {"node_modules", "dist", "build", ".git", ".claude", ".venv", ".ipynb_checkpoints", "__pycache__", "*.egg-info"}
EXCLUDED_FILES = {".DS_Store"}
COPYRIGHT_TEXT = "2026 Mario Gemoll"

# Comment styles by extension
COMMENT_STYLES: dict[str, tuple[str, str]] = {
    ".py": ("# ", ""),
    ".sh": ("# ", ""),
    ".ts": ("// ", ""),
    ".js": ("// ", ""),
    ".mjs": ("// ", ""),
    ".css": ("/* ", " */"),
    ".html": ("<!-- ", " -->"),
    ".md": ("<!-- ", " -->"),
}

# Rules: list of (patterns, license). First match wins.
RULES: list[tuple[list[str], str]] = [
    (["**/*.py", "**/*.sh"], "0BSD"),
]


# --- Logic ---

def matches_pattern(path: str, pattern: str) -> bool:
    import fnmatch
    # Support ** for recursive matching
    if "**/" in pattern:
        # **/*.ext should match any depth
        suffix = pattern.split("**/", 1)[1]
        return fnmatch.fnmatch(os.path.basename(path), suffix)
    return fnmatch.fnmatch(path, pattern)


def get_license_for_file(rel_path: str) -> str | None:
    for patterns, license_id in RULES:
        for pattern in patterns:
            if matches_pattern(rel_path, pattern):
                return license_id
    return None


def extract_from_header(content: str, tag: str) -> str | None:
    for line in content.split("\n")[:5]:
        m = re.search(rf"{tag}:\s*(.+?)(?:\s*(?:-->|\*/|$))", line)
        if m:
            return m.group(1).strip()
    return None


def has_empty_line_after_headers(content: str) -> bool:
    lines = content.split("\n")
    for i, line in enumerate(lines[:10]):
        if "SPDX-License-Identifier:" in line:
            return i + 1 < len(lines) and lines[i + 1].strip() == ""
    return False


def validate_file(file_path: str, expected_license: str, root: str) -> str | None:
    rel = os.path.relpath(file_path, root)
    try:
        content = open(file_path, encoding="utf-8").read()
    except (UnicodeDecodeError, OSError):
        return None

    copyright = extract_from_header(content, "SPDX-FileCopyrightText")
    license_id = extract_from_header(content, "SPDX-License-Identifier")

    if copyright is None:
        return f"  {rel}\n    Missing copyright header (expected: {COPYRIGHT_TEXT})"
    if copyright != COPYRIGHT_TEXT:
        return f"  {rel}\n    Expected copyright: {COPYRIGHT_TEXT}\n    Found copyright:    {copyright}"
    if license_id is None:
        return f"  {rel}\n    Missing license header (expected: {expected_license})"
    if license_id != expected_license:
        return f"  {rel}\n    Expected license: {expected_license}\n    Found license:    {license_id}"
    if not has_empty_line_after_headers(content):
        return f"  {rel}\n    Missing empty line after SPDX headers"
    return None


def should_exclude_dir(name: str) -> bool:
    for pattern in EXCLUDED_DIRS:
        if "*" in pattern:
            import fnmatch
            if fnmatch.fnmatch(name, pattern):
                return True
        elif name == pattern:
            return True
    return False


def check_directory(dir_path: str, root: str) -> list[str]:
    errors: list[str] = []
    for entry in sorted(os.listdir(dir_path)):
        full = os.path.join(dir_path, entry)
        if os.path.isdir(full):
            if not should_exclude_dir(entry):
                errors.extend(check_directory(full, root))
        elif os.path.isfile(full):
            if entry in EXCLUDED_FILES:
                continue
            rel = os.path.relpath(full, root)
            ext = os.path.splitext(full)[1]
            expected = get_license_for_file(rel)
            if expected is not None and ext in COMMENT_STYLES:
                err = validate_file(full, expected, root)
                if err:
                    errors.append(err)
    return errors


def main() -> None:
    root = os.getcwd()
    print("Checking SPDX license headers...\n")
    errors = check_directory(root, root)
    if not errors:
        print("\u2713 All files have correct license headers!")
        sys.exit(0)
    else:
        print(f"\u2717 Found {len(errors)} file(s) with incorrect or missing license headers:\n", file=sys.stderr)
        for err in errors:
            print(err, file=sys.stderr)
            print(file=sys.stderr)
        print("Fix the headers listed above and re-run this check.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
