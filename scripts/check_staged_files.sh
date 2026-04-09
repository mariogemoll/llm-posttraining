#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

set -euo pipefail

MAX_BYTES=$((50 * 1024))

failed=0

while IFS= read -r -d '' file; do
  [ -f "$file" ] || continue

  size=$(wc -c <"$file")
  if [ "$size" -gt "$MAX_BYTES" ]; then
    printf 'File too large: %s (%s bytes > %s bytes)\n' "$file" "$size" "$MAX_BYTES"
    failed=1
  fi
done < <(git diff --cached --diff-filter=AM --name-only -z)

exit "$failed"
