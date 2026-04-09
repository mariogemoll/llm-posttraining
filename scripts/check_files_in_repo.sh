#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

set -euo pipefail

MAX_BYTES=$((50 * 1024))

failed=0

while read -r blob path; do
  [ -z "$path" ] && continue
  size=$(git cat-file -s "$blob")
  if [ "$size" -gt "$MAX_BYTES" ]; then
    printf 'Blob too large: %s (%s bytes > %s bytes)\n' "$path" "$size" "$MAX_BYTES"
    failed=1
  fi
done < <(git rev-list --all --objects)

exit "$failed"
