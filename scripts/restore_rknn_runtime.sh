#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR="${1:-}"
if [[ -z "$BACKUP_DIR" ]]; then
  BACKUP_DIR="$(ls -td /home/elf/rknnrt_backups/* 2>/dev/null | head -1 || true)"
fi

if [[ -z "$BACKUP_DIR" || ! -f "$BACKUP_DIR/manifest.tsv" ]]; then
  echo "backup manifest not found. Pass a backup directory, for example:" >&2
  echo "  scripts/restore_rknn_runtime.sh /home/elf/rknnrt_backups/YYYYmmdd_HHMMSS" >&2
  exit 2
fi

MANIFEST="$BACKUP_DIR/manifest.tsv"
echo "using backup manifest: $MANIFEST"

while IFS=$'\t' read -r target backup_file; do
  [[ -z "${target:-}" ]] && continue
  if [[ "$backup_file" == "__MISSING__" ]]; then
    echo "remove installed file that did not exist before: $target"
    sudo rm -f "$target"
    continue
  fi
  if [[ ! -f "$backup_file" ]]; then
    echo "backup file missing: $backup_file" >&2
    exit 3
  fi
  echo "restore $backup_file -> $target"
  sudo mkdir -p "$(dirname "$target")"
  sudo cp -a "$backup_file" "$target"
done < "$MANIFEST"

sudo ldconfig
echo "restore complete. restart the NPU 8085 service."
