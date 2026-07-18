#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:-/home/elf/librknnrt.so}"
BACKUP_ROOT="${BACKUP_ROOT:-/home/elf/rknnrt_backups}"

if [[ ! -f "$SOURCE" ]]; then
  echo "missing source runtime: $SOURCE" >&2
  echo "copy librknnrt.so to /home/elf/librknnrt.so first, or pass its path as the first argument." >&2
  exit 2
fi

echo "[source]"
ls -lh "$SOURCE"
if command -v strings >/dev/null 2>&1; then
  strings "$SOURCE" | grep -i "librknnrt version" || true
fi

mapfile -t FOUND_TARGETS < <(sudo find /usr /lib -name "librknnrt.so*" 2>/dev/null | sort -u)
TARGETS=()
for found in "${FOUND_TARGETS[@]}"; do
  if [[ -L "$found" ]]; then
    resolved="$(readlink -f "$found" || true)"
    if [[ -n "$resolved" ]]; then
      TARGETS+=("$resolved")
    fi
  else
    TARGETS+=("$found")
  fi
done
mapfile -t TARGETS < <(printf "%s\n" "${TARGETS[@]}" | sed '/^$/d' | sort -u)
if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  TARGETS=("/usr/lib/librknnrt.so")
  echo "no existing librknnrt.so found under /usr or /lib; will install to ${TARGETS[0]}"
fi

echo "[targets]"
printf "%s\n" "${TARGETS[@]}"

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${BACKUP_ROOT}/${STAMP}"
sudo mkdir -p "$BACKUP_DIR/files"
sudo chown "$(id -u)":"$(id -g)" "$BACKUP_DIR"
sudo chown "$(id -u)":"$(id -g)" "$BACKUP_DIR/files"
MANIFEST="${BACKUP_DIR}/manifest.tsv"
: > "$MANIFEST"

echo "[backup]"
for target in "${TARGETS[@]}"; do
  if [[ -e "$target" ]]; then
    safe_name="$(echo "$target" | sed 's#^/##; s#[/:]#_#g')"
    backup_file="${BACKUP_DIR}/files/${safe_name}"
    echo "$target -> $backup_file"
    sudo cp -a "$target" "$backup_file"
    printf "%s\t%s\n" "$target" "$backup_file" >> "$MANIFEST"
  else
    printf "%s\t%s\n" "$target" "__MISSING__" >> "$MANIFEST"
  fi
done

echo "[install]"
for target in "${TARGETS[@]}"; do
  echo "$SOURCE -> $target"
  sudo mkdir -p "$(dirname "$target")"
  sudo cp -f "$SOURCE" "$target"
done

sudo ldconfig

echo "[verify candidates]"
for target in "${TARGETS[@]}"; do
  if [[ -f "$target" ]]; then
    echo "==== $target ===="
    ls -lh "$target"
    strings "$target" | grep -i "librknnrt version" || true
  fi
done

echo "backup manifest: $MANIFEST"
echo "restart the NPU 8085 service after this install."
