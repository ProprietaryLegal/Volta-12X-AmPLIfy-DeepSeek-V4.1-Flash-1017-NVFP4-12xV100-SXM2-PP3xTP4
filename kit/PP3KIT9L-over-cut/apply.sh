#!/usr/bin/env bash
# apply.sh [--manifest MANIFEST] <engine source vllm dir> <venv site-packages vllm dir> <backup dir>
#
# File-level install of a kit (built by the kit tool of a private working repository). MANIFEST defaults to the one
# next to this script; the new files are in files/ next to MANIFEST. Each MANIFEST line (tab-separated):
#   <path relative to the vllm package>  <baseline sha256, or NEW>  <new sha256>
#
# Order, and nothing else:
#   1. every kit file has its new sha256;
#   2. BOTH trees hold exactly the baseline: every changed file has its baseline sha256, every NEW file is absent
#      (a NEW file's directory may be missing: apply creates it, lists it in CREATED_DIRS, and rollback removes it);
#   3. the backup dir does not exist yet; the originals of both trees are saved into it (<backup>/src/<path>,
#      <backup>/venv/<path>), with BACKUP.sha256 (sha256sum format), ADDED (the NEW paths), CREATED_DIRS (the
#      directories apply creates, relative, deepest last) and the MANIFEST;
#   4. the new files are copied into both trees (each to a temporary name in its directory, then renamed);
#   5. every file in both trees has its new sha256.
# Any failure in 1-3 changes nothing in either tree. A failure in 4-5 (a disk error) restores both trees from the
# backup before exiting non-zero. No pip, no build, no restart.
set -euo pipefail

KIT=$(cd "$(dirname "$0")" && pwd)
MANIFEST=$KIT/MANIFEST
if [ "${1:-}" = --manifest ]; then MANIFEST=$(realpath "$2"); shift 2; fi
[ $# -eq 3 ] || { echo "usage: apply.sh [--manifest MANIFEST] <engine source vllm dir> <venv vllm dir> <backup dir>" >&2; exit 2; }
SRC=$(realpath "$1"); VENV=$(realpath "$2"); BACKUP=$3
FILES=$(dirname "$MANIFEST")/files
die() { echo "apply: REFUSED: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }

[ -f "$MANIFEST" ] || die "no manifest $MANIFEST"
[ "$SRC" != "$VENV" ] || die "the two trees are the same directory"
mapfile -t ROWS < <(grep -v '^#' "$MANIFEST" | grep -v '^$')
[ ${#ROWS[@]} -gt 0 ] || die "empty manifest"
ENGINE=$(awk '$1 == "#" && $2 == "engine" {print $3}' "$MANIFEST")
echo "apply: kit for engine ${ENGINE:-?}, ${#ROWS[@]} files, manifest $MANIFEST"

# 1. the kit's own files
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r rel base new <<<"$row"
  [ -f "$FILES/$rel" ] || die "kit file missing: $rel"
  [ "$(sha "$FILES/$rel")" = "$new" ] || die "kit file $rel does not have its manifest sha256"
done

# 2. both trees hold the baseline
for tree in "$SRC" "$VENV"; do
  [ -d "$tree" ] || die "no directory $tree"
  for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r rel base new <<<"$row"
    if [ "$base" != NEW ]; then
      [ -d "$(dirname "$tree/$rel")" ] || die "$tree: no directory for $rel"
    fi
    if [ "$base" = NEW ]; then
      [ ! -e "$tree/$rel" ] || die "$tree/$rel exists, the kit adds it (already applied?)"
    else
      [ -f "$tree/$rel" ] || die "$tree/$rel is missing"
      got=$(sha "$tree/$rel")
      [ "$got" = "$base" ] || die "$tree/$rel is not the baseline (sha256 $got, want $base)"
    fi
  done
done
echo "apply: both trees hold the baseline for every file"

# 3. backup
[ ! -e "$BACKUP" ] || die "backup dir $BACKUP exists; give a new one"
mkdir -p "$BACKUP/src" "$BACKUP/venv"
: > "$BACKUP/ADDED"
: > "$BACKUP/CREATED_DIRS"
# directories a NEW file needs that the trees lack (the same in both trees: they hold the same baseline)
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r rel base new <<<"$row"
  [ "$base" = NEW ] || continue
  d=$(dirname "$rel")
  missing=()
  while [ "$d" != . ] && [ ! -d "$SRC/$d" ]; do
    [ ! -d "$VENV/$d" ] || die "$d exists in $VENV but not in $SRC"
    missing=("$d" "${missing[@]}"); d=$(dirname "$d")
  done
  for d in "${missing[@]}"; do grep -qxF "$d" "$BACKUP/CREATED_DIRS" || echo "$d" >> "$BACKUP/CREATED_DIRS"; done
done
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r rel base new <<<"$row"
  if [ "$base" = NEW ]; then echo "$rel" >> "$BACKUP/ADDED"; continue; fi
  for pair in "src:$SRC" "venv:$VENV"; do
    name=${pair%%:*}; tree=${pair#*:}
    mkdir -p "$(dirname "$BACKUP/$name/$rel")"
    cp "$tree/$rel" "$BACKUP/$name/$rel"
  done
done
cp "$MANIFEST" "$BACKUP/MANIFEST"
(cd "$BACKUP" && find src venv -type f -print0 | sort -z | xargs -0 -r sha256sum > BACKUP.sha256)
(cd "$BACKUP" && sha256sum --quiet -c BACKUP.sha256) || die "backup did not verify"
echo "apply: originals saved in $BACKUP ($(wc -l < "$BACKUP/BACKUP.sha256") files, $(wc -l < "$BACKUP/ADDED") added paths listed)"

# 4-5. install, verify; on any failure restore from the backup
restore() {
  echo "apply: FAILED during install; restoring both trees from $BACKUP" >&2
  for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r rel base new <<<"$row"
    for pair in "src:$SRC" "venv:$VENV"; do
      name=${pair%%:*}; tree=${pair#*:}
      rm -f "$tree/$rel.kit-tmp"
      if [ "$base" = NEW ]; then rm -f "$tree/$rel"; else cp "$BACKUP/$name/$rel" "$tree/$rel"; fi
    done
  done
  tac "$BACKUP/CREATED_DIRS" | while read -r d; do rmdir "$SRC/$d" "$VENV/$d" 2>/dev/null || true; done
  exit 1
}
trap restore ERR
while read -r d; do
  mkdir "$SRC/$d" "$VENV/$d"; echo "apply: created directory $d in both trees"
done < "$BACKUP/CREATED_DIRS"
for tree in "$SRC" "$VENV"; do
  for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r rel base new <<<"$row"
    cp "$FILES/$rel" "$tree/$rel.kit-tmp"
    mv "$tree/$rel.kit-tmp" "$tree/$rel"
    echo "apply: $( [ "$base" = NEW ] && echo added || echo replaced) $tree/$rel"
  done
done
for tree in "$SRC" "$VENV"; do
  for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r rel base new <<<"$row"
    [ "$(sha "$tree/$rel")" = "$new" ] || { echo "apply: $tree/$rel does not verify" >&2; false; }
  done
done
trap - ERR
echo "apply: DONE, both trees verified (${#ROWS[@]} files each); rollback: rollback.sh $SRC $VENV $BACKUP"
