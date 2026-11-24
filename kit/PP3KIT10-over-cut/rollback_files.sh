#!/usr/bin/env bash
# rollback.sh <engine source vllm dir> <venv site-packages vllm dir> <backup dir>
#
# The single file-level revert of a kit applied by apply.sh: restores the originals saved in the backup dir and
# removes the files the kit added, in both trees, then verifies the baseline sha256 of every file. It reads the
# MANIFEST that apply.sh saved in the backup dir. It refuses, changing nothing, unless the backup verifies and every
# kitted file in both trees is either the kit's version or already the baseline (a file changed by someone else
# since is not overwritten). Nothing else is touched.
set -euo pipefail

[ $# -eq 3 ] || { echo "usage: rollback.sh <engine source vllm dir> <venv vllm dir> <backup dir>" >&2; exit 2; }
SRC=$(realpath "$1"); VENV=$(realpath "$2"); BACKUP=$(realpath "$3")
die() { echo "rollback: REFUSED: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }

MANIFEST=$BACKUP/MANIFEST
[ -f "$MANIFEST" ] && [ -f "$BACKUP/BACKUP.sha256" ] && [ -f "$BACKUP/ADDED" ] || die "$BACKUP is not an apply.sh backup"
CREATED=$BACKUP/CREATED_DIRS
[ -f "$CREATED" ] || : > /dev/null  # backups from before directory support have none
(cd "$BACKUP" && sha256sum --quiet -c BACKUP.sha256) || die "backup does not verify"
mapfile -t ROWS < <(grep -v '^#' "$MANIFEST" | grep -v '^$')

for tree in "$SRC" "$VENV"; do
  for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r rel base new <<<"$row"
    if [ -e "$tree/$rel" ]; then
      got=$(sha "$tree/$rel")
      [ "$got" = "$new" ] || [ "$got" = "$base" ] || die "$tree/$rel is neither the kit's nor the baseline (sha256 $got)"
    else
      [ "$base" = NEW ] || die "$tree/$rel is missing"
    fi
  done
done

for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r rel base new <<<"$row"
  for pair in "src:$SRC" "venv:$VENV"; do
    name=${pair%%:*}; tree=${pair#*:}
    if [ "$base" = NEW ]; then
      rm -f "$tree/$rel"; echo "rollback: removed $tree/$rel"
    else
      cp "$BACKUP/$name/$rel" "$tree/$rel.kit-tmp"; mv "$tree/$rel.kit-tmp" "$tree/$rel"
      echo "rollback: restored $tree/$rel"
    fi
  done
done

# directories apply created: removed deepest first; each must be empty now (anything else in it is not ours)
if [ -f "$CREATED" ]; then
  tac "$CREATED" | while read -r d; do
    for tree in "$SRC" "$VENV"; do
      rm -rf "$tree/$d/__pycache__"
      rmdir "$tree/$d" || die "$tree/$d is not empty after removing the kit's files"
      echo "rollback: removed directory $tree/$d"
    done
  done
fi

for tree in "$SRC" "$VENV"; do
  for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r rel base new <<<"$row"
    if [ "$base" = NEW ]; then
      [ ! -e "$tree/$rel" ] || die "$tree/$rel still exists"
    else
      [ "$(sha "$tree/$rel")" = "$base" ] || die "$tree/$rel does not have its baseline sha256 after restore"
    fi
  done
done
echo "rollback: DONE, both trees hold the baseline for every kitted file (${#ROWS[@]} each)"
