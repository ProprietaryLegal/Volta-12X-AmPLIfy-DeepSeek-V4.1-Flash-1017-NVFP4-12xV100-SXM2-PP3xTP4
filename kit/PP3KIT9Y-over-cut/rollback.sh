#!/usr/bin/env bash
# rollback.sh SRC VENV BACKUP: restore both trees and remove the owned packed-576 link.
set -euo pipefail
[ $# = 3 ] || { echo "usage: rollback.sh SRC VENV BACKUP" >&2; exit 2; }
kit=$(cd "$(dirname "$0")" && pwd)
pack=$(realpath "$(dirname "$3")")/9X-packed576
read -r target < "$3/PACKED576"
[ -L "$pack" ] && [ "$(readlink -- "$pack")" = "$target" ] ||
  { echo "rollback: packed-576 link is missing or changed: $pack" >&2; exit 1; }
bash "$kit/rollback_files.sh" "$@"
unlink "$pack"
echo "rollback: removed $pack"
