#!/usr/bin/env bash
# doc/work_log.md に節を足す。**時刻を手で書かない。本文をシェルに通さない。**
#
# 2026-10-26 に立てた。同じ日に2つの事故を踏んだので、規則ではなく器具にした:
#
#   1. **見出しの時刻の捏造を5回。** `date` を印字した同じコマンドの中で、
#      その出力に「書き終わるまでの時間」を足して書いていた。
#      -> 本文には `TIMEMARK` と書き、**このスクリプトが date で置き換える。**
#   2. **引用符無しのヒアドキュメントが本文の `` ` `` をコマンド置換で食った**
#      (見出しの `radius=4` が消えた。`git commit -am` でも同じ罠を踏んでいる)。
#      -> **本文はファイルで渡す**(Write で書く)。ヒアドキュメントを使わないので
#         引用の問題が起きない。シェルは本文を一度も解釈しない。
#
# 使い方:  tools/append_log.sh <本文のファイル>
#   本文の中の TIMEMARK は `YYYY-MM-DD HH:MM` に置き換わる。
#   TIMEMARK を1つも含まない本文は**拒否する**(見出しの時刻を手で書いた形なので)。
set -euo pipefail
cd "$(dirname "$0")/.."

src="${1:?使い方: tools/append_log.sh <本文のファイル>}"
[ -f "$src" ] || { echo "本文のファイルが無い: $src" >&2; exit 1; }

if ! grep -q 'TIMEMARK' "$src"; then
  echo "拒否: 本文に TIMEMARK が無い。**見出しの時刻を手で書かない** ——" >&2
  echo "      '## TIMEMARK 見出し' の形にすること(このスクリプトが date で置き換える)" >&2
  exit 2
fi

T="$(date '+%Y-%m-%d %H:%M')"
printf '\n' >> doc/work_log.md
sed "s/TIMEMARK/$T/g" "$src" >> doc/work_log.md
echo "-> doc/work_log.md に追記。時刻 $T を入れた見出し:"
sed "s/TIMEMARK/$T/g" "$src" | grep '^#' || true
