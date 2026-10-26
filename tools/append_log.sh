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

# **置換は1行目(見出し)だけ。** 2026-10-26 に基盤が踏んだ欠陥:
# この器具の仕組みを記録に書くと、**説明文の中の置き札の名前まで時刻に置き換わり**、
# 「`15:20` を date で置換」という無意味な文になった。
# **器具が、自分の名前を含む文書を壊す。** 置き札を記録に書く日は必ず来る(来た)。
# -> 1行目だけを置換する。見出し以外に時刻が要ることは無い。
#    2行目以降の TIMEMARK は**そのまま残す**(論じている文なので、それが正しい)。
if ! head -1 "$src" | grep -q 'TIMEMARK'; then
  echo "拒否: 本文の**1行目**に TIMEMARK が無い。**見出しの時刻を手で書かない** ——" >&2
  echo "      1行目を '## TIMEMARK 見出し' の形にすること" >&2
  echo "      (置換は1行目だけ。2行目以降の TIMEMARK は本文として残る)" >&2
  exit 2
fi

T="$(date '+%Y-%m-%d %H:%M')"
printf '\n' >> doc/work_log.md
sed "1s/TIMEMARK/$T/g" "$src" >> doc/work_log.md
echo "-> doc/work_log.md に追記。時刻 $T を入れた見出し:"
sed "1s/TIMEMARK/$T/g" "$src" | head -1
if tail -n +2 "$src" | grep -q 'TIMEMARK'; then
  n="$(tail -n +2 "$src" | grep -c 'TIMEMARK')"
  echo "   (2行目以降の TIMEMARK $n 件はそのまま残した —— 論じている文なので)"
fi
