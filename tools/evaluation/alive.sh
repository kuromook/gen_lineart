#!/bin/bash
# 走行の生存確認 (罠 T105: 「起動した」と「走っている」は別の状態)。
#
#   起動シェルは本体を切り離した時点で終了するので、**本体が落ちても通知が来ない。**
#   L の `--exclude-gapfill` 版は、承認も登録も済んで **0頁** だった。
#
# 使い方: tools/evaluation/alive.sh '<ps で探す語>' <ログ>
#   起動の直後ではなく、**少し待ってから** 2つを見る:
#     ① 過程が生きていて CPU を使っているか
#     ② ログが**進んでいるか**(生きているが進んでいない、を分ける)
set -u
PAT="$1"; LOG="${2:-}"
echo "== $(date '+%H:%M:%S') 生存確認: $PAT"
HIT=$(ps -eo pid,etime,pcpu,args | awk -v p="$PAT" 'index($0,p) && !/awk/ && !/alive.sh/ {print $1" "$2" "$3"%"}')
if [ -z "$HIT" ]; then
    echo "  **過程が無い。走っていない。**"
    [ -n "$LOG" ] && { echo "  --- ログの末尾 (落ちた理由はここに在る):"; tail -12 "$LOG"; }
    exit 1
fi
echo "$HIT" | sed 's/^/  /'
if [ -n "$LOG" ]; then
    A=$(wc -l < "$LOG")
    sleep 45
    B=$(wc -l < "$LOG")
    if [ "$A" -eq "$B" ]; then
        echo "  ログは45秒で進んでいない ($B 行)。進捗の印字が疎なだけかもしれないので、"
        echo "  **CPU が 0 でないことと合わせて読む**"
    else
        echo "  ログが進んでいる: $A -> $B 行"
        tail -1 "$LOG" | sed 's/^/    /'
    fi
fi
