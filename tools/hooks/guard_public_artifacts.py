#!/usr/bin/env python3
"""許可表の門を**ハーネスに**回させる。`tools/check_public_artifacts.py` の包み。

CLAUDE.md §4 の規則は器具を持っているのに、担保が「Claude が思い出して回す」だった
(2026-10-30、ops-review が実測)。**沈黙が正常動作の点検は、忘れた日に何も起きない。**
これは忘れても回る形にするための包みで、判定そのものは一切しない ——
表に無い追跡画像・追跡重みが在るかは `check_public_artifacts.py` が決める。

使い方 (hook の `command`):
  guard_public_artifacts.py block   -- 止める。PreToolUse で deny を返す
  guard_public_artifacts.py warn    -- 止めない。systemMessage だけ出す

終了コードは常に 0。**止めるのは JSON の permissionDecision で、例外や終了コードでは
止めない** —— 器具が走れなかったこと(終了コード 2)を「表と一致」とも「違反」とも
読ませないため。走れなかったときは、そう書いて通す。
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CHECKER = ROOT / "tools" / "check_public_artifacts.py"


def emit(obj):
    print(json.dumps(obj, ensure_ascii=False))
    return 0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "warn"
    sys.stdin.read()  # hook の入力は読み捨てる(どの木かは ROOT で決まる)

    if not CHECKER.exists():
        return emit({"systemMessage": f"許可表の門が見つからない: {CHECKER}(点検は走っていない)"})

    r = subprocess.run([sys.executable, "-I", str(CHECKER)],
                       capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode == 0:
        return 0
    if r.returncode != 1:
        return emit({"systemMessage":
                     f"許可表の門が走れなかった(終了コード {r.returncode})。"
                     f"**点検していない**ので、表との一致は不明:\n{r.stderr.strip()[:800]}"})

    # 器具は木ごとに1行出す。**0 件の木は捨て**、違反した木とその道、件数だけ残す。
    lines = [x for x in r.stdout.strip().splitlines() if x.strip()]
    head = [x for x in lines if not x.rstrip().endswith("表に無い 0")
            and "計 0 件" not in x and "5MB 超" not in x and " MB" not in x
            and not re.match(r"^\s+[\d.]+ MB", x)]
    total = []
    reason = ("**表に無い追跡画像か追跡重みが在る**(`tools/check_public_artifacts.py` 終了コード1)。\n"
              "許可表は「選んだものの表」で、足すのはユーザーの承認。Claude は自分で足さない。\n"
              + "\n".join(head + total))
    if mode == "block":
        return emit({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        })
    return emit({"systemMessage": reason})


if __name__ == "__main__":
    sys.exit(main())
