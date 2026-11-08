#!/usr/bin/env python3
"""**器具が無かった2つの規則**に器具を付ける。`git add` を見る PreToolUse の門。

CLAUDE.md §4 の2行は、2026-10-30 の棚卸しの時点で**文字しか無かった**:

  - **原稿由来の納品物は `data/` 配下。絶対にコミットしない**
  - **`.gitignore` は門ではない** —— ignore は**既に追跡されたもの**も **`git add -f`** も止めない

事故の形はこれで、2026-10-23 に `results/**` が ignore 済みなのに原寸の切り出し2枚が
入った。**ignore を足しても再発は止まらない**(`-f` は ignore を越える)。止まるのは
`-f` そのものを見る門だけである。

判定:
  deny  -- `-f`/`--force` 付きで、`data/` 配下か監視対象の拡張子を足そうとしている
  warn  -- `-f`/`--force` 付きで、上のどちらでもない(越えた事実だけ報せる)
  黙る  -- `-f` が無い

**道の解決はしない。** `git add -f .` のような広い指定は中身を列挙しないと分からないので、
**広い指定は `-f` が付いていれば deny する**(狭めて出し直すのは安い。黙って通すのは高い)。
終了コードは常に 0 —— 止めるのは JSON の permissionDecision。
"""
import json
import re
import shlex
import sys
from pathlib import Path

WATCHED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".psd", ".clip",
           ".pth", ".pt", ".ckpt", ".safetensors", ".onnx", ".bin", ".h5", ".pb"}
# `-f` は短い旗の束の中にも在る(`git add -fv` は妥当)。集合の一致では取りこぼす。
FORCE = re.compile(r"^(--force$|-[A-Za-z]*f)")
# `add` の引数のうち、道ではないもの
NOT_A_PATH = re.compile(r"^-")
BROAD = {".", "./", "*", "-A", "--all", "-u", "--update", ":/"}


def git_add_segments(command):
    """`&&` `;` `|` で切って、`git add` で始まる区間だけ返す。"""
    out = []
    for seg in re.split(r"&&|\|\||;|\n|\|", command):
        try:
            words = shlex.split(seg)
        except ValueError:
            words = seg.split()
        if not words:
            continue
        # `git`, `git -C <dir>`, `env ... git` のいずれでも `add` を見つける
        if "git" not in words:
            continue
        i = words.index("git")
        rest = words[i + 1:]
        j = 0
        while j < len(rest) and rest[j].startswith("-"):
            j += 2 if rest[j] in ("-C", "-c", "--git-dir", "--work-tree") else 1
        if j < len(rest) and rest[j] == "add":
            out.append(rest[j + 1:])
    return out


def judge(args):
    """(decision, 理由) を返す。decision は 'deny' / 'warn' / None。"""
    if not any(FORCE.match(a) for a in args):
        return None, ""
    paths = [a for a in args if not NOT_A_PATH.match(a)]
    if not paths or BROAD & set(paths):
        return "deny", ("`git add -f` を**広い指定**(`.` / `*` / `-A` など)で使っている。"
                        "何が入るか読めないので止める。**道を名指しして出し直す**。")
    bad = []
    for p in paths:
        s = Path(p)
        if "data" in s.parts:
            bad.append(f"{p} —— **`data/` 配下は原稿由来の納品物。絶対にコミットしない**")
        elif s.suffix.lower() in WATCHED:
            bad.append(f"{p} —— 監視対象(画像/重み)。**許可表に足すのはユーザーの承認**で、"
                       "Claude は自分で足さない")
    if bad:
        return "deny", ("`git add -f` が ignore を越えて次を足そうとしている:\n  "
                        + "\n  ".join(bad))
    return "warn", ("`git add -f` は **ignore を越える**(`.gitignore` は門ではない)。"
                    f"足すのは {' '.join(paths)} —— 監視対象でも `data/` でもないので止めないが、"
                    "越えたことは残す。")


def main():
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return 0
    command = (payload.get("tool_input") or {}).get("command") or ""

    worst, reasons = None, []
    for args in git_add_segments(command):
        d, why = judge(args)
        if d == "deny":
            worst, reasons = "deny", reasons + [why]
        elif d == "warn" and worst != "deny":
            worst, reasons = "warn", reasons + [why]
    if worst is None:
        return 0

    text = "\n".join(reasons)
    if worst == "deny":
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": text,
        }}, ensure_ascii=False))
    else:
        print(json.dumps({"systemMessage": text}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
