#!/usr/bin/env python3
"""**人の判定の錨を、取り直す前に読む。** 規律10 の器具。

規律10 は「動く錨は、動かす時期と条件を先に登録する」で、登録に書くのは
**いつ取り直すか / 誰に聞くか / 取り直したら前後の棒を両方、名前を付けて印字すること**。
この器具はその3条項が**書いてあるか**を点検し、**書いてある条件を印字する**。

**条件そのものは機械が判定できない**(「人の棒に乗る主張を新しく出すとき」は事象で、
ファイルには現れない)。だから `status` は期限を自分で決めない —— 言うのは
「登録が在るか」「引き金は何か」「**脚に動く余地が在るか**」の3つだけ。
最後の1つは記録から計算できる本物の点検で、天井/床に張り付いた脚で暦の取り直しを
しても確認しか返らないことを印字する(`names_are_not_checks`: 判定できない条件を
判定したふりをする器具は、器具が無いより悪い)。

**登録簿の数は記録から引き直す。** 登録 `0/15` と記録の再計算を並べて印字し、
食い違ったら終了コード1(規律2 の既知セルを登録簿に当てた形)。

使い方:
  judge_anchor.py [--registry <道>] status                      # 全錨の登録と再計算
  judge_anchor.py bars <錨A> <錨B>            # 前後の棒を両方、名前を付けて印字(規律10 第3条項)
  judge_anchor.py reread <旧錨> <新錨>        # 棒が食い違ったとき、どの主張を読み直すか(条例3)

終了コード: 0 = 指摘なし / 1 = 指摘あり(未登録・食い違い・読み直し対象あり) / 2 = 器具が走れない
**何も書き換えない。**
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config/judge_anchor_registry.json"

# 規律10 が登録に要求する3条項。**欠けたら黙らずに「未登録」と言う**
REQUIRED_CLAUSES = [
    ("events", "いつ取り直すか"),
    ("who", "誰に聞くか"),
    ("print_both", "取り直したら前後の棒を両方印字すること"),
]


def load_registry(path=None):
    path = Path(path) if path else REGISTRY
    if not path.exists():
        print(f"登録簿が無い: {path}", file=sys.stderr)
        sys.exit(2)
    return json.loads(path.read_text())


def recompute(anchor):
    """記録から腕ごとの数を引き直す。**形が分からなければ黙らずにそう言う。**"""
    path = ROOT / anchor["records"]
    if not path.exists():
        # **記録を指して実体が無いのは指摘である**(指している先が無い登録は点検ではない)
        return None, f"**記録が無い**: {anchor['records']}"
    docs = json.loads(path.read_text())
    docs = docs if isinstance(docs, list) else list(docs.values())
    if docs and "verdict" in docs[0]:        # 絶対判定(1枚ずつ)
        tally = defaultdict(lambda: [0, 0])
        for d in docs:
            t = tally[d["arm"]]
            t[1] += 1
            t[0] += 1 if d["verdict"] == "usable" else 0
        return {a: (k, n) for a, (k, n) in tally.items()}, None
    if docs and "choice" in docs[0]:         # 比較判定(2枚並べ)
        return None, f"比較判定の記録({len(docs)}件)。腕ごとの「使える率」は定義されない —— **棒ではない**(lesson 11)"
    return None, "記録の形に再計算器を持たない"


def pin_state(k, n):
    if n == 0:
        return "—"
    if k == 0:
        return "床に張り付き"
    if k == n:
        return "天井に張り付き"
    return f"中間({k}/{n}) = 動く余地あり"


def cmd_status(reg):
    bad = 0
    pr = reg.get("period_rule", {})
    print("══ 取り直す周期の決め方(登録済み)")
    print(f"   駆動      : {pr.get('driver')} —— {pr.get('derivation','')[:60]}…")
    for e in pr.get("events", []):
        print(f"   引き金    : {e}")
    print(f"   同一セッション: {'要' if pr.get('same_session') else '不要'} / 誰に: {pr.get('who')}")
    cal = pr.get("calendar")
    print(f"   暦        : {'登録なし' if cal is None else cal} —— {pr.get('calendar_reason','')}")
    print(f"   暦が意味を持つ条件: {pr.get('calendar_becomes_meaningful_when')}")

    for a in reg["anchors"]:
        print()
        print(f"══ {a['name']}  ({a['taken_on']} / {a['judge']})")
        print(f"   役割  : {a['role']}")
        print(f"   問い  : {a['question']}  単位: {a['unit']}")
        print(f"   母集団: {a['population']}")
        got, why = recompute(a)
        if got is None:
            if why.startswith("**記録が無い**"):
                bad += 1
            print(f"   再計算: **できない** —— {why}")
            print(f"   登録  : {json.dumps(a['result'], ensure_ascii=False)}")
        else:
            print("   登録 対 再計算(記録から引き直し):")
            print("     腕          登録     再計算   一致   脚の余地")
            for arm in a["arms"]:
                want = a["result"].get(arm, "—")
                if arm in got:
                    k, n = got[arm]
                    have = f"{k}/{n}"
                    ok = "○" if want == have else "**×**"
                    if want != have:
                        bad += 1
                    print(f"     {arm:11s} {want:8s} {have:8s} {ok:6s} {pin_state(k, n)}")
                else:
                    bad += 1
                    print(f"     {arm:11s} {want:8s} {'—':8s} **×**   記録に無い")
        rep = a.get("n_repeats")
        print(f"   再掲  : {rep} 件" + (f" —— {a['intra_rater_reason']}" if not rep and a.get("intra_rater_reason") else ""))
        print(f"   本人内一貫性: {a.get('intra_rater') or '**未測定**'}")

        rt = a.get("retake") or {}
        is_bar = a.get("is_bar")
        if is_bar is None:
            bad += 1
            print("   **`is_bar` が登録簿に無い** —— 棒かどうかを役割の文面から当てない(名前は点検ではない)")
            is_bar = True
        print("   取り直しの登録(規律10 の3条項):")
        for key, label in REQUIRED_CLAUSES:
            val = rt.get(key)
            # **指し手は解決して印字する。** 登録簿に写しを置かない型なので、ここで展開する
            if val == "period_rule.events":
                val = " / ".join(pr.get("events", [])) or None
            if val in (None, "", [], False):
                if not is_bar:
                    print(f"     {label:34s}: 対象外(**棒ではない**)")
                    continue
                bad += 1
                print(f"     {label:34s}: **未登録**")
            else:
                print(f"     {label:34s}: {val if not isinstance(val, bool) else '○'}")
        if rt.get("retro"):
            print(f"     事後登録: {rt.get('retro_note','')}")
    return bad


def frac(got, arm):
    if got and arm in got:
        k, n = got[arm]
        return k, n
    return None


def cmd_bars(reg, a_name, b_name):
    by = {a["name"]: a for a in reg["anchors"]}
    for n in (a_name, b_name):
        if n not in by:
            print(f"錨が登録簿に無い: {n}", file=sys.stderr)
            return 2
    A, B = by[a_name], by[b_name]
    ga, _ = recompute(A)
    gb, _ = recompute(B)
    print("══ 前後の棒を両方、名前を付けて印字(規律10 第3条項)")
    for lab, x in (("前", A), ("後", B)):
        print(f"   {lab}: {x['name']}  ({x['taken_on']} / {x['judge']} / {x['unit']})")
        print(f"       棒: {x['bar']}")
    print()
    print(f"   腕          {a_name:22s} {b_name:22s} 差(後−前)")
    arms = list(dict.fromkeys(A["arms"] + B["arms"]))
    for arm in arms:
        fa, fb = frac(ga, arm), frac(gb, arm)
        sa = f"{fa[0]}/{fa[1]}" if fa else "**未測定**"
        sb = f"{fb[0]}/{fb[1]}" if fb else "**未測定**"
        if fa and fb:
            d = f"{fb[0]/fb[1] - fa[0]/fa[1]:+.3f}"
        else:
            d = "読めない(片側が未測定)"
        print(f"   {arm:11s} {sa:22s} {sb:22s} {d}")
    print()
    print("   **行ごとに印字。腕を跨いで平均しない。片側が未測定の行は差を出さない**(規律1/8)")
    return 0


def standing_sentence(claim, old_anchor, new_anchor, gn):
    """条例3 の「書ける形」。新錨での扱いは**実測から埋める** —— 決め打ちにしない。"""
    arms = (claim.get("flips_if") or {}).get("arms") or []
    if old_anchor["unit"] != new_anchor["unit"]:
        state = f"比較不能(単位が違う: {old_anchor['unit']} 対 {new_anchor['unit']})"
    elif not arms:
        state = "この主張は腕に乗っていない(棒の版と独立)"
    elif all(frac(gn, a) is None for a in arms):
        state = "未測定"
    else:
        rows = []
        for a in arms:
            f = frac(gn, a)
            rows.append(f"{a} {f[0]}/{f[1]}" if f else f"{a} 未測定")
        state = "同じ向き(" + " / ".join(rows) + ")"
    return (f"{claim['statement']} —— {old_anchor['name']}"
            f"({old_anchor['taken_on']} / {old_anchor['judge']} / {old_anchor['population'][:28]}… / "
            f"{old_anchor['unit']})に対して。{new_anchor['name']} では{state}。")


def cmd_reread(reg, old, new):
    by = {a["name"]: a for a in reg["anchors"]}
    if old not in by or new not in by:
        print("錨が登録簿に無い", file=sys.stderr)
        return 2
    go, _ = recompute(by[old])
    gn, _ = recompute(by[new])
    rule = reg["disagreement_rule"]
    print(f"══ 棒が食い違ったとき、どれを読み直すか({old} → {new})")
    print("   読み直す条件(全て満たすときだけ):")
    for c in rule["reread_requires_all_of"]:
        print(f"     - {c}")
    print(f"   満たさなければ: {rule['otherwise']}")
    print()
    groups = defaultdict(list)
    for c in reg["claims"]:
        if c["rode_on"] not in (old,):
            groups["人の棒(この錨)に乗っていない"].append((c, "棒の版が動いても動かない"))
            continue
        art = ROOT / c["artifacts"]
        if not art.exists():
            groups["再採点不能(消さない)"].append((c, f"腕の実体が無い: {c['artifacts']}"))
            continue
        fi = c.get("flips_if", {})
        moved = []
        for arm in fi.get("arms", []):
            fa, fb = frac(go, arm), frac(gn, arm)
            if not (fa and fb):
                moved.append(f"{arm}: 片側未測定")
                continue
            da = fa[0] / fa[1]
            db = fb[0] / fb[1]
            if fi.get("direction") == "up" and db > da:
                moved.append(f"{arm}: {fa[0]}/{fa[1]} → {fb[0]}/{fb[1]} **上がった**")
            elif fi.get("direction") == "down" and db < da:
                moved.append(f"{arm}: {fa[0]}/{fa[1]} → {fb[0]}/{fb[1]} **下がった**")
        if moved:
            groups["読み直す"].append((c, "; ".join(moved)))
        else:
            groups["錨札つきで立つ"].append((c, "反転させうる向きに動いていない"))
    n_reread = len(groups["読み直す"])
    for title in ("読み直す", "錨札つきで立つ", "再採点不能(消さない)", "人の棒(この錨)に乗っていない"):
        items = groups.get(title) or []
        print(f"   [{title}] {len(items)} 件")
        for c, why in items:
            print(f"     {c['id']}: {c['statement'][:62]}")
            print(f"        理由: {why}  / 記録の持ち主: {c['owner_of_record']}")
            if title != "読み直す":
                a = by.get(c["rode_on"])
                if a:
                    print("        書ける形: " + standing_sentence(c, a, by[new], gn))
    print()
    print("   **撤回ではない。古い錨の行は消さない**(規律10)。新しい行を足して両方に錨札を付ける(規律1)")
    return 1 if n_reread else 0


def main():
    argv = sys.argv[1:]
    reg_path = None
    if argv and argv[0] == "--registry":
        if len(argv) < 2:
            print("--registry に道が無い", file=sys.stderr)
            return 2
        reg_path, argv = argv[1], argv[2:]
    reg = load_registry(reg_path)
    cmd = argv[0] if argv else "status"
    if cmd == "status":
        bad = cmd_status(reg)
        print()
        print(f"指摘 {bad} 件" if bad else "\n指摘なし")
        return 1 if bad else 0
    if cmd == "bars" and len(argv) == 3:
        return cmd_bars(reg, argv[1], argv[2])
    if cmd == "reread" and len(argv) == 3:
        return cmd_reread(reg, argv[1], argv[2])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
