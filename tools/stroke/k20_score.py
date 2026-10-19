#!/usr/bin/env python
"""K20 K19 の採点器。**回答を受け取る前に書く**(基盤 `reply_k_bridging_approved` §1 の助言)。

帯は `key.json` に開封前宣言で焼き込んであるので、先に書いても帯は動かない。
逆に回答を見てから書くと、**回答の分布を見ながら集計の形を決められる。**

`--answers` 無しで走らせると**入力なしで通せる確認だけ**を回す(dry run):
  器具確認の構造、帯の読み替え、幅の層化、本番 28 件が ≤8px の全数であること。

採点の順序は登録どおり **器具確認を先に読み、主値は最後**。
"""
import argparse, csv, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent.parent.parent
NAME = {1: "飛ばし線", 2: "かすれ", 3: "切れていない", 4: "判断できない"}
NULL_PRECISION = 0.225          # 帰無値(統計量 = 精度)。K10 本番 40 件の かすれ 9/40
BAR = 0.50                      # 棒。費用の恒等式 比 = (1-精度)/精度 から
WIDE = 8.0                      # ベタの尺度(K4a)


def cp95(k, n):
    if n == 0:
        return None, None
    return (0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(.975, k + 1, n - k)))


def okey(d):
    """同定の鍵。**出力の中の位置から振らない**(K10 訂正 4 の一般形)。"""
    return (int(d["panel"]), int(d["s0"]), int(d["s1"]), int(d["e0"]), int(d["e1"]),
            round(float(d["gap"]), 2))


def parse_answers(txt):
    """`#1:1 #2:4 ...` を読む。`$26:2` のような打ち間違いも拾う。"""
    out = {}
    for m in re.finditer(r"[#$]?\s*(\d+)\s*[:：]\s*([1-4])", txt):
        out[int(m.group(1))] = int(m.group(2))
    return out


def chance_agreement(marg, items):
    """帰無模型: 周辺分布から層ごとに独立に引く。一致数の厳密分布を返す。"""
    ps = [marg.get(NAME[a], 0.0) for a in items]
    dist = np.zeros(len(ps) + 1); dist[0] = 1.0
    for p in ps:
        nd = np.zeros(len(ps) + 1)
        for i in range(len(ps)):
            nd[i] += dist[i] * (1 - p); nd[i + 1] += dist[i] * p
        nd[len(ps)] += dist[len(ps)]
        dist = nd
    return ps, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="results/k19_author_request_20261019/key.json")
    ap.add_argument("--k10", default="results/k10_author_request_20261018")
    ap.add_argument("--k18", default="results/k18_gap_cost_20261019/per_pair.csv")
    ap.add_argument("--widths", default="results/k19_author_request_20261019/tip_widths.json")
    ap.add_argument("--answers", help="回答のテキスト(無ければ dry run)")
    ap.add_argument("--out", default="results/k19_author_request_20261019/score.json")
    a = ap.parse_args()

    K = json.load(open(ROOT / a.key))
    key, reg = K["key"], K["registered"]
    by_kind = defaultdict(list)
    for did, it in key.items():
        by_kind[it["kind"]].append((int(did), it))
    per = {(r["panel"], r["s0"], r["s1"], r["e0"], r["e1"]): r
           for r in csv.DictReader(open(ROOT / a.k18))}
    widths = {}
    if (ROOT / a.widths).exists():
        widths = {int(k): v for k, v in json.load(open(ROOT / a.widths)).items()}

    print("=" * 72)
    print("登録(key.json に焼き込み済み。**結果を見てから動かさない**)")
    for k in ("main_is_census", "reading", "power", "anchor_rule", "blinding"):
        if reg.get(k):
            print(f"  {k}: {reg[k]}")
    for k, v in reg["bands"].items():
        print(f"  帯 {k}: {v}")
    print("=" * 72)

    # --- 構造の確認(入力なしで通る) ---
    n_main = len(by_kind["main"])
    gaps = [float(it["gap"]) for _d, it in by_kind["main"]]
    struct = dict(n_main=n_main, n_repeat=len(by_kind["repeat"]),
                  n_anchor=len(by_kind["anchor"]), n_cross=len(by_kind["cross"]),
                  main_gap_max=round(max(gaps), 2), main_gap_min=round(min(gaps), 2),
                  all_main_le_8=bool(max(gaps) <= 8.0),
                  repeat_keys_in_main=sum(1 for _d, it in by_kind["repeat"]
                                          if okey(it) in {okey(x) for _y, x in by_kind["main"]}),
                  anchors_distinct_pairs=len({(it["panel"], it["s0"], it["s1"])
                                              for _d, it in by_kind["anchor"]}),
                  cross_has_prev=sum(1 for _d, it in by_kind["cross"]
                                     if it.get("prev_answer")))
    print("\n[構造] " + json.dumps(struct, ensure_ascii=False))
    ok = (struct["n_main"] == 28 and struct["all_main_le_8"]
          and struct["repeat_keys_in_main"] == struct["n_repeat"]
          and struct["anchors_distinct_pairs"] == struct["n_anchor"]
          and struct["cross_has_prev"] == struct["n_cross"])
    print(f"  構造の確認: {'合格' if ok else '**不合格 —— 止めて相談**'}")

    # --- 帯の読み替え(入力なしで通る) ---
    k10ans = json.load(open(ROOT / a.k10 / "answers.json"))
    marg = {k: v / k10ans["n_main"] for k, v in k10ans["primary"].items()}
    det = [it["prev_answer"] for _d, it in by_kind["cross"] if it["prev_answer"] != 4]
    und = [it for _d, it in by_kind["cross"] if it["prev_answer"] == 4]
    ps, dist = chance_agreement(marg, det)
    band = next(k for k in range(len(ps) + 1) if dist[k:].sum() <= 0.05)
    print(f"\n[帯の読み替え] 周辺分布 " +
          " ".join(f"{k} {v:.3f}" for k, v in marg.items()))
    print(f"  採点する確定回答 {len(det)} 件 一致確率 {[round(p,3) for p in ps]}"
          f"  期待 {sum(ps):.3f}")
    print(f"  **帯 {band}/{len(det)}(上側 {dist[band:].sum():.4f})** / "
          f"採点しない未決定 {len(und)} 件")

    # --- 幅の層化(作者の質問で足した) ---
    if widths:
        w_main = [widths[d] for d, _i in by_kind["main"] if d in widths]
        w_anc = [widths[d] for d, _i in by_kind["anchor"] if d in widths]
        print(f"\n[幅の層化] 本番の幅 中央値 {np.median(w_main):.2f}px "
              f"(>{WIDE}px: {sum(1 for w in w_main if w > WIDE)}/{len(w_main)})  "
              f"錨 中央値 {np.median(w_anc):.2f}px "
              f"(>{WIDE}px: {sum(1 for w in w_anc if w > WIDE)}/{len(w_anc)})")
        print("  **錨は幅で見分けられる。錨の帯はこの限界つきで読む**(登録済みの欠陥)")

    if not a.answers:
        print("\n[dry run] 回答が無いのでここまで。"
              "器具確認の構造・帯の読み替え・幅の層化は入力なしで通った。")
        return

    # ---------------- ここから回答が要る ----------------
    txt = Path(a.answers).read_text() if Path(a.answers).exists() else a.answers
    ans = parse_answers(txt)
    missing = [d for d in sorted(int(x) for x in key) if d not in ans]
    print(f"\n[回答] {len(ans)}/{len(key)} 件" +
          (f"  **欠け {missing}**" if missing else ""))

    res = {}
    # IC-1 反復
    firsts = {okey(it): ans.get(d) for d, it in by_kind["main"] if d in ans}
    pairs = [(ans[d], firsts.get(okey(it))) for d, it in by_kind["repeat"]
             if d in ans and firsts.get(okey(it))]
    agree = sum(1 for x, y in pairs if x == y)
    mrg = Counter([ans[d] for d, _i in by_kind["main"] + by_kind["repeat"] if d in ans])
    tot = sum(mrg.values())
    chance = sum((v / tot) ** 2 for v in mrg.values()) if tot else 0.0
    mode = max(mrg.values()) / tot if tot else 0.0
    res["repeat"] = dict(agree=agree, n=len(pairs), chance=round(chance, 4),
                         expected=round(chance * len(pairs), 2), mode_share=round(mode, 3),
                         marginal={NAME[k]: v for k, v in sorted(mrg.items())},
                         verdict=("判定不能(1 つの選択肢が 7 割以上)" if mode >= 0.70 else
                                  "合格" if (agree >= 6 and agree > chance * len(pairs))
                                  else "**不合格**"))
    # IC-2 錨(幅で層化)
    av = [(d, ans[d], widths.get(d)) for d, _i in by_kind["anchor"] if d in ans]
    a3 = sum(1 for _d, v, _w in av if v == 3); a1 = sum(1 for _d, v, _w in av if v == 1)
    res["anchor"] = dict(n=len(av), not_cut=a3, tobashi=a1,
                         by_answer={NAME[k]: v for k, v in sorted(Counter(
                             v for _d, v, _w in av).items())},
                         wide=dict(n=sum(1 for _d, _v, w in av if w and w > WIDE),
                                   not_cut=sum(1 for _d, v, w in av if w and w > WIDE and v == 3)),
                         thin=dict(n=sum(1 for _d, _v, w in av if w and w <= WIDE),
                                   not_cut=sum(1 for _d, v, w in av if w and w <= WIDE and v == 3)),
                         verdict=("合格" if (a3 >= 6 and a1 <= 1) else "**不合格**"),
                         caveat="**錨は幅で見分けられる(本番 2.00px 対 錨 7.99px)。"
                                "合格しても『提示が接続を隠していない』の証拠としては弱い**")
    # IC-3 突き合わせ
    cd = [(ans[d], it["prev_answer"]) for d, it in by_kind["cross"]
          if d in ans and it["prev_answer"] != 4]
    cu = [(ans[d], it["prev_answer"]) for d, it in by_kind["cross"]
          if d in ans and it["prev_answer"] == 4]
    ca = sum(1 for x, y in cd if x == y)
    res["cross"] = dict(n_scored=len(cd), agree=ca, band=band,
                        expected=round(sum(ps), 3), upper=round(float(dist[band:].sum()), 4),
                        verdict=("合格" if ca >= band else "**不合格**"),
                        undecided_n=len(cu),
                        undecided_still_4=sum(1 for x, _y in cu if x == 4),
                        undecided_note="**採点しない。**確定側に動いたなら漂流ではなく"
                                       "境界が鋭くなったということ")
    for k in ("repeat", "anchor", "cross"):
        print(f"\n[器具確認 {k}] " + json.dumps(res[k], ensure_ascii=False))

    # --- 主値 ---
    prev = {}
    pk = json.load(open(ROOT / a.k10 / "key.json"))["key"]
    pa_ = json.load(open(ROOT / a.k10 / "answers.json"))["answers"]
    for did, it in pk.items():
        if it["kind"] == "main":
            prev[(int(it["panel"]), int(it["s0"]), int(it["s1"]))] = int(pa_[did])
    rows = []
    for d, it in by_kind["main"]:
        if d in ans:
            rows.append((float(it["gap"]), ans[d]))
    for r in per.values():
        if r["answer"] != "0" and float(r["gap_sk"]) <= 8.0:
            rows.append((float(r["gap_sk"]), int(r["answer"])))
    res["primary"] = {}
    for th in (6.0, 8.0):
        sel = [v for g, v in rows if g <= th]
        k = sum(1 for v in sel if v == 2)
        lo, hi = cp95(k, len(sel))
        verdict = ("棒を越えた" if lo and lo > BAR else
                   "**軸は死ぬ**" if hi and hi < BAR else
                   "判定保留(軸は生きているが材料で確認できない)")
        res["primary"][f"gap<={th:g}"] = dict(
            n=len(sel), kasure=k, precision=round(k / len(sel), 4) if sel else None,
            cp95=[round(lo, 4), round(hi, 4)] if sel else None,
            ratio=round((len(sel) - k) / k, 3) if k else None,
            by_answer={NAME[x]: sum(1 for v in sel if v == x) for x in (1, 2, 3, 4)},
            null=NULL_PRECISION, bar=BAR, verdict=verdict)
        print(f"\n[主値 隙間<={th:g}px] " +
              json.dumps(res["primary"][f"gap<={th:g}"], ensure_ascii=False))
    res["limits"] = [
        "**盲検ではない。**作者は『短い隙間ほどかすれ、精度 0.625』という分析を読んでいる",
        "**錨は幅で見分けられる**(本番 2.00px 対 錨 7.99px)。錨の帯はこの限界つき",
        "≤8px の足跡 96 件のうち聞けるのは 42 件。**残り 54 件は誰にも答えが出ない**",
        "材料は全部 train(既読)。確認には新しい顔が要る",
    ]
    json.dump(dict(registered=reg, structure=struct, result=res),
              open(ROOT / a.out, "w"), indent=1, ensure_ascii=False)
    print("\n[限界] " + "\n         ".join(res["limits"]))
    print(f"\n出力: {ROOT / a.out}")


if __name__ == "__main__":
    main()
