#!/usr/bin/env python
"""**便B(v3、機械が先に答えた表)の答えの取り込み** —— 宛先は Track R(写し E)。採点はしない(便B は学習専用で棒を読まない)。

**答えを見る前に決めたこと**:
- 答えのファイルは sha256 で固定する。固定の列(頁〜種類)が作った表(`_le20_prefill`)と全行一致し、鍵の並び(項目, 上から)とも一致すること
- 機械が閉じた行は、作者の欄に書いてある機械の答えを取らず、`answer_source` = `machine_empty_c` / `machine_exclusion_confirmed` の札で
  機械の答えを書く。作者がその行のコメント欄に書いたら逐語で残す(機械の答えは書き換えない)
- 作者が答えた行は `answer_source` = `author`。空欄は「未記入」。コメントは逐語
- 第2回の層名の聞き取り(軸2)の答えがある名前の行には、その答えを別の列 `axis2_round2` に並べる(混ぜない。Q と R が両方の札で共有する)

    ./venv/bin/python tools/census/stage_hearing_round2_b_answers.py
"""
import collections
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KEY = "data/census_full/hearing_round2_b_key_20261110_le20.csv"
MACHINE = "data/census_full/hearing_round2_b_machine_answers_20261112.csv"
XLSX = "data/census_full/hearing_round2_b_bundle{b}_20261110_le20_prefill.xlsx"     # v3: 選択肢に「テンプレート」(B-1・B-2 はこれで答えた)
XLSX_R = "data/census_full/hearing_round2_b_bundle{b}_20261110_le20_prefill_r.xlsx"  # v4: R の選択肢(グレー彩色・ベタ・陰影を足し、テンプレートを外す)
SHEET_VERSION = {"B-1": "v3", "B-2": "v3"}   # ここに無い束は v4(作者に渡した表の版。答える前に決まる)
AXIS2 = "data/census_full/name_hearing_round2_answers_20261112.csv"
BUNDLES = [f"B-{i}" for i in range(1, 8)]
ANSWERED = {"B-1": ("data/census_full/hearing_round2_b_bundleB-1_answered_20261112.xlsx",
                    "ce72c9097b0ec2257f76ef674bd64cbac2a56caadb6dc18ccf8639713a2016cc", "2026-11-12"),
            "B-2": ("data/census_full/hearing_round2_b_bundleB-2_answered_20261112.xlsx",
                    "2ee33cecb337d1c32b7ab36fe88a0f90787c08de50caaf3923bd1c4f7572676c", "2026-11-12")}
OUT = "data/census_full/hearing_round2_b_answers.csv"


def main() -> int:
    from openpyxl import load_workbook
    print("[便B の答えの取り込み(学習専用。採点しない)]")
    key = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(ROOT, KEY), encoding="utf-8")):
        key[r["bundle"]].append(r)
    mach = {(r["page_path"], r["layer_id"]): r for r in csv.DictReader(open(os.path.join(ROOT, MACHINE), encoding="utf-8"))}
    ax2 = {(r["layer_name"], r["v_layer_kind"]): r["answer"] for r in csv.DictReader(open(os.path.join(ROOT, AXIS2), encoding="utf-8"))
           if r["layer_name"]}
    out = []
    for b in BUNDLES:
        if b not in ANSWERED:
            n_m = sum(1 for k in key[b] if (k["page_path"], k["layer_id"]) in mach)
            print(f"    {b}: 未回収({len(key[b])}行、うち機械 {n_m})")
            continue
        path, sha, as_of = ANSWERED[b]
        got = hashlib.sha256(open(os.path.join(ROOT, path), "rb").read()).hexdigest()
        if got != sha:
            print(f"    **{b}: sha256 が違う** {got[:16]}")
            return 1
        a = list(load_workbook(os.path.join(ROOT, path))["層"].iter_rows(values_only=True))
        ver = SHEET_VERSION.get(b, "v4")
        o = list(load_workbook(os.path.join(ROOT, (XLSX if ver == "v3" else XLSX_R).format(b=b)))["層"].iter_rows(values_only=True))
        if len(a) != len(o) or any(x[:8] != y[:8] for x, y in zip(a, o)):
            print(f"    **{b}: 固定の列が作った表と違う**")
            return 1
        c = collections.Counter()
        for k, r in zip(key[b], a[1:]):
            if (int(k["item"]), int(k["pos"])) != (r[0], r[3]):
                print(f"    **{b}: 鍵と表の並びが違う**")
                return 1
            m = mach.get((k["page_path"], k["layer_id"]))
            if m:
                stage, src = m["machine_answer"], m["answer_tag"]
            else:
                stage, src = (r[8] or "未記入"), "author"
            c[(src, stage)] += 1
            out.append(dict(bundle=b, item=k["item"], pos=k["pos"], page_path=k["page_path"], layer_id=k["layer_id"], name=k["name"],
                            stage=stage, answer_source=src, author_comment=r[9] or "", axis2_round2=ax2.get((k["name"], k["kind"]), ""),
                            sheet_version=ver, as_of=as_of))
        print(f"    {b}: {len(a) - 1}行(表 {ver}、sha256 {sha[:16]})→ " + " / ".join(f"{s}{'' if src == 'author' else '(' + src + ')'} {n}"
                                                                         for (src, s), n in c.most_common()))
    both = [x for x in out if x["axis2_round2"]]
    if both:
        print(f"    軸2(第2回の層名の答え)も在る行 {len(both)}: " + " / ".join(
            f"{a2[:2]} ∧ {s} {n}" for (a2, s), n in collections.Counter((x['axis2_round2'], x['stage']) for x in both).most_common()))
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"    → `{OUT}`(非公開、{len(out)}行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
