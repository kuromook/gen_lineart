#!/usr/bin/env python
"""**二重の頁の聞き取りの答えを取り込み、採る頁の名簿を作る** —— `hearing_duplicate_folders_v5.py` の答え。

**答えを見る前に決めたこと**(表の説明と 11-12 の返事):
- 「別の作品・別の巻」→ 二重ではない。組の頁は全部採る
- 「版違い」→ 組ごとに「採るフォルダー」の頁を採り、ほかのフォルダーの頁を落とす。採るフォルダーが組に無ければ止める
- 「分からない」→ census は採る側を決めない。**但し作者がコメントで使い方を書いたときは、コメントを逐語で残し、
  その指示を名簿に別の理由の札で載せる**(答えの札と混ぜない。下の `basis` 列)
- **名簿が覆うのは二重の組に入った頁だけ。** 同じフォルダーにある、二重でない頁には触れない(聞いていない)。その数を印字する
- 落とすのは頁の単位。対 v5 の行(1頁に対が複数あることがある)の数も印字する
- **v2(ユーザーの決定、2026-11-12 逐語「フォルダごとはずしていい」)**: 落とす頁のあるフォルダーは、二重の組に入っていない頁も
  まとめて落とす。名簿に `basis` = ユーザーの決定 の札で足す。**v1(組の頁だけ)は消さずに残し、前後の数を両方印字する**(規律10)

    ./venv/bin/python tools/census/duplicate_folders_v5_answers.py
"""
import collections
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from duplicate_pages_v5 import OUT as DUP_CSV, QC  # noqa: E402
from hearing_duplicate_folders_v5 import KEY_OUT, XLSX  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANSWERED = ("data/census_full/hearing_duplicate_folders_v5_answered_20261112.xlsx",
            "b9a40cc6052bfb4bc798aed50b79f8e959606d72d3939e4ada8bed07154fb6ff", "2026-11-12")
OUT_V1 = "data/census_full/duplicate_pages_v5_roster_20261112.csv"
OUT = "data/census_full/duplicate_pages_v5_roster_v2_20261112.csv"
FOLDER_RULE = "ユーザーの決定(2026-11-12「フォルダごとはずしていい」)・落とす頁のあるフォルダーの、二重の組に入っていない頁"
PFX = "F:\\manuscripts\\"
NOT_USE = "どれも途中ファイル つかわない"


def main() -> int:
    from openpyxl import load_workbook
    print("[二重の頁の答え → 採る頁の名簿]")
    path, sha, as_of = ANSWERED
    got = hashlib.sha256(open(os.path.join(ROOT, path), "rb").read()).hexdigest()
    if got != sha:
        print(f"    **sha256 が違う** {got[:16]}")
        return 1
    a = list(load_workbook(os.path.join(ROOT, path))["組"].iter_rows(values_only=True))
    o = list(load_workbook(os.path.join(ROOT, XLSX))["組"].iter_rows(values_only=True))
    if len(a) != len(o) or any(x[:8] != y[:8] for x, y in zip(a, o)):
        print("    **固定の列が作った表と違う** —— 止める")
        return 1
    print(f"    答えの表 sha256 {got[:16]}({as_of})/ 固定の列は {len(o) - 1} 行とも作った表と一致")
    key = list(csv.DictReader(open(os.path.join(ROOT, KEY_OUT), encoding="utf-8")))
    folder_row = {}
    ans = {}
    for k, x in zip(key, a[1:]):
        for f in k["folders"].split(" | "):
            folder_row[f] = int(k["row"])
        ans[int(k["row"])] = dict(answer=x[8] or "未記入", take=(PFX + x[9]) if x[9] else "", comment=x[10] or "")
    dup = list(csv.DictReader(open(os.path.join(ROOT, DUP_CSV), encoding="utf-8")))
    groups = collections.defaultdict(list)
    for r in dup:
        groups[r["group"]].append(r)
    out = []
    cnt = collections.Counter()
    for g, rs in groups.items():
        row = folder_row[rs[0]["folder"]]
        A = ans[row]
        if A["answer"].startswith("別の作品"):
            dec = {r["page_path"]: ("採る", "答え: 別の作品・別の巻") for r in rs}
        elif A["answer"].startswith("版違い"):
            if A["take"] not in {r["folder"] for r in rs}:
                print(f"    **組 {g}(行 {row})に採るフォルダーが無い** —— 止める")
                return 1
            dec = {r["page_path"]: (("採る", "答え: 版違い・採るフォルダー") if r["folder"] == A["take"] else ("落とす", "答え: 版違い・採らないフォルダー"))
                   for r in rs}
        elif A["comment"] == NOT_USE:
            dec = {r["page_path"]: ("落とす", "答えは分からない・コメント「どれも途中ファイル つかわない」") for r in rs}
        else:
            dec = {r["page_path"]: ("未決", "答え: " + A["answer"]) for r in rs}
        for r in rs:
            d, basis = dec[r["page_path"]]
            cnt[(row, d)] += 1
            out.append(dict(hearing_row=row, group=g, page_path=r["page_path"], folder=r["folder"], n_pairs_v5=r["n_pairs_v5"],
                            decision=d, basis=basis, answer=A["answer"], take_folder=A["take"], comment=A["comment"]))
    print("    行ごと(頁): ")
    for row in sorted(ans):
        print(f"      {row}: {ans[row]['answer'][:6]} → " + " / ".join(f"{d} {cnt[(row, d)]}" for d in ("採る", "落とす", "未決") if cnt[(row, d)]))
    tot = collections.Counter(o_["decision"] for o_ in out)
    pairs = collections.Counter()
    for o_ in out:
        pairs[o_["decision"]] += int(o_["n_pairs_v5"])
    print(f"    計 頁 {len(out)}: " + " / ".join(f"{d} {tot[d]}(対 v5 の行 {pairs[d]})" for d in ("採る", "落とす", "未決") if tot[d]))
    # 名簿の外: 落とす側と同じフォルダーにある、二重の組に入っていない頁(聞いていないので触れない)
    qc = list(csv.DictReader(open(os.path.join(ROOT, QC), encoding="utf-8")))
    in_roster = {o_["page_path"] for o_ in out}
    drop_folders = {o_["folder"] for o_ in out if o_["decision"] == "落とす"}
    outside = {r["source_path"] for r in qc if r["pair_quality"] == "ok" and r["source_path"].rsplit(chr(92), 1)[0] in drop_folders
               and r["source_path"] not in in_roster}
    print(f"    **名簿の外**: 落とす頁のあるフォルダー {len(drop_folders)} にある、二重の組に入っていない頁 {len(outside)}(QC ok)—— 触れていない")
    by = collections.Counter(p.rsplit(chr(92), 1)[0].replace(PFX, "") for p in outside)
    for f, n in sorted(by.items()):
        print(f"      {f}: {n}")
    write(OUT_V1, out)
    n_qc = {}
    for r in qc:
        if r["pair_quality"] == "ok":
            n_qc[r["source_path"]] = n_qc.get(r["source_path"], 0) + 1
    for p in sorted(outside):
        f = p.rsplit(chr(92), 1)[0]
        row = folder_row.get(f, "")
        out.append(dict(hearing_row=row, group="", page_path=p, folder=f, n_pairs_v5=n_qc[p], decision="落とす", basis=FOLDER_RULE,
                        answer="", take_folder="", comment=""))
    t2 = collections.Counter(o_["decision"] for o_ in out)
    print(f"    **v1(組の頁だけ)**: 採る {tot['採る']} / 落とす {tot['落とす']}(頁 {sum(tot.values())})")
    print(f"    **v2(フォルダーごと)**: 採る {t2['採る']} / 落とす {t2['落とす']}(頁 {len(out)})—— 足した {len(outside)}頁は全部「落とす」")
    write(OUT, out)
    return 0


def write(path, out):
    with open(os.path.join(ROOT, path), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(sorted(out, key=lambda d: (str(d["hearing_row"]), d["page_path"])))
    print(f"    → `{path}`(非公開)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
