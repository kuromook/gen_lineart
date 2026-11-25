#!/usr/bin/env python
"""**Q29 の作者の20行の答えを層ごとに記録する**(Track Q の頼み `reply_census_q29_result_20261125.md`、2026-11-25。新しい走行ではない)。

- 答え = Q の聞き取りの記録 `data/frames/q29_hearing/q29_effect_lines_20261125_answers.jsonl`(項目ごとに最後の答え)
- 項目 ↔ 頁・層 = 同じフォルダーの `arm_log.json`(`id` = 項目、`src` = 画の便の q29_NN)
- 作者の層の種類 `author_kind` = 「効果線」を選んだ行はその語、「どれでもない」を選んだ行は**作者の一言をそのまま**(言い換えない。規律6)
- 腕の値(AN・AF3)は Q の列なので写すだけ。census は読まない。落とした行は理由別に印字する(規律9)

出力 `data/census_full/q29/q29_author_layer_answers_20261125.csv`(非公開)。
"""
import collections
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.path.join(ROOT, "..", "lineart-layer-exclusion", "data", "frames", "q29_hearing")
OUT = os.path.join(ROOT, "data", "census_full", "q29", "q29_author_layer_answers_20261125.csv")


def main() -> int:
    log = {r["id"]: r for r in json.load(open(os.path.join(H, "arm_log.json"), encoding="utf-8"))}
    ans, n_lines = {}, 0
    for line in open(os.path.join(H, "q29_effect_lines_20261125_answers.jsonl"), encoding="utf-8"):
        if line.strip():
            n_lines += 1
            a = json.loads(line); ans[a["item_id"]] = a
    print(f"答えの行 {n_lines}(項目 {len(ans)}、選び直しで上書き {n_lines - len(ans)})/ 項目の表 {len(log)}")
    drop = collections.Counter()
    out = []
    for i in sorted(log):
        r = log[i]
        if i not in ans:
            drop["答えが無い"] += 1; continue
        a = ans[i]
        kind = a["label"] if a["value"] != "__none__" else (a["comment"].strip() or "(一言なし)")
        out.append(dict(item_id=i, q29_src=r["src"], page_path=r["page_path"], layer_id=r["layer_id"],
                        hidden=int(bool(r["hidden"])), q_set=r["set"], AN=r["AN"], AF3=r["AF3"], added=r["added"],
                        answer_value=a["value"], answer_label=a["label"], author_comment=a["comment"],
                        author_kind=kind, received_at=a["received_at"]))
    drop.update("項目の表に無い答え" for k in ans if k not in log)
    print("落とした: " + (" / ".join(f"{k} {v}" for k, v in drop.items()) or "0"))
    print("作者の層の種類: " + " / ".join(f"{k} {v}" for k, v in collections.Counter(o["author_kind"] for o in out).most_common()))
    print(f"  非表示の層 {sum(o['hidden'] for o in out)} / 足す層(R5){sum(o['added'] == '1' for o in out)} / 頁 {len({o['page_path'] for o in out})}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"出力: {OUT}({len(out)}行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
