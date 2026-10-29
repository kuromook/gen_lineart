#!/usr/bin/env python3
"""**人の判定を集める前に回す門。** 条例2 の器具。

なぜ門にするか: **作者の目が最も高い資源で、集め直しが一番高い。** この track は
2026-10-04 に77件集めてから材料が答えられない形だと分かり、同日に203件を読んで
計器チェックを撤回している。どちらも**集める前に測れた**(lesson 11 自身がそう言っている
—— 「腕を分ける軸を測って、それが選好を予測するか見る。予測するなら人の判定は
既に計算できたものを足さない」)。

見るもの(落ちた項目は**理由別に数えて印字する**。規律9: 選別は測定である):

  - **画面に「何について答えるのか」の文面が在るか**(2026-10-26)。合成画像なら
    合成前の各層と、問う対象の印
  - **計算可能な軸を先に測ったか**、そして**その軸で腕が分かれていないか**
    (分かれていれば人に聞く前に答が出ている。限界値は登録簿に先に書く)
  - **全腕が同じ元画像の劣化なら、損傷量を腕ごとに測ったか**(lesson 11)
  - **再掲が在るか**(本人内一貫性は採点器の上限。無い収集は上限の無い収集)
  - 「どちらとも言えない」が選べるか(選べない形なら理由)
  - 層化とタイル共有の申告(共有するなら偏りの向き)
  - **錨脚の同梱と同一セッション**(条例1)
  - 棒・停止則・**読む順**が先に書いてあるか
  - **登録日が収集開始より前か**(規律9-5)

使い方:
  judge_protocol.py <登録 JSON>            # 門。型そのものを渡すと落ちる(埋めていないから)
  judge_protocol.py --template             # 型の道を印字

終了コード: 0 = 通る / 1 = 落ちた項目が在る / 2 = 器具が走れない
**何も書き換えない。集めるかどうかを決めるのは人。**
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "config/judge_protocol_template.json"
ANCHORS = ROOT / "config/judge_anchor_registry.json"


def check(spec):
    """(落ちた項目, 通った項目) を返す。**理由別に数えるので、最初の失敗で止めない。**"""
    bad, ok = [], []

    def need(cond, label, why):
        (ok if cond else bad).append((label, why))

    def filled(key):
        v = spec.get(key)
        return v not in (None, "", [], {}, 0)

    need(filled("name"), "名前", "この収集の名前(合格したら錨の名前になる)が無い")
    d_reg, d_col = spec.get("registered_on"), spec.get("collection_starts_on")
    need(bool(d_reg) and bool(d_col) and str(d_reg) <= str(d_col),
         "登録が収集より前", f"登録 {d_reg} / 収集開始 {d_col} —— 後から書いた登録は登録ではない(規律9-5)")

    need(filled("question"), "問いの文面", "判定者に見せる文そのものが無い(要約では画面に出せない)")
    need(filled("unit_on_screen"), "画面の単位の文面",
         "**何について答えるのかを画面に書いた文面**が無い(2026-10-26)")
    if spec.get("composite_stimulus"):
        need(filled("layers_shown"), "合成前の層", "合成画像なのに合成前の各層を並べていない(2026-10-26)")
        need(filled("marked_target"), "問う対象の印", "合成画像なのに問う対象に印が無い(2026-10-26)")

    need(filled("judge"), "判定者", "判定者が構成の一部(規律1)")
    nj = spec.get("n_judges")
    if nj and nj > 1:
        ok.append(("判定者2人以上", "**取り直しではなく別の錨**として登録すること(規律1)"))

    need(filled("arms"), "腕", "腕の一覧が無い")

    axes = spec.get("computable_axes")
    if not axes:
        bad.append(("計算可能な軸", "**未測定**。人に頼む前に、選好を予測する軸を測る(2026-10-04)"))
    else:
        limit = spec.get("axis_separation_limit")
        if limit is None:
            bad.append(("軸の限界値", "`axis_separation_limit` を先に登録していない"))
        for ax in axes:
            name, sep = ax.get("name"), ax.get("separation_auc")
            if sep is None:
                bad.append((f"軸 {name}", "測っていない(null)。測るまで集めない"))
            elif limit is not None and sep >= limit:
                bad.append((f"軸 {name}", f"separation_auc {sep:.3f} ≥ 限界 {limit} —— "
                                          "**この軸で腕が分かれている。計算で答が出るので人に聞かない**"))
            else:
                ok.append((f"軸 {name}", f"separation_auc {sep:.3f} < 限界 {limit}"))

    if spec.get("common_source"):
        dm = spec.get("damage_measure") or {}
        if not dm:
            bad.append(("損傷量", "全腕が同じ元画像の劣化なのに腕ごとの損傷量が無い(lesson 11)"))
        else:
            vals = [v for v in dm.values() if isinstance(v, (int, float))]
            shown = ", ".join(f"{k} {v}" for k, v in dm.items())
            if len(vals) < 2:
                bad.append(("損傷量", f"腕ごとに揃っていない({shown})"))
            elif min(vals) <= 0 < max(vals):
                # **損傷0の腕が在れば比は無限大。** 比で見て「開きは小さい」と印字するのは嘘になる
                bad.append(("損傷量の分離", f"**損傷0の腕と損傷>0の腕が混在**({shown}) —— "
                                            "この軸で腕が完全に分かれる。**判定の差を手法の差として読めない**(lesson 11)"))
            elif max(vals) == 0:
                ok.append(("損傷量", f"全腕が損傷0({shown}) —— 劣化の族ではない。lesson 11 は効かない"))
            elif max(vals) / min(vals) >= 1.5:
                bad.append(("損傷量の分離", f"腕ごとの損傷量が {min(vals):.4g}〜{max(vals):.4g} と "
                                            f"1.5倍以上開いている({shown}) —— "
                                            "**判定の差を手法の差として読めない**(lesson 11)"))
            else:
                ok.append(("損傷量", f"腕ごとに測ってあり、開きは 1.5倍未満({shown})"))

    need(filled("n_items"), "件数", "件数が無い")
    need(filled("repeats"), "再掲",
         "**再掲が無い。** 本人内一貫性は採点器の上限で、測らなければ上限が無いまま走る")
    if not spec.get("undecidable_option"):
        need(filled("forced_choice_reason"), "選べない形の理由",
             "「どちらとも言えない」が無いのに理由が書いていない")

    need(filled("stratification"), "層化", "層化の書き方が無い(手がかりで選べる組を作らないため)")
    if spec.get("tile_overlap_across_arms") is None:
        bad.append(("タイル共有の申告", "腕の間でタイルを共有するかが書いていない"))
    elif spec.get("tile_overlap_across_arms"):
        need(filled("overlap_bias_note"), "共有の偏りの向き", "共有するなら偏りの向きを書く")
    else:
        ok.append(("タイル共有の申告", "共有しない"))

    legs = spec.get("anchor_legs")
    registered_legs = []
    if ANCHORS.exists():
        reg = json.loads(ANCHORS.read_text())
        for a in reg.get("anchors", []):
            if a.get("is_bar"):
                registered_legs = (a.get("retake") or {}).get("legs") or []
    if not legs:
        bad.append(("錨脚", f"錨脚を同梱していない(登録簿の脚: {registered_legs or '登録なし'})。条例1"))
    else:
        missing = [l for l in registered_legs if l not in legs]
        need(not missing, "錨脚", f"登録簿の脚 {missing} が入っていない(条例1)")
    need(spec.get("same_session") is True, "同一セッション",
         "錨脚と新しい腕を同一セッションで取らないと、基準の移動が腕の差と交絡する(条例1)")

    need(filled("bar"), "棒", "合格条件が先に書いていない")
    need(filled("stopping_rule"), "停止則", "停止則が先に書いていない")
    need(filled("read_order"), "読む順", "読む順が先に書いていない(2026-10-04 はこれが結論を決めた)")
    return bad, ok


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    if argv[0] == "--template":
        print(TEMPLATE)
        return 0
    path = Path(argv[0])
    if not path.exists():
        print(f"登録が無い: {path}", file=sys.stderr)
        return 2
    spec = json.loads(path.read_text())
    bad, ok = check(spec)
    print(f"══ 判定収集の門: {path.name}  (schema {spec.get('schema')})")
    print(f"   通った {len(ok)} 項目 / 落ちた {len(bad)} 項目")
    print()
    for label, why in bad:
        print(f"   **落ちた** {label}: {why}")
    if bad:
        print()
    for label, why in ok:
        print(f"   通った   {label}: {why}")
    print()
    if bad:
        print("   **集めない。** 落ちた項目を埋めてから回す —— 集め直しが一番高い")
    else:
        print("   通った。集めてよい。**集めたら config/judge_anchor_registry.json に錨として登録する**")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
