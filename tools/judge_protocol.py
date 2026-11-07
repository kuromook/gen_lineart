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
        # **通った側に落ちた側の文面を出さない。** 2026-11-07 に Track O の下書きで露見した ——
        # 「通った 名前: …が無い」と印字していて、読むと矛盾し、引くと嘘になる
        ok.append((label, None)) if cond else bad.append((label, why))

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

    arms_list = spec.get("arms") or []
    blanks = [i for i, x in enumerate(arms_list) if not (isinstance(x, str) and x.strip())]
    need(filled("arms") and not blanks, "腕",
         "腕の一覧が無い" if not filled("arms") else
         f"一覧の {len(blanks)} 本が空のまま(位置 {blanks}) —— **腕の名前が無い = その腕の材料が無い**")

    # **何のための収集かを先に宣言する。** 軸の規則は「腕の差を主張する収集」にだけ効く ——
    # 本人内一貫性を測る収集では脚が分かれているのは当たり前で、分離は落とす理由にならない
    purpose = spec.get("purpose")
    need(purpose in ("compare_arms", "intra_rater", "calibration"), "収集の目的",
         f"`purpose` が compare_arms / intra_rater / calibration のどれでもない({purpose!r})")

    need(filled("normalisation"), "正規化の宣言",
         "**腕の間でファイル形式や経路が違うと、軸が圧縮を読む**(Track N 2026-10-30)。"
         "通す経路を走る前に書く")

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
                # **分離 AUC は引き金で、判定ではない**(2026-11-07 に実測で分かった) ——
                # anchor-20261004 の脚は `near_white_frac` で 0.939 分かれるのに、
                # その軸は作者の分類を再現しない(的中 0.429 / 取りこぼし 0.571、閾値を
                # 60件に合わせた上限値)。**分離は腕の違いを言うだけで、判定の代わりにならない**。
                # この規則をそのまま当てると、**lesson 10 を出した収集自身が止まっていた**。
                rv = ax.get("reproduces_verdicts") or {}
                hit, miss = rv.get("hit_rate"), rv.get("miss_rate")
                if purpose in ("intra_rater", "calibration"):
                    ok.append((f"軸 {name}", f"separation_auc {sep:.3f} ≥ 限界 {limit} だが、"
                                             "この収集は**手法の差を主張しない**"
                                             f"(目的 {purpose})"))
                elif hit is None or miss is None:
                    bad.append((f"軸 {name}", f"separation_auc {sep:.3f} ≥ 限界 {limit} なのに、"
                                              "**その軸が作者の分類を再現するかの読みが無い**(規律13)。"
                                              "器具 `tools/axis_vs_verdicts.py`。"
                                              "読みが無いと、計算で答が出ているのか腕が違うだけなのか区別できない"))
                elif rv.get("reproduces"):
                    bad.append((f"軸 {name}", f"separation_auc {sep:.3f} ≥ 限界 {limit} で、"
                                              f"**作者の分類を再現する**(的中 {hit} / 取りこぼし {miss})"
                                              " —— 計算で出る答に人の時間を使わない(条例1)"))
                else:
                    ok.append((f"軸 {name}", f"separation_auc {sep:.3f} ≥ 限界 {limit} だが、"
                                             f"**作者の分類を再現しない**(的中 {hit} / 取りこぼし {miss}"
                                             f"、記録 {rv.get('records', '?')})。"
                                             "分離は腕の違いで、判定の代わりにはならない(規律13)"))
            else:
                ok.append((f"軸 {name}", f"separation_auc {sep:.3f} < 限界 {limit}"))

    # **規律13**(2026-10-31) —— 近い列で代用しない。軸を作者の判定の**代わり**に使うなら、
    # その軸が作者の分類を再現すると示し、**的中率と取りこぼしを印字する**まで使わない
    proxies = spec.get("axes_used_as_proxy")
    if proxies:
        for px in proxies:
            name = px.get("name")
            hit, miss = px.get("hit_rate"), px.get("miss_rate")
            if hit is None or miss is None:
                bad.append((f"代用の軸 {name}",
                            "**作者の判定の代わりに使うのに、的中率と取りこぼしが無い**(規律13)"))
            else:
                ok.append((f"代用の軸 {name}", f"的中率 {hit} / 取りこぼし {miss} を印字済み(規律13)"))
    else:
        ok.append(("代用の軸", "無し —— 軸は人に聞くかどうかを決めるだけで、判定の代わりには使わない"))

    if not spec.get("common_source"):
        # **偽の宣言で lesson 11 の点検を丸ごと飛ばせてはいけない**(沈黙が合格と見分けがつかない)
        need(filled("common_source_reason"), "同一元でない理由",
             "`common_source: false` と書くなら、**なぜ全腕が同じ元画像の加工でないのか**を書く"
             " —— 書かないと lesson 11 の点検が黙って飛ぶ")
    if spec.get("common_source") and purpose == "calibration":
        ok.append(("損傷量", "**校正では損傷が独立変数**なので、開きの規則(lesson 11)は当てない —— "
                             "代わりに段の単調性を見ている"))
    elif spec.get("common_source"):
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

    # **文脈は後からしか測れない。** 2026-10-04 は時刻しか残っていないので、
    # 「綺麗な絵の直後だったか」を後から読めない(条例4 が警告している汚染そのもの)
    need(spec.get("records_order_and_predecessor") is True, "順番と直前の項目を記録する",
         "`records_order_and_predecessor: true` が無い —— **判定は文脈に依存しうる**ので、"
         "項目の位置と直前の項目を記録しないと、後から文脈の効果を測れない")

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
    if purpose == "calibration":
        # **校正は「判定者がどこで反転するか」を測る。** 損傷は独立変数なので lesson 11 の
        # 開きの規則は当てない —— 代わりに**段が実測で単調であること**と、
        # **どの腕もモデルの出力でないこと**を見る。ここを緩めると、校正の名で手法を集められる
        need(spec.get("claims_about_methods") is False, "手法について主張しないことの明示",
             "目的が calibration なら `claims_about_methods: false` と書く —— "
             "**条例5 が禁じているのは手法の判定を頼み直すこと**で、判定者の校正はそれではない")
        need(filled("operator"), "落とし方の宣言",
             "**閾値は(判定者 × 落とし方)の性質**である。落とし方を名前で書く")
        levels = spec.get("levels") or []
        need(len(levels) >= 3, "段", f"段が {len(levels)} 本しかない(3本未満では反転を挟めない)")
        vals = [lv.get("measured_removed_frac_median") for lv in levels]
        if any(v is None for v in vals):
            bad.append(("段の実測", "**狙いではなく実測**の損傷量を段ごとに入れる(丸ごと消すので狙いを外す)"))
        elif any(b < a for a, b in zip(vals, vals[1:])):
            bad.append(("段の単調性", f"実測が単調でない({vals}) —— 段が入れ替わっていたら独立変数ではない"))
        else:
            ok.append(("段", f"{len(levels)} 本、実測 {vals[0]:.3f} 〜 {vals[-1]:.3f} で単調"))
        allowed = {lv.get("name") for lv in levels} | set(registered_legs)
        strays = [x for x in (spec.get("arms") or []) if x not in allowed]
        need(not strays, "校正に手法の出力を入れない",
             f"段にも錨脚にも無い腕 {strays} が在る —— **校正の名で手法を集めることになる**")

    if purpose == "intra_rater":
        extra = [x for x in (spec.get("arms") or []) if x not in registered_legs]
        need(not extra, "本人内一貫性の収集に新しい腕を入れない",
             f"目的が intra_rater なのに登録簿の脚に無い腕 {extra} が在る —— "
             "軸の規則を迂回して腕を集めることになる")
        need(spec.get("claims_about_arms") is False, "腕について主張しないことの明示",
             "目的が intra_rater なら `claims_about_arms: false` と書く(読む側がそう読めるように)")

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
        print(f"   通った   {label}" + (f": {why}" if why else ""))
    print()
    if bad:
        print("   **集めない。** 落ちた項目を埋めてから回す —— 集め直しが一番高い")
    else:
        print("   通った。集めてよい。**集めたら config/judge_anchor_registry.json に錨として登録する**")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
