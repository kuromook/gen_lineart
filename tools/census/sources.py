"""集計の材料と、その出所。

**供給元から取った数は、ここで誰かが測るまで、測った者の名を帯びる**(T132)。
だからこの表は「どの数を誰が測ったか」を列に持つ。`measured_by` を空にしてはいけない。
"""

# 供給元が報告した数。**ここでは測っていない。** 再測するには DB 側の材料が要る
# (列 `needs` にそれを書く。足りないものは基盤経由で抽出側へ依頼する)
SUPPLIER_REPORTED = [
    dict(
        key="db_files",
        predicate="構造 DB の `pages` 行 = 原稿ファイル(.clip 7,667 + .lip 1,986)",
        n_pages=9653, n_layers=None,
        measured_by="抽出側(um350)", as_of="2026-10-26",
        needs="",   # **満たされた**(便 n1 が年 × 形式の頁数を持ってきた)。
        # **D ドライブの棚卸しの発注は `other_drive_files` の行に在る** ——
        # **その母集団についての依頼なので、そちらが `needs` の正本である**(条例8)
        satisfied_when=("source", "pages_by_year_all"),
        caveat="ファイル数。**異なり数とは別の量**(T96 の形)。"
               "**年 × 形式の行は `population_by_format.csv`** —— "
               "**`.lip` 1,986頁もこの 9,653 に入っている**"
               "(作者の決定 2026-10-27)。"
               "**【2026-10-30 作者】母集団は F ドライブだけである** —— "
               "**もう1台のドライブの札 2025 / 2025b に原稿が在るが、"
               "構造 DB にも 9,653 にも入っていない**(行は `other_drive_files`)。"
               "**C ドライブの札 2026 は作者が対象外と決めた**"
               "(「まだ製作中のものが多い」)。"
               "**「数を知らない」と「入れないと決めた」を同じ扱いにしない**",
    ),
    dict(
        key="db_distinct",
        predicate="sha256 で重複を畳んだ異なり数(.clip 6,194 + .lip 1,224)",
        n_pages=7418, n_layers=None,
        measured_by="census track(2026-10-26 に全 9,653 行から数え直した)",
        as_of="2026-10-26",
        needs="",   # **満たされた。** `request9` の sha256 で測れた
        satisfied_when=("source", "file_sha256"),
        caveat="**ファイル 9,653 と異なり 7,418 は同じ母集団の別の量。**並べて書く。"
               "**ハッシュは抽出側が計算した列**なので、先方の読みが違えば同じく違う",
    ),
    dict(
        key="clip_files",
        predicate="`.clip` ファイルのみ 7,667(異なり 6,194)。"
                  "**重複の3つの量はこの母集団の数**",
        n_pages=7667, n_layers=None,
        measured_by="抽出側(um350)/ 重複は census track が数え直した",
        as_of="2026-10-26",
        needs="",   # **満たされた**(2026-10-31、6札の納品。読みは事前登録どおり)
        satisfied_when=("source_prefix", "ink_darkness_stage2"),
        caveat="**7,667 − 6,194 を「複製の数」と名乗らない** —— "
               "余り 1,473 / 双子を持つファイル 2,458 / 塊 985 は**3つの別の量**",
    ),
    dict(
        key="db_layers",
        predicate="構造 DB の `layers` 行のうち **`.clip` の分**。"
                  "**437,561 と同じ走行の数**(便 n2、2026-10-27)",
        n_pages=None, n_layers=312898,
        measured_by="抽出側(um350)/ census track が依頼8 の表から数え直した",
        as_of="2026-10-27",
        needs="",   # **満たされた。** 便 n2 が同じ走行の形式ごとの数を持ってきた
        satisfied_when=("source", "clip_only_layers"),
        caveat="**数は前の走行と同じ 312,898 だが、同じ数であることは同じ走行であることでは"
               "ない** —— 便 n2 で**同じ走行の数として取り直した**(抽出側の言明: "
               "この DB を作り直したとき `.clip` の層の数は変わっていない)。"
               "**これで `437,561 − 312,898` を使わずに `.lip` の 124,663 が言える** —— "
               "`A − B` は量ではなく手続きなので、**同じ走行の形式ごとの数として来るまで"
               "待った。** 形式ごとの行は `population_by_format.csv`",
    ),
]

# ここで読める CSV。行は自分で数える(ただし **供給元の配列から数えているので、
# 供給元の集計と一致しても独立な確認ではない** —— 転記の一致である。T132)
LOCAL_CSV = [
    dict(
        key="req3_layer_kind",
        path="inbox/f_request3_layer_kind_rendered_20261026.csv",
        predicate="依頼3の872頁の、**フォルダー以外の全層**(種別×描画×drawn_ratio)",
        supplier="抽出側(um350)", as_of="2026-10-26",
        caveat="**フォルダー層は入っていない。** 層数を added33 の `n_layers` と比べない",
    ),
    dict(
        key="body_initcolor",
        path="inbox/body_initcolor_delivered_20261026.csv",
        predicate="トーン・グラデ・ベタの**本体の InitColor**(1層1行、2,095層)",
        supplier="抽出側(um350)", as_of="2026-10-26",
        satisfies="body_initcolor",
        caveat="**1層1行なのでここで測れる。** `set` 列に作品名が入るので"
               "**段 A では種類に畳む**(畳んだ世代の数を印字する)",
    ),
    dict(
        key="file_sha256",
        path="inbox/request9_file_sha256_all_20261026.csv",
        predicate="**全 9,653 ファイルの `page_path + sha256`**(1ファイル1行)",
        supplier="抽出側(um350)", as_of="2026-10-26",
        caveat="**これで異なり数と重複の3つの量がここで測れる。** ただし"
               "**ハッシュそのものは抽出側が計算した** —— 先方の読みが違えば同じく違う",
    ),
    dict(
        key="req3_body_blocks",
        path="inbox/faint_alpha_905_20261027.csv",
        predicate="905頁の `body_init` / `body_blocks` / `body_stored` と"
                  "**アルファ≥1 の比**(62,105層。依頼3の 56,576層を全部含む)",
        supplier="抽出側(um350)", as_of="2026-10-27",
        caveat="**依頼3の3列がこれで埋まった** —— 下限の層が何層かを計数できる。"
               "**併せて `drawn_ratio` の閾 26 が落としている薄い画素が見える**",
    ),
    dict(
        key="grad_v1_v2_map",
        path="inbox/grad_v1_v2_layer_map_20261026.csv",
        predicate="グラデ v1/v2 の**層ごとの対応**(258層)",
        supplier="抽出側(um350)", as_of="2026-10-26",
        caveat="**層を鍵にした突き合わせはこれが無いとできない** —— "
               "PNG の名前は頁の中の順番を encode している",
    ),
    dict(
        key="layer_names_all",
        path="inbox/census_r7_layer_names_20261027.csv",
        predicate="**全数の層名**(16,241 の異なる名前 / 437,561層)。"
                  "`is_folder`・年の範囲・種類の内訳・親の名前の異なり数つき",
        supplier="抽出側(um350)", as_of="2026-10-27",
        caveat="**これで名前側が全数に届いた。** だが"
               "**素性側(画素の分布)は8頁しか届かない** —— "
               "**両面が一致したと言える範囲は8頁で、名前側の数は全数である。**"
               "**同じ行に並べない。** そして**フォルダーの行を含む**"
               "(`黒消し` は層では0件、フォルダーで 609)",
    ),
    dict(
        key="layer_structure_all",
        path="inbox/census_r8_layer_structure_20261027.csv",
        predicate="**全数の層の構造**(437,561層の `parent_id`/`order_in_parent`/"
                  "`ancestors`/`composite`/`clip`/`visibility`/`opacity`/`z_global`)",
        supplier="抽出側(um350)", as_of="2026-10-27",
        caveat="**`z_global` の向きは供給元が v2 で訂正した**(v1 は逆。CSV は正しかった)"
               " —— **こちらでは兄弟の `order_in_parent` との一致でしか確かめていない。**"
               "**画素は入っていない**ので、**何を隠しているかは測れない**"
               "(手前に在ることは重なっていることではない)",
    ),
    dict(
        key="pages_by_year_all",
        path="inbox/census_n1_pages_by_year_20261027.csv",
        predicate="**全数の年 × 形式の頁数**(9,653頁 = `.clip` 7,667 + `.lip` 1,986。22行)",
        supplier="抽出側(um350)", as_of="2026-10-27",
        caveat="**頁の数はこの便だけが持っている** —— 依頼8 の表は**層の在る頁**しか"
               "持たないので、**読めなかった 4頁(`.clip`)がそこには出てこない。**"
               "`n_pages` と `n_pages_with_layers` は**別の列**にしてある(T138)",
    ),
    dict(
        key="clip_only_layers",
        path="inbox/census_n2_layers_by_format_20261027.csv",
        predicate="**同じ走行の形式ごとの層数**(`.clip` 312,898 / `.lip` 124,663。"
                  "どちらもフォルダーを含む)",
        supplier="抽出側(um350)", as_of="2026-10-27",
        caveat="**行はここから出していない** —— 年の次元が無いので、"
               "`population_by_format.csv` の行は依頼8 の表から数えて、"
               "**この便とは突き合わせだけをする。** "
               "**同じ DB の同じ走行なので、一致は一致であって独立な確認ではない**(T132)",
    ),
    dict(
        key="lip_layers",
        path="inbox/census_n4_lip_layers_20261027.csv",
        predicate="**`.lip` の層の表**(124,663層 / 1,986頁。"
                  "`v_layer_kind`・`layer_type`・`rendered`)",
        supplier="抽出側(um350)", as_of="2026-10-27",
        caveat="**`rendered` は全行が空** —— 抽出側の言明は"
               "**「描けなかったのではなく、描いていない」。**"
               "**走行のたびに全行を数えて確かめている**(言明と中身の突き合わせ)。"
               "**この便の層の属性は依頼8 の表にも在る**ので、"
               "**行はこの便から出さず、突き合わせに使う**(T132)",
    ),
    dict(
        key="added33_structure",
        path="inbox/pages_added33_structure_20261026.csv",
        predicate="後から入った33頁の構造(層数・フォルダー数・画布・dpi)",
        supplier="抽出側(um350)", as_of="2026-10-26",
        caveat="**`canvas_unit` 列が無い。** 行ごとに mm か px かを決められない(T126)"
               " —— **単位は便 n5(`canvas_px`)で決まった**(2026-10-28)",
    ),
    dict(
        key="mask_cover_request3",
        path="inbox/mask_cover_request3_20261026.csv",
        predicate="依頼3のマスクの被覆(**16,273行・766頁。フォルダーの行を 1,839 含む**)",
        supplier="抽出側(um350)", as_of="2026-10-26",
        caveat="**原点を 0 に置いて読んでいた** —— **ずれていた 383行(55頁)は "
               "2026-10-30 の便で差し替えた**(鍵で上書き。この便は書き替えていない)。"
               "**2026-10-29 まで「便にその列が無いので、ずれている層の数はこの材料では"
               "測れない」と書いていたのは誤り** —— 足りなかったのは列ではなく読み方で、"
               "**器具が毎走行で数を印字する**。"
               "**依頼3の層の便はフォルダーを落としているが、この便は含む** —— "
               "**層数を2つの便の間で引き算しない**(T138)",
    ),
    dict(
        key="mask_cover_maskorigin",
        path="inbox/mask_cover_request3_maskorigin55_20261030.csv",
        predicate="**原点を直して測り直したマスクの被覆**"
                  "(1,501行・55頁。**合計が 0 でない層を持つ頁の全部**)",
        supplier="抽出側(um350。器具 `1d3909a`)", as_of="2026-10-30",
        caveat="**古い便を書き替えず、鍵(`page_path`,`layer_id`)で差し替える** —— "
               "**値が動いたのは 383行で、「合計が 0 でない層」と一致した**"
               "(鍵が増えた行 0。**一致しなければ直しが別のことをしている印**)。"
               "**この便は 55頁しか無い** —— **766頁の表と行数を比べない**(T138)",
    ),
    dict(
        key="canvas_px",
        path="inbox/census_n5_added33_canvas_20261027.csv",
        predicate="33頁の `canvas_unit` と**画素の画布**(`v_canvas_px_w/h`)",
        supplier="抽出側(um350)", as_of="2026-10-27",
        caveat="**宣言(`canvas_unit`)と量(画素)は別の側である** —— "
               "**両方で確かめた**(`mm / 25.4 * dpi` が実測の画素と一致するか)。"
               "**2頁は `canvas_unit` が空**で、px と読んだのは**読みであって宣言ではない**",
    ),
    dict(
        key="ink_darkness_stage1",
        path="inbox/ink_darkness_stage1_20261030.csv",
        predicate="**墨の濃さの1段目**(1行1層、4,063層)。"
                  "**縮小も白点正規化もしない層の実体**",
        supplier="抽出側(um350)", as_of="2026-10-30",
        caveat='**墨の濃さの1段目**(6フォルダー / L1 3,773層・C1 288・C2 2)。**sha256 検証済み。** **1行1層、縮小も白点正規化もしない層の実体。** 畳んだ表は `ink_darkness_stage1.csv`。**作品名とフォルダーの対応は未確定**(そもそも1対1に決まらなかったので候補を全部測ってある。5作品のうち1つは構造 DB の外に在り未測定(名前は便にのみ書く))。**既知セルは2つとも素直に立っていない** —— C1 は W1 の用紙 91層が全面純黒(抽出側が「読み違いの可能性。不明」と明記)、C2 は該当2層が純黒でない。**主結論は画素形式2の 3,284層だけで引き直しても動かない**(純黒 0.8637 対 0.8503)',
    ),
    dict(
        key="work_arcs_readback",
        path="inbox/work_arcs_20261030.csv",
        predicate="**編の表を DB に入れた後の書き戻し**(42組)。"
                  "**こちらが送った配分と突き合わせるための材料**",
        supplier="抽出側(um350)", as_of="2026-10-30",
        caveat="**これは新しい測定ではなく、送ったものの書き戻しである** —— "
               "**一致しても「DB が正しい」ではなく「転記が合っている」**(T132)。"
               "**こちらで引き直したのは鍵の集合・編の値・`source_sha256`**"
               "(`tools/census/arcs.py`。食い違い 0)。"
               "**先方の照合**(42組が全部 `pages` に当たる / 頁数も一致 / "
               "ビュー `v_page_arc` の行 2,359)**は引き直していない** —— "
               "**作業木の外の断面(2026-10-26)に `work_arcs` は無い**(条例5)",
    ),
    dict(
        key="ink_darkness_stage2_2015",
        path="inbox/ink_darkness_stage2_2015_20261031.csv",
        predicate="**墨の濃さの2段目、札 2015**(1行1層、111層)",
        supplier="抽出側(um350。器具 `38f2859`。**1段目と1文字も変えていない**)",
        as_of="2026-10-31",
        caveat="**6札に分かれて届いた1つの納品の一部** —— "
               "**ファイルごとに行数を比べない。札ごとに作品の数が違う**(T138)。"
               "**sha256 検証済み。** 畳んだ表は `ink_darkness_stage2.csv`",
    ),
    dict(
        key="ink_darkness_stage2_2016",
        path="inbox/ink_darkness_stage2_2016_20261031.csv",
        predicate="**墨の濃さの2段目、札 2016**(1行1層、11,318層)",
        supplier="抽出側(um350。器具 `38f2859`。**1段目と1文字も変えていない**)",
        as_of="2026-10-31",
        caveat="**6札に分かれて届いた1つの納品の一部** —— "
               "**ファイルごとに行数を比べない。札ごとに作品の数が違う**(T138)。"
               "**sha256 検証済み。** 畳んだ表は `ink_darkness_stage2.csv`",
    ),
    dict(
        key="ink_darkness_stage2_2017",
        path="inbox/ink_darkness_stage2_2017_20261031.csv",
        predicate="**墨の濃さの2段目、札 2017**(1行1層、19,344層)",
        supplier="抽出側(um350。器具 `38f2859`。**1段目と1文字も変えていない**)",
        as_of="2026-10-31",
        caveat="**6札に分かれて届いた1つの納品の一部** —— "
               "**ファイルごとに行数を比べない。札ごとに作品の数が違う**(T138)。"
               "**sha256 検証済み。** 畳んだ表は `ink_darkness_stage2.csv`",
    ),
    dict(
        key="ink_darkness_stage2_2018",
        path="inbox/ink_darkness_stage2_2018_20261031.csv",
        predicate="**墨の濃さの2段目、札 2018**(1行1層、9,098層)",
        supplier="抽出側(um350。器具 `38f2859`。**1段目と1文字も変えていない**)",
        as_of="2026-10-31",
        caveat="**6札に分かれて届いた1つの納品の一部** —— "
               "**ファイルごとに行数を比べない。札ごとに作品の数が違う**(T138)。"
               "**sha256 検証済み。** 畳んだ表は `ink_darkness_stage2.csv`",
    ),
    dict(
        key="ink_darkness_stage2_2019",
        path="inbox/ink_darkness_stage2_2019_20261031.csv",
        predicate="**墨の濃さの2段目、札 2019**(1行1層、6,124層)",
        supplier="抽出側(um350。器具 `38f2859`。**1段目と1文字も変えていない**)",
        as_of="2026-10-31",
        caveat="**6札に分かれて届いた1つの納品の一部** —— "
               "**ファイルごとに行数を比べない。札ごとに作品の数が違う**(T138)。"
               "**sha256 検証済み。** 畳んだ表は `ink_darkness_stage2.csv`",
    ),
    dict(
        key="ink_darkness_stage2_2019b",
        path="inbox/ink_darkness_stage2_2019b_20261031.csv",
        predicate="**墨の濃さの2段目、札 2019b**(1行1層、2,470層)",
        supplier="抽出側(um350。器具 `38f2859`。**1段目と1文字も変えていない**)",
        as_of="2026-10-31",
        caveat="**6札に分かれて届いた1つの納品の一部** —— "
               "**ファイルごとに行数を比べない。札ごとに作品の数が違う**(T138)。"
               "**sha256 検証済み。** 畳んだ表は `ink_darkness_stage2.csv`",
    ),
    dict(
        key="other_drive_inventory",
        path="inbox/d_drive_file_sha256_20261031.csv",
        predicate="**もう1台のドライブの 1,185 ファイルの `sha256`**"
                  "(`.clip` 1,085 / `.lip` 100。読み取り誤り 0)",
        supplier="抽出側(um350。器具は依頼9 と同じ `hash_clip_files.py`)",
        as_of="2026-10-31",
        caveat="**ファイル全体のバイト列の hash だけ。** "
               "**`.clip` も `.lip` も、SQLite としても画像としても開いていない**"
               "(先方の言明)。**だから層も頁も構造も、この材料には無い。** "
               "**突き合わせはこちらでやった**(`tools/census/other_drive.py`)—— "
               "**母集団に双子を持つファイル 385 / 異なり 245。** "
               "**同じ絵かどうかには届かない**(描き直しは別の hash)",
    ),
]

# **届いているが、まだこの集計に入っていない便。**
#
# **これを書く理由**: 「数が無い」には2つ在る —— **材料が無い**のと
# **材料は在るが誰も入れていない**。`needs` の欄は前者を指すので、
# **後者をそこに混ぜると「頼んである」と読まれる。** 別の列に出す。
ARRIVED_NOT_INTEGRATED = [
    dict(
        path="inbox/work_exports_works_20261030.csv",
        what="**作品フォルダーの中の画像ファイルの棚卸し**(290作品 / 画像 11,549 / "
             "**画像が 0 件の作品 89(頁 2,866)**)。**Track M 宛の便の写し**",
        why_not_yet="**画像は母集団の頁ではない**し、"
                    "**「書き出しの画像かどうかは判断していない」と先方が書いている**"
                    "(資料・素材・スキャンも同じ拡張子で混ざる)。"
                    "**census が受け取ったのは、この便が示した1件の食い違いだけ** —— "
                    "**年フォルダの直下に置かれた `.clip` が4つ在り、"
                    "作品フォルダーが無いのでこの棚卸しに入っていない**"
                    "(**構造 DB には頁として在る。だから 7,663 と 11,549 の母集団は"
                    "同じではない**)",
    ),
    dict(
        path="inbox/occ_maskorigin_189_layers_20261030.csv",
        what="**原点を直して測り直した遮蔽の計測**(189頁 / 25,100層)",
        why_not_yet="**基盤の走行の材料で、この集計は遮蔽の被覆を1列も持っていない** —— "
                    "**写しとして置いてある**(`cov26_wide` が動いた層は先方の報告で "
                    "1,451層・124頁。**こちらでは引き直していない**)",
    ),
]


def audit_arrived() -> int:
    """**宣言した便が実在するか。** 無い便を「届いている」と書いたら指摘。"""
    import os
    bad = []
    print("  [届いているが未統合の便]")
    for a in ARRIVED_NOT_INTEGRATED:
        here = os.path.exists(a["path"])
        if not here:
            bad.append(a["path"])
        print(f"    {os.path.basename(a['path']):46s} "
              f"{'在る' if here else '**不在**'}  {a['what']}")
    for path in bad:
        print(f"    **{path} を「届いている」と書いたが、ファイルが無い**")
    return 1 if bad else 0


# **どの材料にも1件も入っていない母集団。** 「0件だった」ではなく「見ていない」
ABSENT = [
    dict(
        key="lip_files",
        predicate="`.lip` ファイル 1,986(異なり 1,224)/ 124,663層。"
                  "**母集団に入っている**(作者の決定 2026-10-27: 「入れる」)。"
                  "**ただし画素は1頁も無い**",
        n_pages=1986,
        measured_by="抽出側(um350)",
        as_of="2026-10-27",
        needs="**`.lip` の画素。** 便 n4 の `rendered` は**全行が空**で、"
              "抽出側の言明は**「描けなかったのではなく、描いていない」。** "
              "**描く発注を出すかどうかは、まだ誰も決めていない**",
        satisfied_when=("source", "lip_rendered"),
        caveat="**「母集団に入っている」と「全部の表に入っている」は別である** —— "
               "**名前の側(依頼7)と構造の側(依頼8)には前から入っていた**"
               "(437,561 = `.clip` 312,898 + `.lip` 124,663)。"
               "**素性の側(画素の分布)には 0 頁。** "
               "`ext_lip = 0` の行は「`.lip` を数えて 0 だった」ではなく"
               "**「その材料に `.lip` が無い」**(T126)。"
               "**形式ごとの行は `population_by_format.csv`**",
    ),
    dict(
        key="comicstudio_pages",
        predicate="**ComicStudio の頁。母集団に入れる**(ユーザーの決定 2026-10-30)。"
                  "**この集計には1頁も入っていない** —— "
                  "**0 は「数えて 0」ではなく「材料が無い」**(T126)",
        n_pages=0,
        measured_by="census track(**この集計に入っている行を数えた**)",
        as_of="2026-10-30",
        needs="**その場で取り直した頁の一覧**(`.cpg` のパスを鍵に、作品・年・層の"
              "ファイル数・バイト数・版)。**先方の 8月の調査の数(`.cpg` 4,144 / 作品 96)は"
              "引用であって在庫ではない** —— 先方自身が「数え直していない」と書いている。"
              "**9,653 との重なりも要る**(別形式なので別の頁のはずだが、測っていない)",
        satisfied_when=("source", "comicstudio_pages_list"),
        caveat="**表の形は census が決めた**(ユーザーの指示): "
               "**`.clip` の層の表に足さず、別の表を2本立てる** —— "
               "**埋まらない列が空で並ぶと「空」と「不明」が見分けられない**(T148)。"
               "**共有するのは両形式で同じ述語が引ける列だけ**"
               "(パス・名前・拡張子・バイト数・版・形式)。"
               "**`format` の3つめの値として `population_by_format.csv` に入る**",
    ),
    dict(
        key="other_drive_files",
        predicate="**もう1台のドライブ(D)の札 2025 / 2025b の下のファイル** —— "
                  "`.clip` 1,085 + `.lip` 100(約 19GB)。"
                  "**ディレクトリーを歩いて数えただけで、1つも開いていない**"
                  "(抽出側 2026-10-30 19:15)。**この集計には1頁も入っていない**",
        n_pages=1185,
        measured_by="抽出側(um350。**ディレクトリー歩行。ファイルは1つも開いていない**)",
        as_of="2026-10-30",
        needs="",   # **満たされた**(2026-10-31、1,185行の hash。読み取り誤り 0)
        satisfied_when=("source", "other_drive_inventory"),
        caveat="**2026-10-31 に hash が届き、被りが数になった** —— "
               "**母集団の 9,653 と1バイトも違わないファイルは 385**"
               "(異なりでは 245)、**残り 800 は母集団に無い。** "
               "`census/other_drive_overlap.csv` と `other_drive_twin_years.csv`。"
               "**同じ絵かどうかには届かない**(描き直し・保存し直しは別の hash)。"
               "**頁として同定していない。** F 側は `pages` の行がファイルなので"
               "同じ述語で 1,185 になるが、**重複も版も未測で、"
               "9,653頁との被りは 0 とも全部とも言えない**(hash が未着)。"
               "**中身が作者の言う編かも分からない** —— "
               "**名前が合う所までで、先方が「中身は見ていないので不明」と明記した**"
               "(条例6)。**同じ名前のフォルダーが2つの札の下に在り、`.clip` の数も"
               "同じ(40)** —— **hash が来るまで同じものか違うものか分からない。"
               "二重に数えない**(条例9)。"
               "**42組の編の表に無いフォルダーが2つ在る**(`.clip` 49 と 1)。"
               "**札 2026 はこの数に入っていない**(作者が対象外と決めた)",
    ),
]

# **値そのものが近日動く列。** 動いたらこの集計の数は古くなる(基盤 2026-10-26)
VOLATILE = [
    dict(
        column="drawn_ratio",
        scope="`v_layer_kind` = `mask_fill` の層",
        what="抽出側が読み手を直すと、いま 0 と読まれている値が 1 付近へ動く",
        why="`mask_fill`(トーン・べた塗り)の本体は `InitColor 255` = 全面で、0 は「空」ではない",
        n_affected_in_req3=11455,
        note="動かす前に一声もらう約束。**この集計の mask_fill 行は、その連絡で無効になる**",
    ),
]


def by_key(key: str) -> dict:
    """材料を**鍵で引く。** 添字で引くと、材料を1つ挿した日に別の CSV を読む
    (2026-10-26 に実際そうなった: `LOCAL_CSV[1]` が added33 から InitColor に変わった)。
    """
    for s in LOCAL_CSV:
        if s["key"] == key:
            return s
    raise KeyError(f"材料が宣言されていない: {key}")


# **`needs` が古くなっていないかの点検。**
# 2026-10-26 に `db_layers` の `needs` が「`.lip` 後の層数はまだ来ていない」と
# 書いたままで、**`db_all_layers`(437,561)がその更新だった** ——
# **同じ表の2行が食い違っていた。** 手で見つけたので、**次を器具が見つける形にする。**
#
# `satisfied_when = ("source", <鍵>)` は「その材料が `LOCAL_CSV` に在れば満たされた」。
# **満たされているのに `needs` が空でない行**は指摘。
# **満たされていないのに `needs` が空の行**も指摘(頼み忘れ)。


def audit_needs() -> int:
    """**この点検が届く範囲**(名前より狭く書く):

    - **届く**: 宣言した材料が `LOCAL_CSV` に入ったのに `needs` が残っている /
      入っていないのに `needs` が空(頼み忘れ)
    - **届かない**: **`satisfied_when` を宣言していない行**(`clip_files` のように
      条件なしの行は、何も点検されない)。そして**材料が届いても中身が
      その問いに答えていない場合** —— 鍵が在ることと、答えが在ることは別
    """
    rows = SUPPLIER_REPORTED + ABSENT
    have = {s["key"] for s in LOCAL_CSV}
    bad = []
    print("  [`needs` 欄の点検] 満たされた材料が来たのに `needs` が残っていないか")
    for s in rows:
        cond = s.get("satisfied_when")
        needs = (s.get("needs") or "").strip()
        if cond is None:
            state = "条件なし"
            ok = True
        else:
            kind, key = cond
            # **札ごとに分かれて届く納品が在る**(墨の濃さ2段目は6ファイル)——
            # **どれか1つを代表に選ぶと、残りが届かなくても満たされたことになる。**
            # だから**前綴りで見る形**を足した(2026-10-31)
            satisfied = ((kind == "source" and key in have)
                         or (kind == "source_prefix"
                             and any(h.startswith(key) for h in have)))
            # **「未着」と「届いているが未統合」を言い分ける** ——
            # 同じ語にすると、**入れ忘れが依頼待ちに見える**
            pending = any(a.get("supplies") == key for a in ARRIVED_NOT_INTEGRATED)
            state = (f"{key} は在る" if satisfied else
                     f"{key} は**届いているが未統合**" if pending else
                     f"{key} は未着")
            ok = (not needs) if satisfied else bool(needs)
            if not ok:
                bad.append((s["key"], state, needs))
        print(f"    {s['key']:16s} {state:28s} needs={'空' if not needs else '有り'}"
              f"  {'' if ok else '**食い違い**'}")
    if bad:
        for key, state, needs in bad:
            print(f"    **{key}: {state} なのに needs が"
                  f"{'残っている' if needs else '空'}**")
        return 1
    print("    食い違いなし")
    return 0


# **作者から直接受けた属性。** 受領 CSV ではなく**便そのもの**が材料なので、
# `LOCAL_CSV` と同じ名簿に入れない —— **行の数が「層」や「頁」ではなく「言明」である。**
# **測定ではないので、`measured_by` ではなく「誰が言ったか」を書く**(条例6)
AUTHOR_NOTICES = [
    dict(
        source="`note_n_*_20261030.md`(3通)",
        supplier="作者 —— Track N 経由",
        as_of="2026-10-30",
        n_rows=4,
        note='**作者の裁定3件と、この表についての但し書き1件** → `author_rulings.csv`。**census は作者に直接聞いていない**(Track N 経由)。**作者の言葉をそのまま引く。** **対象の名前は公開物に書かない** —— 対応は `data/census_full/author_rulings_subjects.csv`(`.gitignore` 配下)。**5つのうち1つは構造 DB の外に在り、残り4つも候補が1対1に決まっていない**',
    ),
]
