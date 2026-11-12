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
        key="brushstyle_patternstyle_pages",
        path="inbox/clip_brushstyle_patternstyle_pages_20261103.csv",
        predicate="**形式に `PatternStyle` が残っている頁**(頁ごと1行、9,648行。`.clip` 7,662 は測定、"
                  "`.lip` 1,986 は開けたかだけ)。**④の同定ではない**",
        supplier="抽出側(um350。作業用の器具、commit なし)", as_of="2026-11-03",
        caveat="**sha256 検証済み。** **公開表は `census/pattern_style_pages.csv`** —— "
               "**供給元の合計と全部一致、構造 DB の `vector_objects > 0` の数と `.clip` 全頁で一致。** "
               "**納品の `n_vector_layers` はコマ枠の層を含む**(構造 DB の `n_vector_layers` は外す)",
    ),
    dict(
        key="l1_layer_bbox",
        path="inbox/census_l1_layer_bbox_20261104.csv",
        predicate="**線画の層(墨の濃さの層別 L1)の描かれた画素の範囲**(1行1層、49,786層・1,724頁。恒久除外の頁の 14層を落とした)",
        supplier="抽出側(um350。器具 `measure_layer_extent.py` `e13cf72`)", as_of="2026-11-04",
        caveat="**sha256 は便の先頭16桁と一致。** **公開表は `census/l1_layer_extent.csv`** —— "
               "**鍵の集合は墨の濃さの表から引き直して一致、`n_px_alpha_gt0` は 49,784層で一致**(違う2層は1段目の表が古い側)",
    ),
    dict(
        key="face_layer_alpha_mask",
        path="inbox/census_face_layer_alpha_mask_20261104.csv",
        predicate="**面の層の本体のアルファとマスクの面の値**(1行1層 × 面、189,974行・1,724頁)",
        supplier="抽出側(um350。器具 `measure_layer_extent.py` `e13cf72`)", as_of="2026-11-04",
        caveat="**sha256 は便の先頭16桁と一致。** **公開表は `census/face_binary_alpha.csv`** —— "
               "**マスクは `mask_cover_request3_v2`、本体は `faint_alpha_905_v2` と重なる層で全部一致。** "
               "**形式5 の本体は調査中**",
    ),
    dict(
        key="zlib_affected",
        path="inbox/zlib_affected_layers_20261101.csv",
        predicate="**zlib の頭を取り違えて画素を失っていた (頁, 層, 面)** —— "
                  "1,273行 / 本体 586層・マスク 687層 / 944頁 / 1,545ブロック",
        supplier="抽出側(um350)", as_of="2026-11-01",
        caveat="**sha256 検証済み。** **全 `.clip` 7,663頁の走査で、読めなかった頁は 0。** "
               "**取り違えたブロックは 74バイト目から読むと全部読めた**(読めないもの 0)—— "
               "**だから直した読み手で取り戻せる。** "
               "**この集計の材料への当たりは `census/reader_defect.csv`** —— "
               "**主結論の母集団(L1 形式2 の 40,323層)で本体に当たる層は 0**",
    ),
    dict(
        key="zlib_scan_pages",
        path="inbox/zlib_scan_pages_20261101.csv",
        predicate="**全 7,663頁の走査の集計**(取り違えの無い頁も入っている)",
        supplier="抽出側(um350)", as_of="2026-11-01",
        caveat="**sha256 検証済み。** **帰無値の側の表である** —— "
               "**当たらなかった頁も行として在る**ので、"
               "**「944頁」を母集団なしに引用しない**(12.3%)。"
               "**こちらで合計を引き直した**: ブロック 1,545 / 当たった頁 944(一致)",
    ),
    dict(
        key="occ_v2_still_unread",
        path="inbox/occ_v2_still_unread_mono_20261101.csv",
        predicate="**遮蔽の `_v2` でもまだ空のまま残った 28層(27頁)**",
        supplier="抽出側(um350)", as_of="2026-11-01",
        caveat="**sha256 検証済み。** **既知セルとして使っている** —— "
               "**抽出側は「28層は全部 zlib の取り違えだった」と書いた。"
               "こちらで影響表に当てて 28/28 を確かめた**(`reader_defect.csv`)。"
               "**遮蔽そのものは基盤の領分**で、この集計は被覆を1列も持っていない",
    ),
    dict(
        key="req3_layer_kind",
        path="inbox/f_request3_layer_kind_rendered_v2_20261102.csv",
        predicate="依頼3の872頁の、**フォルダー以外の全層**(種別×描画×drawn_ratio)",
        supplier="抽出側(um350。器具 `config/b1_rebuild/layer_kind.py`)", as_of="2026-11-02",
        caveat="**【2026-11-02】`_v2` に替えた**(読み手の2つの直し `a1131bc` の後の作り直し)。**前の版は撤回しない** —— **動いた行は全部、直しか宣言された理由で説明が付き、供給元の数と一致した**(`material_versions.csv`)。"
               "**`drawn_ratio` が 988行、`v_layer_kind` が other → raster に 1,276行動いた** —— "
               "**後者は読み手の直しではなく 10-26 17:57 の構造 DB の変更**(`layer_type = 3` を raster と数える)。"
               "**フォルダー層は入っていない。** 層数を added33 の `n_layers` と比べない",
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
        path="inbox/faint_alpha_905_v2_20261102.csv",
        predicate="905頁の `body_init` / `body_blocks` / `body_stored` と"
                  "**アルファ≥1 の比**(62,105層。依頼3の 56,576層を全部含む)",
        supplier="抽出側(um350)", as_of="2026-10-27",
        caveat="**【2026-11-02】`_v2` に替えた**(読み手の2つの直し `a1131bc` の後の作り直し)。**前の版は撤回しない** —— **動いた行は全部、直しか宣言された理由で説明が付き、供給元の数と一致した**(`material_versions.csv`)。"
               "**動いたのは 1,060行で全部増える向き**(1ビット 971 / 取り違え 89)。"
               "**薄い画素の数 `alpha_1_25_px` が動いたのは 15行だけ**。"
               "**依頼3の3列がこれで埋まった** —— 下限の層が何層かを計数できる。"
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
        key="mask_cover_request3_v2",
        path="inbox/mask_cover_request3_v2_20261102.csv",
        predicate="依頼3のマスクの被覆の `_v2`(**16,273行・766頁。元の版 + 10-30 の原点の直し + "
                  "11-02 の読み手の直し**)",
        supplier="抽出側(um350。器具は元と同じ `list_mask_cover.py`)", as_of="2026-11-02",
        caveat="**動いたのは 113行**(取り違え 99 + 1ビットのマスク 14)。**元の版と原点の直しの便は"
               "そのまま読み、重ねた結果とこの便を鍵で突き合わせてから、この便の値を表に使う** —— "
               "**10-30 の差し替えの印字(383行)は残す。** "
               "**取り違えで被覆が減った 59行は全部初期値 255、増えた 32行は全部初期値 0**(抽出側の実測) —— "
               "**読めなかったブロックは初期値のまま数えられていた**",
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
        path="inbox/ink_darkness_stage2_2019_v2_20261102.csv",
        predicate="**墨の濃さの2段目、札 2019**(1行1層、6,124層)。"
                  "**【2026-11-02】`_v2` に替えた** —— **動いたのは形式5 の `線画` 2層だけ**"
                  "(すでに「調査中」で外してある側。**主結論の母集団は1行も動いていない**)",
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
        key="mono_1bit_affected_layers",
        path="inbox/mono_1bit_affected_layers_20261031.csv",
        predicate="**1ビットの層を空として読んでいたため、どの納品で層が落ちていたか**"
                  "(1行1層。`page_path`/`layer_id`/`layer_name`/`v_layer_kind`/"
                  "`formats`/`n_px_alpha_1bit`)",
        supplier="抽出側(um350。読み手 `clip_render.py` `1927ece`)",
        as_of="2026-10-31",
        caveat="**census の墨の濃さには影響しない**(先方が測り直して確かめた)—— "
               "**2段目のモノクロ層 185行のうち 181行は1ビット対応後もブロックが全部空"
               "(本当に画素が無い。名前はどれも Blender の層)、残り4行は形式2 で読めていた。** "
               "**効いているのは遮蔽の全数計測(10-28、基盤)** —— "
               "**遮る層のうち 6,274層・2,777頁を空として数えていた。** "
               "**`_v2` は 2026-11-01 に届いた** —— **`empty` → `ok` が 6,538層、"
               "被覆が変わった層が約2万で全部増える向き(減った層は 0)。"
               "だから「低めに出ている」の但し書きはこの表では消えた**(先方の言明)。"
               "**ただし 27頁の 28層は `_v2` でも空のままで、この一覧からも漏れていた** ——"
               "**【11-03】先方が数え直した「一覧の漏れ 81層・76頁」は、下の 81層と同じ層である** —— "
               "**census が鍵で引いた: 抜き取り(本体)に在って一覧に無い層 81・76頁、81層とも本体に取り違えが在り、"
               "一覧に在って抜き取りに無い層 0。** **この一覧で1ビットの層を数えると 81層少なく出る**"
               "(先方: 11-02 以降の作り直しには効かない)。"
               "**一覧を作った走査で 81層が展開エラーになっていた"
               "(影響は 6,506層・2,777頁ではなく 6,587層・2,804頁)。"
               "28層は全部 zlib の頭の取り違えで、こちらで 28/28 を確かめた**"
               "(`reader_defect.csv` の既知セル)。"
               "**この集計は遮蔽の被覆を1列も持っていない**(写しだけ)",
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
        path="inbox/deliver_census_koma_zlib_rebuild_20261102.md",
        what="**コマ枠の全表 `koma_frames_v2_zlib`**(7,658枠。取り違えがコマ枠のマスクに当たった"
             "113頁を作り直した。**IoU ≥ 0.99 が 6,528 → 6,551**)。**宛先は census**",
        why_not_yet="**この集計はコマ枠の表を公開表にしていない**(10-30 の版も未統合のまま)。"
                    "**sha256 3本一致。** **古い列は1行も動いていない**(先方の言明。引き直していない)",
    ),
    dict(
        path="inbox/deliver_b1_layer_colour_v2_20261102.md",
        what="**層の色の表 `layer_colour` の作り直し**(依頼3 72,839行 / ol29 / yearroot4。"
             "**動いた層 1,072**)。**先方は自分の構造 DB の `layer_colour` も置き換えた**",
        why_not_yet="**こちらの構造 DB は 10-26 17:57 の断面のままで、`layer_colour` は前の版** —— "
                    "**`mask_material.py` の `formats` はこの古い列を読む。** "
                    "**断面を取り直すかは、DB を丸ごと受け取るときに決める**(器具は読むだけ)。"
                    "**依頼3の層の種別の `_v2` だけは公開表に入れた**(`material_versions.csv`)",
    ),
    dict(
        path="inbox/push_h_i_affected_rows_20261102.md",
        what="**10-23 の調査の表3本と、器具の残っていない表15件の当たり**(抽出側 → 基盤・Track L)",
        why_not_yet="**この集計は3本のどれも読んでいない**(トーンの全数 / 柔らかい塗り / 塗り・トーンの棚卸し)。"
                    "**15件のうち census が読むのは `body_initcolor`・`grad_v1_v2_layer_map`・"
                    "調査中の候補の3件で、当たりは 0 / 0 / マスクの IoU 2枠(区分は変わらない)**",
    ),
    dict(
        path="inbox/push_h_fill_tone_inventory_affected_v2_20261102.md",
        what="**塗り・トーンの棚卸しの当たりの一覧 `_v2`**(10,435行。抽出側 → 基盤・Track L)",
        why_not_yet="**この集計は棚卸しの表を読まない**(先方も「census の表には効かない」と書いた)。"
                    "**sha256 2本一致**",
    ),
    dict(
        path="inbox/push_h_tone_layers_full_affected_v2_20261103.md",
        what="**トーンの全数の当たりの一覧 `_v2`**(414行 + 増えた 377行。抽出側 → 基盤・Track L)",
        why_not_yet="**この集計はトーンの全数の表を読まない。** **sha256 2本一致**",
    ),
    dict(
        path="inbox/push_sketch_set_author_answer_and_picks_20261103.md",
        what="**下絵の層のフォルダーについての作者の答えと、8月の対で下絵に選んだ層の一覧**(1,848対。"
             "`_v2` は目視の列を埋めたもの)。**宛先は census**",
        why_not_yet="**作者の答えは但し書きと `worksheet_folder_reach.csv` に取った。** "
                    "**対の一覧は抽出側の器具の選びの記録で、census の表の材料ではない**(対は作者の決定で作り直しを後回し)。"
                    "**目視は先方のもの**(下絵 97 / うち位置が合わない 3 / 資料 2。census は見ていない)。sha256 一致",
    ),
    dict(
        path="inbox/push_sketch_set_fill_occlusion_and_eye_check_20261103.md",
        what="**下絵の層のフォルダーの全面の塗りつぶし 243層が遮蔽の表で遮蔽物になっていた件**(抽出側 → 基盤)",
        why_not_yet="**遮蔽の採点は基盤の領分**で、この集計は被覆を1列も持っていない。sha256 一致",
    ),
    dict(
        path="inbox/push_occlusion_worksheet_rule_decided_20261103.md",
        what="**作者の決定: 作業用の層(下絵の層のフォルダーの中と、名前が `調整` の層)は遮蔽物にしない。"
             "半透明の主値は 10-27 の定義のまま、重みつきの被覆を別の列で足す**。遮蔽の表 `_v4` を測り直し中",
        why_not_yet="**遮蔽の採点は基盤の領分。** **census の遮蔽物の名前の表(`occluder_*`)は名前の規則で数えた類で、"
                    "`調整` を類の名前に持たない。** **ただし下絵の層のフォルダーの中の層が類に入っているかは数えていない**"
                    "(類は層名とフォルダーの鎖で引くので、中に入りうる。条例6)。sha256 一致",
    ),
    dict(
        path="inbox/deliver_soft_fill_layers_v2_20261102.md",
        what="**柔らかい塗りの表 `_v2`**(1,754 → 1,758行。**10-26 のマスクの初期値の直しの分も動いた**)",
        why_not_yet="**この集計は柔らかい塗りの表を読まない**(宛先は基盤・Track L)",
    ),
    dict(
        path="inbox/deliver_beta_ako6_ako7_v2_20261102.md",
        what="**ベタ2作品の `_v2`**(ベタが 14層増えた。**番号のずれた8枚が在る**)",
        why_not_yet="**宛先は基盤・Track L。** **納品の木に増えたフォルダー2つは "
                    "`delivery.UNDECLARED_KNOWN` に入れた**(番号がずれているので、名前で畳む表に入れない)",
    ),
    dict(
        path="inbox/deliver_m_colour_pilot2_four_pages_20261102.md",
        what="**Track M の採用4頁の層 170枚とマスク 21枚**(`colour_pilot2/`、`dataset_` が付かない)",
        why_not_yet="**宛先は Track M で、この集計はラスタを読まない**(条例10)。"
                    "**数だけ引き直した**: 層の数 40 / 50 / 77 / 45 は先方の既知セルと一致、"
                    "**マスクを持つ層 21 は、こちらが可視を問わず数えた `has_mask` と4頁とも一致**",
    ),
    dict(
        path="inbox/deliver_foundation_priority65_zlib_20261102.md",
        what="**優先65頁の成果物の作り直し**(層の一覧 151層 / 全層 PNG 2層 / トーン 7面 / コマ枠 6枠)",
        why_not_yet="**宛先は基盤・Track L。** **納品の木に増えたフォルダー3つは "
                    "`delivery.UNDECLARED_KNOWN` に入れた**",
    ),
    dict(
        path="inbox/note_p_census_columns_for_split_20261107.md",
        what="**Track P の依頼: 発注候補 941頁の作品の単位・カラーかモノクロか・作業途中の扱い**",
        why_not_yet="**答えた**: 器具 `p_split_columns.py`、頁ごとの全量は非公開(P と基盤へ)。"
                    "**作品の単位は作者の編の裁定と、ファイルの中身の共有で束ねた**(名前の似かたでは束ねない。編の境を跨ぐ共有は束ねない)。"
                    "**カラーかモノクロかの列は無い**(`colour_ratio_vs_author.csv` で作者の分類を再現しなかった)",
    ),
    dict(
        path="inbox/deliver_m_holdout8_20261107.md",
        what="**抽出側 → Track M の納品の写し**(8頁の層の画素とマスクのラスタ、`holdout8_20261107`)",
        why_not_yet="**ラスタは census が読まない**(条例10)。**census の材料表への外の既知セル**: "
                    "**形のあるマスク(葉)が census の材料の下限と8頁とも一致、層の数も census の行と8頁とも一致**(抽出側が別の器具で数えた)。"
                    "sha256 一覧(先頭 `63d5883d0514fc0b`)で 752本とも一致を census が確かめた。"
                    "**描画できなかった2層は構造 DB で `v_layer_kind = text`**(抽出側の推定を構造の側で確かめた)",
    ),
    dict(
        path="inbox/push_occlusion_edge_distance_request_20261107.md",
        what="**基盤 → 抽出側の依頼の写し**(覆われた線の画素の、遮蔽物の輪郭からの深さ)",
        why_not_yet="**宛先は抽出側、採点は基盤。** census の作業は無い",
    ),
    dict(
        path="inbox/reply_interview_ui_parts_20261107.md",
        what="**基盤の通達: 聞き取り UI の共通部品と指針**(と追補 `reply_interview_ui_parts_update_20261107.md`)",
        why_not_yet="**次に聞き取りの画面を作るときから使う。** いま作る画面は無い。返事は要らない",
    ),
    dict(
        path="inbox/deliver_p_order300_part1_20261109.md",
        what="**抽出側 → census・Track P の納品: 300頁の層の画素とマスク**(part1 = 8便 240頁、part2 `deliver_p_order300_part2_20261109.md` = 残り2便。着手の返信 `reply_p_order300_started_20261109.md`)",
        why_not_yet="**ラスタは census が読まない**(条例10)。抽出側の照合: 層は census の行と、材料は名簿と300頁とも一致、"
                    "書けなかったマスク3枚は census の面の表の `mask_unread` 3層と同じ層。"
                    "**未統合**: 300頁の層の行の画素の列(manifest の値)をまだ埋めていない",
    ),
    dict(
        path="inbox/deliver_stage_layer_pixels_20261109_v2.md",
        what="**抽出側の納品: 工程の段の候補 53,243層の画素の数**(初版 `deliver_stage_layer_pixels_20261109.md` は表のパスの誤記だけ違う。コマ枠のベクターで埋めた 505層 `deliver_stage_layer_pixels_koma_vector_20261110.md`)",
        why_not_yet="**統合した**: `layer_role_table.py` で1層1行の表に入れた(505層は印の列 `all_masks_from_koma_vector`)。sha256 は一覧と一致",
    ),
    dict(
        path="inbox/note_koma_mask_not_saved_20261110.md",
        what="**抽出側の知らせ: 新しい版の CSP がコマフォルダーのマスクなど作り直せるラスタを保存していない**(2022b〜2024c と 2017 の1頁)",
        why_not_yet="**census の面の表の `mask_unread` もこれと思われる(推定。抽出側)。** 段の表の 505層はコマ枠のベクターで埋めた値を使う。面の表・材料表の作り直しは頼んでいない",
    ),
    dict(
        path="inbox/note_census_role_predictor_column_order_20261109.md",
        what="**Track E の発注: 層の役割の予測器のための1層1行の表**(ユーザー決定: 予測器は E が持つ。基盤の申し送り `reply_judge_layer_role_assigned_20261109.md`)",
        why_not_yet="**答えた**: `layer_role_table.py`(53,243層・42列、非公開)。分け方の鍵は `split_units.py`(全作品フォルダー、P の単位と一致)",
    ),
    dict(
        path="inbox/note_census_hearing_round2_design_20261110.md",
        what="**Track E の発注: 次の聞き取りの便の設計**(便A 無作為・棒を読む唯一の便 / 便B 薄い段 / 便C 迷う層。ユーザー承認)",
        why_not_yet="**便A の頁を先に引いた**(`stage_hearing_round2_a.py`、seed 20261110、14項目・97層。画素の残り46層を抽出側に発注)。"
                    "**便B は E が仮説を決めてから**(引ける列を返事に書いた)。表はまだ作者に渡していない",
    ),
    dict(
        path="inbox/note_p_census_line_layer_name_misses_20261110.md",
        what="**Track P の共有: 層の名前 `線画` で線を引けなかった3頁**(返信不要)",
        why_not_yet="**層の役割の予測器(Track E)の材料。** census の作業は無い。3頁とも段の候補の表の外(名前で両方が当たる頁ではない)",
    ),
    dict(
        path="inbox/push_occlusion_edge_koma_mask_question_20261110.md",
        what="**基盤 → 抽出側の問い合わせの写し**(遮蔽の深さの表は読めないコマのマスクをどう扱ったか。第2段の切り出しの発注 `push_occlusion_edge_stage2_crops_20261109.md` の写しも)",
        why_not_yet="**宛先は抽出側、採点は基盤。** census の作業は無い",
    ),
    dict(
        path="inbox/note_census_request_drawing_stages_20261108.md",
        what="**Track E の発注: 工程の段(あたり / 下絵 / 清書)で同定した層**(先に、やるかを決める数3つ)",
        why_not_yet="**答えた: いまは「できない」。** 段を再現する列も作者の裁定も無い(作業用フォルダーで代えた前例は的中 2/129)。"
                    "名前の側の大きさだけ出した(線画系と下絵系の両方が在る頁 `.clip` 2,102 / `.lip` 730、的中率は測れない)。"
                    "**`.clip` の画素の3分けは段の同定が来てから。** 提案: 作者に段を付けてもらう標本(E とユーザーが決める)",
    ),
    dict(
        path="inbox/deliver_occlusion_edge_20261109.md",
        what="**抽出側 → 基盤の納品の写し**(遮蔽物の輪郭からの深さ、13,809層・2,000頁。IC1 全層一致)",
        why_not_yet="**宛先は基盤、採点も基盤。** census の作業は無い。本文の sha256 は一覧と一致(CSV は取っていない)",
    ),
    dict(
        path="inbox/note_n_census_process_pilot20_20261110.md",
        what="**Track N の依頼: 工程の聞き取りを20作の試しから**(2021〜2023b から10作 / 2024 系から10作、裁定済み11作を除く。ユーザー承認)",
        why_not_yet="**表を作った**: 器具 `process_hearing_pilot20.py`(seed 20261110、`.clip` の頁が在る作品、札の直下の頁は候補外)。"
                    "**11-10 に回収して `author_process.csv` に取った**(20行、値は書き換えない)。選択肢に「線画が無い」が無かった(11行がコメントで補った)",
    ),
    dict(
        path="inbox/reply_foundation_occ_edge_koma_mask_20261110.md",
        what="**抽出側 → 基盤の返事の写し**(遮蔽の深さの表は読めないコマのマスクを掛けない扱い。影響のありうる層 241頁・347層、鍵の CSV つき)",
        why_not_yet="**宛先は基盤。** census の作業は無い。本文と CSV の sha256 は一覧と一致",
    ),
    dict(
        path="inbox/deliver_colour_rounding_fix_20261110.md",
        what="**抽出側の追加便**(書き出しが丸めで飛ばした層 49 = P の300頁 45 + 他の便 4、と P の300頁の書けなかったマスク3枚をベクターで作った値)",
        why_not_yet="**P の300頁の分は重ねた**: `p_layer_rows.py` → `p_order300_page_layers_20261110_v2.csv`(`pixel_source` / `mask_source` に印)。"
                    "**他の便の4層は M の判断**(holdout8・pilot。census の表に画素の列は無い)",
    ),
    dict(
        path="inbox/note_fill_region_exclude_lettering_20261110.md",
        what="**fill-region の発注: 「絵から除外」する層の列**(描き文字 / 黒消し / エフェクト / 絵。作者「国勢調査によって除外されるべき」「クレンジング」)",
        why_not_yet="**受けた。分担はユーザーの決定(11-10)**: 予測は Track E の層の役割の予測器に区分を足す。census は名前の当て方の列(推測)・"
                    "区分の聞き取りの表・的中と取りこぼし。**名前の列から始める。聞き取りは E の便の設計を待つ**",
    ),
    dict(
        path="inbox/push_occlusion_edge_stage2_cancel_20261110.md",
        what="**基盤 → 抽出側の取り消しの写し**(遮蔽の深さ 第2段の切り出し。作者「この実験に意味はない」)と追記(閉じたのは食い込みの検証だけ、遮蔽の調査は続く)",
        why_not_yet="**宛先は抽出側。** census の作業は無い(追記 `push_occlusion_edge_stage2_cancel_addendum_20261110.md` も同じ)",
    ),
    dict(
        path="inbox/deliver_stage_round2a_and_draw_colour_20261110.md",
        what="**抽出側の納品: 便A の残り 46層の画素と、段の候補 53,289層の描画色**(sha256 一覧と一致。描画色の意味は抽出側の推定: CSP のレイヤーカラー = 表示の置き換え)",
        why_not_yet="**層の役割の表に入れた**: `layer_role_table.py` → `layer_role_table_20261110_v2.csv`(53,289層、描画色4列)。"
                    "**既知セル: P の300頁の表(census が manifest から組んだ)と重なる 9,453層で3列とも一致**。ラベルの在る層で enable=1 は 0",
    ),
    dict(
        path="inbox/note_census_b_hypotheses_measured_20261110.md",
        what="**Track E の決定: 便B は名前を使わず 芯4頁 + 広4頁**(当て込みの閾値と札に書く)/ 重なり順の向きの問い / 「どちらも無い」頁の構造の列の見積りの問い",
        why_not_yet="**頁を引いた**(描画色2頁を足して10頁、`stage_hearing_round2_b.py`)。**550行で、8頁が1頁で20行を超える** —— 作者の大きさの規律と合わないので表は渡さず E とユーザーに返した。"
                    "向き: 位置1 = 手前(用紙 7,642層が全部、兄弟の中で最小の `order_in_parent`)",
    ),
    dict(
        path="inbox/note_census_exclude_axis_and_20row_split_20261110.md",
        what="**Track E: 除外は別の軸(2軸)/ 便A・便B を20行ずつに割る / 描画色はラベルの層に値が無く測れない**",
        why_not_yet="**便A を E の割り方で5束にした**(`stage_hearing_round2_a_sheets.py`。A-2 は1頁42行の例外。再掲2頁は 11-09 の並びと一致)。作者に渡すかはユーザー",
    ),
    dict(
        path="inbox/note_census_template_first_hold_hearing_20261110.md",
        what="**Track E: 順番を変える(ユーザーの指示)** —— 軸2 の聞き取りは保留、先にテンプレートの決定論的な除外。(a) 全層の層名の棚卸しの依頼",
        why_not_yet="**(a) を出した**: `layer_name_inventory.py` → `layer_name_inventory_20261110.csv`(非公開、層名 × 種別 16,479行)。(b) 画素の署名・黒消しの統計量は承認待ちで E が保留",
    ),
    dict(
        path="inbox/reply_e_sketch_contents_to_role_predictor_20261110.md",
        what="**基盤 → E の回送の写し**(下絵の層の中身の仕分けを役割の予測の枠へ。白い塗りがアルファで墨に数えられた件)",
        why_not_yet="**宛先は E。** census に効く点: **census の画素の数はアルファで数えている**(白で塗った画素も `n_px_a_gt0` に入る)。E に添えた",
    ),
    dict(
        path="inbox/note_census_a1_keep_choices_20261111.md",
        what="**Track E: 便A の選択肢は変えない(空は画素の列で決まる)/ A-2 を出す / 黒消しの別名の層名は材料のまま / 便B と軸2 に「空」の選択肢と機械の印**",
        why_not_yet="**A-2 をユーザーに送った**(添えた一言が A-1 と違うと E に記録)。答えの取り込みに「分からない∧空」の印字を足した。"
                    "**便B の印は 65/111層の画素が無く付けられない** —— 表は作らず、抽出側への一覧 `stage_layers_order_round2b_20261111.csv` を用意(発注はユーザーの承認待ち)",
    ),
    dict(
        path="inbox/note_fill_region_exclusion_impact_20261111.md",
        what="**fill-region: テンプレートと黒消しの別名の層は fill-region の単位に当たらない。当たるのは黒消し(別の名前の道)と描き文字・エフェクト**",
        why_not_yet="**census の作業なし。** 作者が黒消しと答えた単位は別の名前の道にある —— 別名の候補として作者に聞く項目に残す",
    ),
    dict(
        path="inbox/note_census_chousei_confirmed_20261112.md",
        what="**Track E: 作者がテンプレートと答えた塗りの層の名前を、ユーザーの承認で除外の表の「確定」にした**(落ちる層 52,009 → 54,392)",
        why_not_yet="**受けた。** 除外の表の正本は E。census が a の層の一覧を作るときは、この行を入れる",
    ),
    dict(
        path="inbox/deliver_q1_trial17_images_20261112.md",
        what="**抽出側の納品: 除外の track の17行の画(34枚、原寸)と点検の表**。2行は問う層が非表示で、頁の合成に写らない",
        why_not_yet="**除外の track へ回送した**(sha256 36件一致。census は画を開いていない)。非表示の2行の扱いは除外の track が決める",
    ),
    dict(
        path="inbox/note_census_name_hearing_round2_approved_20261112.md",
        what="**Track E: 層名の聞き取り 第2回(20行)の発注。問いは「どの頁でも絵から除外してよいか」。1行目は名前ではなく効果の列で引く**(ユーザーの承認あり)",
        why_not_yet="**表を作った**(`name_hearing_round2_sheet.py`。下書きの数を棚卸しと構造 DB で引き直して全部一致、名前の行と1行目の重なり 0層)。"
                    "**答えを取り込んで E に返した**(`name_hearing_round2_answers.py`。よい 12 / だめ 7 / 分からない 1、第1回とは別のファイル)",
    ),
    dict(
        path="inbox/deliver_template_pages_step3_20261112.md",
        what="**抽出側の納品: 第3段の全67頁(4,248層)**。画像素材の画素0 は空ではない見込み(元の画像を読んでいない)",
        why_not_yet="**Track E へ回送した**(sha256 一致)。読みは E",
    ),
    dict(
        path="inbox/note_q_census_bin1_received_ic1_ok_20261112.md",
        what="**除外の track: 便1 を受けた(既知の3層は一致)/ 規則を追補で直した / 便1 の一覧に作品単位の列がほしい**",
        why_not_yet="**作品単位の列つきの一覧を渡した**(発注した一覧は変えない)",
    ),
    dict(
        path="inbox/note_q_census_q1_trial17_approved_20261112.md",
        what="**除外の track: 作者に見せる17行にユーザーの承認。表と画の発注**",
        why_not_yet="**並びの鍵と見せる表を作り(層の名前は出さない)、抽出側へ画を発注した**(17層 × 2枚、原寸)。画は census が開かずに回送する",
    ),
    dict(
        path="inbox/deliver_grey_stats_bin1_20261112.md",
        what="**抽出側の納品: 階調の統計の便1(825層の色の内訳と連結成分 5,301行)**。既知の3層の画素数は一致",
        why_not_yet="**除外の track へ回送した**(sha256 一致。既知の3層は層の役割の表 v3 でも一致)。読みは除外の track",
    ),
    dict(
        path="inbox/deliver_template_pages_step3_smoke_20261112.md",
        what="**抽出側の納品: 第3段の試しの1頁(99層)**。残りの頁は続けて流す",
        why_not_yet="**Track E へ回送した。** 塗りの層の数え方がトーンの便と違う(この便は全面の約束を使わない)と添えた",
    ),
    dict(
        path="inbox/note_census_a_taken_over_canvas_and_tone_verdict_20261112.md",
        what="**Track E: a を引き取った / 作者がテンプレートと答えた塗りの層の名前を除外の表に足す提案(ユーザーの承認待ち)/ 規則が届き得る単位の上限 / トーンは名前を使わず効果の列**",
        why_not_yet="**受けた。** 除外の表に足す件はユーザーに上げた",
    ),
    dict(
        path="inbox/reply_census_pairs_v5_arrived_20261112.md",
        what="**基盤: 対の v5 が届いた。二重の頁の一覧は v5 で作ってよい(QC で外す頁を先に落とす)**",
        why_not_yet="**一覧を作った**(`duplicate_pages_v5.py`: 114組・257頁)。フォルダーのつながりで8行に畳んだ聞き取りの表を作った(`hearing_duplicate_folders_v5.py`)。"
                    "**ユーザーの承認(11-12「よい」)で作者に渡した。** 答えから採る頁の名簿を作り、基盤・D・抽出側に返した"
                    "(`duplicate_folders_v5_answers.py`: 二重の組の頁のうち 採る 174 / 落とす 83)。"
                    "**v2: ユーザーの決定(11-12「フォルダごとはずしていい」)で、同じフォルダーの組の外の23頁も落とした(落とす 83 → 106)。v1 は残した**",
    ),
    dict(
        path="inbox/deliver_tone_mask_pixels_ako6_20261112.md",
        what="**抽出側の納品: 1作品フォルダーのトーンの層のマスクの画素(4,784 + 167層)と、トーンの設定の値**。塗りの層は全面を描いた約束の値で数えた",
        why_not_yet="**sha256 一致。census で数え直して、塗りの層のうち自分のマスクが 0 の層 2,865(E と一致)。** 表 v3 には入れていない(塗りの層はほぼ母集団の外)。"
                    "塗りの層を表に入れるときはマスクそのものの画素の列を足す",
    ),
    dict(
        path="inbox/note_census_tone_clause_measured_and_a2_read_20261112.md",
        what="**Track E: トーンの条項に母数が付いた(1作品フォルダーで 2,865層がマスクで全部消えている)/ 便A の束2 の読み(採点なし)**",
        why_not_yet="**受けた。** 聞き取りの項目にする提案は除外の track の持ち分なので写した",
    ),
    dict(
        path="inbox/note_census_round2b_marked_and_tone_name_verified_20261112.md",
        what="**Track E: 便B の空の印を手元でも付けた(6層)/ トーンの名前の形と効果の列の照合 / 全層の 2×2 の依頼**",
        why_not_yet="**2×2 を出した**(全層で 名前∧有効 89,059 / 名前∧無効 61 / 無効でない側の取りこぼし 2,001)。空の6層は census と一致",
    ),
    dict(
        path="inbox/note_census_step3_signature_and_canvas_ask_20261112.md",
        what="**Track E: 第3段(テンプレートの画素の署名)を手元の表で走らせた / キャンバスの大きさの依頼**",
        why_not_yet="**キャンバスの集計を出した**(`template_pages_step3_order.py`。.clip で 16種)",
    ),
    dict(
        path="inbox/note_census_step3_corpus_order_approved_20261112.md",
        what="**Track E: 第3段の頁の発注(1作品単位1頁・層を全部、先に1頁)。ユーザーの承認あり**",
        why_not_yet="**抽出側へ発注した**(67頁・4,248層。.lip だけの15単位は外した)。納品待ち",
    ),
    dict(
        path="inbox/note_census_step3_rule_addendum_20261112.md",
        what="**Track E: 第3段の読みの規則に条項を1つ足した(発注は変わらない)**",
        why_not_yet="**census の作業なし。** 発注の中身は変えていない",
    ),
    dict(
        path="inbox/reply_census_duplicate_pages_20261112.md",
        what="**基盤の回送: 対の中で二重に入っている頁のどれを使うかは census(ユーザーの決定)**。採る版を名簿の列にしてほしい",
        why_not_yet="**受けた。対の新しい版ができてから、その頁で一覧を作り直し、作者に20行の試しで聞く**(作者に渡す前にユーザーの承認)",
    ),
    dict(
        path="inbox/reply_p_census_colour_stage3_order_20261112.md",
        what="**Track P: カラーかの聞き取りの第3段は P の頁を持つフォルダーから(ユーザー了承)** / 11-09 の層の種別の依頼の催促",
        why_not_yet="**種別の列を渡した**(`p_layer_kind_300.py`、40,879行、引けない行 0)。第3段の表はまだ作っていない",
    ),
    dict(
        path="inbox/note_p_census_layer_kind_300_20261109.md",
        what="**Track P: 300頁の全層に層の種別の列を**",
        why_not_yet="**11-12 に渡した**(11-09 から census が取りこぼしていた)",
    ),
    dict(
        path="inbox/note_q_census_bin1_amendment_fixed_order_go_20261112.md",
        what="**Track Q: 便1 の一覧を登録の追補で固定。発注を出してよい**",
        why_not_yet="**抽出側へ発注した**(SCP、着弾を確認)。納品待ち",
    ),
    dict(
        path="inbox/deliver_stage_round2b_and_draw_colour_20261112.md",
        what="**抽出側の納品: 便B の残り 65層の画素と描画色**(sha256 一覧と一致。頁のエラー 0、描いた 63 / 画素0 2)",
        why_not_yet="**層の役割の表 v3 に入れた**(53,354層。v2 の 53,289層は全列そのまま。既知セル: 描画色が P の300頁の表と 9,453層で一致)。"
                    "末尾に作者の定義 C の空の列を足した。便B の7束の表を作った(作者には便A の後に出す)",
    ),
    dict(
        path="inbox/note_census_round2b_65_order_approved_20261111.md",
        what="**Track E: 便B の65層の発注にユーザーの承認(列は変えない)。便B は全部通常の層なので、トーンの条項は試せないと札に書く**",
        why_not_yet="**既に 11-11 12:23 に発注済み**(行き違い)。納品待ち。札の但し書きは受けた",
    ),
    dict(
        path="inbox/note_census_tone_order_approved_and_mask_fill_lead_20261111.md",
        what="**Track E: トーンの層は塗りの種別に居るらしい(名前の形から。推定)。1作品の塗りの層を全部、の発注にユーザーの承認**",
        why_not_yet="**同じ作品フォルダーで、トーンのフォルダーの外の塗りの層 167 を抽出側へ追加で発注した**(これで塗りの層 4,482 が全部)。"
                    "効果の値の出所と網の角度を抽出側に聞いた",
    ),
    dict(
        path="inbox/note_census_empty_definition_fixed_20261111.md",
        what="**Track E: 「完全に空」は作者が決めた —— 自分のマスクだけ見る(C)+ 画素は在るが自分のマスクで全部消えている層も空(トーンの形)**。空を段の選択肢にしない",
        why_not_yet="**受けた。** 便B の機械の印を C にし、段の選択肢から空を外した(行と頁は不変)。段の候補の表で空 4,980層(E と一致)。表の空の列の定義に C が正本と書いた",
    ),
    dict(
        path="inbox/note_census_tone_layer_flag_20261111.md",
        what="**Track E: トーンの層を示す列が手元に無い**(段の候補の表で `tone_enabled` が全部 0)。列の出所と、トーンのフォルダーの中の層を1作品ぶん",
        why_not_yet="**答えた: `tone_enabled` が真の層は構造 DB に 91,060層で全部が塗りの層(`mask_fill`)。段の候補の表は raster と vector なので母集団の外。** "
                    "1作品フォルダーぶんの構造の列を渡した。**画素はユーザーの承認(11-11「全部でいい」)で抽出側へ発注した**(4,784層、先に1頁で試す。"
                    "マスクそのものの画素の列を1つ足して頼んだ)",
    ),
    dict(
        path="inbox/note_q_e_census_hello_b_and_kuro_keshi_registered_20261111.md",
        what="**Track Q: b を持つ / 黒消しの登録を固定 / 便1 の対照の引き方の案**",
        why_not_yet="**受けた。** 対照の一覧を Q の案で引いた(`grey_stats_bin1_order.py`)",
    ),
    dict(
        path="inbox/note_q_census_order_grey_stats_bin1_approved_20261111.md",
        what="**Track Q: 階調の統計の便1 にユーザーの承認。対照の引き方と数を Q に渡してから発注**",
        why_not_yet="**一覧を引いた(825層、既知の3層を含む)。** 引き方と数を Q に渡した。Q の追補の後、**11-12 に抽出側へ発注した**",
    ),
    dict(
        path="inbox/note_census_empty_definition_20261111.md",
        what="**Track E: 「完全に空」の定義の提案(定義D)** —— 作者の裁定「画素が無くてもマスクがあれば空ではない」。表の空の列はマスクを見ていない",
        why_not_yet="**census で数え直して一致**(段の候補 53,289層の表で、画素0 4,977層 → 定義D 1,664層)。空の列の定義を器具に書いた(値は変えない。"
                    "描画できなかった15層は「測れない」)。便B の選択肢と機械の印を定義D にした(表は作者に出す前、行は不変)。定義D は作者の確認待ち",
    ),
    dict(
        path="inbox/note_census_order_grey_stats_20261111.md",
        what="**Track E → census: 抽出側への階調の統計の発注の起案**(小さい便1 → 大きい便2。ユーザーの承認あり)。対照の層の一覧を census が引く",
        why_not_yet="**保留(ユーザーの決定 11-11: 除外の track の登録を待つ)**。基盤の回送どおり、登録の後に census が対照の一覧を引いて出す",
    ),
    dict(
        path="inbox/reply_track_q_opened_20261111.md",
        what="**基盤: b(除外の判定)の track を立てた**(Track Q)。新しい画素の列の発注は Q の登録の後に census 経由",
        why_not_yet="**受けた。** a(テンプレート)の聞き取りの答えを Q と E に写した。**11-12 にユーザーが a の担当を census から E に改めた**(census は一覧・発注・列)",
    ),
    dict(
        path="inbox/note_census_a2_wording_and_empty_count_20261111.md",
        what="**Track E の訂正: 便A の空の層は97行に5**(census の「全体で1」は回収した16行の中の数で、述語が無かった)/ A-2 の一言の曝露は2行",
        why_not_yet="**受けた。** 器具の印字に母集団を書き、16行の中 1 / 97行の中 5 を両方出す(E と一致)",
    ),
    dict(
        path="inbox/deliver_occlusion_edge_stage2_crops_20261110.md",
        what="**抽出側 → 基盤の納品の写し(無効)**: 取り消し後に作られた第2段の切り出し。`note_occlusion_edge_stage2_crops_deleted_20261110.md` で全部消したと知らせがあった",
        why_not_yet="**census は写していない**(原稿の画素は census の木に無い)。作業なし",
    ),
    dict(
        path="inbox/note_p_order_layer_rows_300_20261107.md",
        what="**Track P の依頼: 300頁の層の行**(holdout8 と同じ64列。抽出側への発注 `push_p_order_300_pages_20261107.md` と名簿 CSV の写しつき)",
        why_not_yet="**構造の列を先に答えた**: 器具 `p_layer_rows.py`、全量は非公開。64列 + マスクの画素の数(供給元の表の値を写す。"
                    "**面の表 `>0/=255/1〜254`(キャンバス全体)と `mask_cover` `>=128/1〜127`(バッファの範囲)は区切りが違うので片方から作らない**)"
                    " + 第1段のカラーかモノクロか + 作業途中。"
                    "**11-10 に画素の列を入れた**: 抽出側の manifest(10便、sha256 照合)の値を写す。`black_among_drawn` / `drawn_gray_mean` / `preview_dark` は"
                    "manifest に無く空。既知セル M の872頁の表と重なる 6,412層で、manifest に無い空の層の描画色 451・形式 19 のほか値の違い 3層(読み手の版)",
    ),
    dict(
        path="inbox/reply_foundation_occlusion_edge_estimate_20261107.md",
        what="**抽出側 → 基盤の返事の写し**(遮蔽物の輪郭からの深さを受ける。見込み3〜4日)",
        why_not_yet="**宛先は基盤。** census の作業は無い",
    ),
    dict(
        path="inbox/note_m_order_layer_rows_holdout8_20261107.md",
        what="**Track M の発注: 標本外の8頁の層の行**(抽出側の 11-02 の層の表と同じ64列)",
        why_not_yet="**答えた**: 構造の19列を構造 DB から作り、画素の45列は空(画素は M が抽出側に発注済み)。"
                    "**既知セル: 抽出側の 872頁・72,839行と構造の19列が全行一致**。器具 `m_layer_rows.py`、全量は非公開",
    ),
    dict(
        path="inbox/push_m_order_holdout8_pages_20261107.md",
        what="**Track M → 抽出側の発注の写し**(8頁の層の画素とマスクのラスタ)",
        why_not_yet="**宛先は抽出側。** census の材料表の数を既知セルに使うと書いてある",
    ),
    dict(
        path="inbox/note_m_order_pages_with_mask_material_20261106.md",
        what="**Track M の発注: マスクの面の材料を持つ頁の一覧**(作者の承認「2と3を一緒にやる １を並行してだす」)",
        why_not_yet="**答えた**: `census/mask_material_pages.csv`(札ごと)/ 頁ごとの全量は非公開で M と基盤へ写し。"
                    "**材料 ≥ 3 は下限 1,394頁・上界 2,371頁・トーンを除いて 1,048頁**。既知セル(M の採用4頁)4/4 一致",
    ),
    dict(
        path="inbox/note_n_census_hamlabi_pages_20261107.md",
        what="**Track N: 2札の作品のタイルの13頁が、バイト同一の組に入るか**",
        why_not_yet="**答えた**: 13頁とも2札でバイト同一(違う4組には入らない)",
    ),
    dict(
        path="inbox/reply_census_wip_column_received_20261107.md",
        what="**Track L の受領: 作業途中の列を当てた** —— L の `done*` 25頁は 25頁とも `作業途中`、残した40頁は `裁定なし`",
        why_not_yet="**返信は要らない形。** L は現行の根拠を `裁定なし` ではなく裁定 R1 に置いている(census の但し書きどおり)",
    ),
    dict(
        path="inbox/note_n_census_a1_analog_source_20261106.md",
        what="**Track N: A1「アナログ主体」の出所は作者の 10-30 の言明**(作品単位・主体。段階の裁定ではない)",
        why_not_yet="**既に `author_rulings.csv` の1行目に在った**(10-30)。**census の前便の「裁定なし」は誤り** —— 段階の表だけを見た。"
                    "当たらなかった2作の答え(1作は構造 DB の外、1作は2札が同じ内容)も非公開の対応表に在った。"
                    "2札の作品はバイトでも同名 30組中26組が同一(11-06)",
    ),
    dict(
        path="inbox/reply_census_done_folder_answer_20261106.md",
        what="**抽出側の答え: `done*` についての作者の答えは、こちらのセッションに無い**(10-23 より前の会話は残っていない)",
        why_not_yet="**答えは基盤の作業記録に在った**(10-26 01:07 問い I / 01:23 問い L)。`wip_pages.py` の R2・R3",
    ),
    dict(
        path="inbox/note_l_ask_census_done_is_wip_20261106.md",
        what="**Track L の照会: 「途中のファイル」の列**(作者 11-06「done は途中のファイル … census に問い合わせる必要がある」)",
        why_not_yet="**列を作った**(`census/wip_pages.csv`、頁ごとは非公開の全量)。**作業途中 215頁・29フォルダー、裁定4件** —— "
                    "R1〜R3 は 10-26 に基盤経由で作者が答えていた(**問い L「全部同じ運用だ 共通して背景がない」は census の記録に無かった**)、"
                    "R4 は問いに出ていなかった8フォルダーへのユーザーの指示(11-06)。"
                    "**`page_versions.version_role` は代用しない**(872頁・途中の値が無い。規律13)",
    ),
    dict(
        path="inbox/note_n_census_process_period_overlap_20261106.md",
        what="**Track N の依頼: 工程と時期を分けたい**(A1 の5作の作品番号 / 同じ時期のアナログ・デジタルの候補 / 記入式の表)",
        why_not_yet="**問い1は答えた**: A1 の5作のうち作者の工程の裁定が在るのは1作だけ(札 2024b、アナログ/アナログ)。"
                    "**2作は裁定なし、1作は構造 DB に頁が無い、1作は母集団の外(札 2025)**。"
                    "**問い2・3(候補と表)は篩の列を決めてから**",
    ),
    dict(
        path="inbox/deliver_census_layer_bbox_and_face_alpha_20261104.md",
        what="**発注2件の納品**(線画の層の範囲 / 面の層のアルファ・マスクの値)。**宛先は census、写しは N・M・基盤**",
        why_not_yet="**統合した**: 材料は `LOCAL_CSV` の `l1_layer_bbox` / `face_layer_alpha_mask`、"
                    "公開表は `l1_layer_extent.csv` / `face_binary_alpha.csv`。"
                    "**N の問い(頁の座標とタイル)への答えは供給元のもので、census は照合していない**",
    ),
    dict(
        path="inbox/reply_census_sketch_set_counts_20261104.md",
        what="**下絵の層のフォルダーの全数の数え方の答え**: **census の数え方が正しい**"
             "(便の 7,013頁 / 46,327層 / 32,963 は除外の頁を含めた数。除くと 7,012 / 46,310 / 32,948)",
        why_not_yet="**数の表ではなく答え。** **こちらの公開表は前から除外の頁を除いた数で、動かない。** "
                    "**送られた CSV 3本に除外の頁は 0行**(先方の数え直し)",
    ),
    dict(
        path="inbox/deliver_occ_full_v4_20261103.md",
        what="**遮蔽の全数計測 `_v4`**(作者の規則: 下絵の層のフォルダーの中の層と `調整` を遮蔽物にしない。"
             "重みつきの被覆の列を足した。7,662頁・230,264層。除外の1頁を落とした)。**宛先は基盤で、census は写し**",
        why_not_yet="**遮蔽の採点は基盤の領分。** **sha256 は6本すべて便の先頭16桁と一致。** "
                    "**census が使ったのは `is_worksheet` の1列だけ** —— 構造 DB の祖先の鎖と `調整` の名前から引いた "
                    "32,857層と全層一致(`worksheet_folder_reach.csv` の既知セル)",
    ),
    dict(
        path="inbox/deliver_occ_full_v3_zlib_20261101.md",
        what="**遮蔽の全数計測 `_v3`**(zlib の頭の取り違えを直した読み手 `a1131bc` で、"
             "944頁を測り直した全頁の表。**巨大な CSV 6本もこちらの inbox に届いている**)。"
             "**宛先は基盤で、census は写し**",
        why_not_yet="**遮蔽の採点は基盤の領分**で、この集計は被覆を1列も持っていない。"
                    "**sha256 は7本すべて検証済み**(`occ_full_v3_20261101.sha256`)——"
                    "**読んでいないが、届いたものが壊れていないことは確かめた。** "
                    "**`_v2` の 27頁もこの 944頁に入っている**(先方の言明)",
    ),
    dict(
        path="inbox/reply_m_layer_mask_raster_20261101.md",
        what="**5頁のレイヤーマスクのラスタ 111枚と Q1〜Q9 の答え**"
             "(抽出側 → Track M。置き場は学習機の納品の木)。**census は写し**",
        why_not_yet="**宛先は Track M で、この集計はマスクのラスタを読まない**"
                    "(**原稿の絵を読まないのが条例10**)。"
                    "**census が取ったのは数と答えだけ** —— "
                    "**`init_color` はマスクごとに違う(111枚で 255 が 85 / 0 が 26)/ "
                    "この表の `PN2` の `mask_fill` 65枚のうち形が在るのは 19枚 / "
                    "`layer_type` の2の位についての先方の言明は、"
                    "こちらの全数の引き直しでは<ins>コマ枠フォルダー 8,734層で食い違う</ins>**"
                    "(`caveats.csv` の2つの行)。"
                    "**納品の木に 19 番目のフォルダーが増えたので、"
                    "`delivery.UNDECLARED_KNOWN` に名前で入れた**",
    ),
    dict(
        path="inbox/deliver_occ_full_v2_mono1bit_20261101.md",
        what="**遮蔽の全数計測 `_v2`**(1ビットの層を読めるようにして 2,777頁を測り直した。"
             "全頁の表で 7,663頁 / 230,383層)。**宛先は基盤で、census は写し**",
        why_not_yet="**遮蔽の採点は基盤の領分**で、この集計は被覆を1列も持っていない。"
                    "**巨大な CSV 本体はこちらの inbox に来ていない**(便と sha256 だけ)—— "
                    "**要るときに先方の `outbox` から取り寄せる。** "
                    "**census が受け取ったのは2つの数だけ** —— "
                    "**`empty` → `ok` が 6,538層、被覆が変わった層が約2万で全部増える向き"
                    "(減った層は 0)**。"
                    "**追補で分かった27頁は `_v2` でもまだ空である**"
                    "(`occ_v2_still_unread` の行)",
    ),
    dict(
        path="inbox/zlib_1bit_sample_planes_20261101.csv",
        what="**抜き取りで1ビットが見えた (頁, 層, 面)** 7,270行"
             "(本体 6,587面・2,804頁 / マスク 683面・192頁)",
        why_not_yet="**各面の先頭3ブロックの抜き取りで、全ブロックではない**"
                    "(先方が明記)—— **だから「1ビットの層の全数」として読めない。** "
                    "**この集計が取ったのは数2つだけ**(`pixel_format` の行): "
                    "**本体 6,587面は、モノクロ一覧の 6,506層と走査で漏れた 81層の和に"
                    "ちょうど一致した / モノクロ以外の層に1ビットは見つからなかった**",
    ),
    dict(
        path="inbox/note_m_propose_track_pattern_brush_20261101.md",
        what="**新 track「パターンブラシ」の起案**(Track M → 基盤。作者の裁定が根拠)",
        why_not_yet="**起案は基盤の領分**(新しい track を生やすのは `foundation`)。"
                    "**census が取ったのは但し書きだけ** —— "
                    "**線の出自は4つで、④ パターンブラシは同定する列が無く、"
                    "作者の裁定で凍結**(`caveats.csv` の行)。"
                    "**同定の列は持たない**(規律13)",
    ),
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
        path="inbox/deliver_add1bit_layers_20261031.md",
        what="**1ビットの層を空として読んでいたための書き足し納品**"
             "(全層 priority65 に 149層・62頁 / カラー層に 60層)。"
             "**宛先は基盤と Track M で、census は写し**",
        why_not_yet="**だが納品の木を歩く表は数が動いた** —— "
                    "`occluder_balloon_two_sided.csv` の `undecidable` が **307 → 456**"
                    "(+149)。**判定できない層が増えたのであって、判定が増えたのではない**"
                    "(書き足し納品に素性の列が無い)。"
                    "**sha256 はこちらで引き直した** —— 150件 / 91件すべて一致。"
                    "**census はこの納品を発注していない**(「便が無いまま成果物ができる」形)",
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
        source="`note_m_glossary_entries_pattern_brush_and_tone_20261101.md`"
               " + `note_m_propose_track_pattern_brush_20261101.md`",
        supplier="作者 —— Track M 経由",
        as_of="2026-11-01",
        n_rows=2,
        note='**パターンブラシの裁定と、トーンレイヤの構成の説明** → `caveats.csv` の2行。'
             '**どちらも「track に限定しないナレッジ」だと作者が判断した**(逐語: '
             '「**これはトラックに限定しないナレッジなのでcensus か抽出だと思うが、'
             'どちらがいいだろうか**」→「**用語集に置くのが良さそうだ**」)。'
             '**用語集は基盤の文書**なので追記は基盤が行う —— '
             '**census が持つのは但し書きの側**: **④ パターンブラシは同定する列が無く'
             '(名前でも構造でも画素でも引けない)、作者の裁定で凍結** / '
             '**`tone_density_pct` から役(ベース / 陰影)を引く列は信頼できない** '
             '——「**このルールは現実の運用ではあまり守られていないことが多かった**」',
    ),
    dict(
        source="`note_n_*_20261030.md`(3通)",
        supplier="作者 —— Track N 経由",
        as_of="2026-10-30",
        n_rows=4,
        note='**作者の裁定3件と、この表についての但し書き1件** → `author_rulings.csv`。**census は作者に直接聞いていない**(Track N 経由)。**作者の言葉をそのまま引く。** **対象の名前は公開物に書かない** —— 対応は `data/census_full/author_rulings_subjects.csv`(`.gitignore` 配下)。**5つのうち1つは構造 DB の外に在り、残り4つも候補が1対1に決まっていない**',
    ),
    dict(
        source="`note_n_author_stage_rulings_20261031.md`",
        supplier="作者 —— Track N 経由(記入式の表)",
        as_of="2026-10-31",
        n_rows=11,
        note='**制作の方式の裁定11組** → `author_process.csv`。**単位は(年の札, 作品フォルダー)で、頁ではない。** **下絵と線画を別の列で聞いた** —— **下絵は11組すべてアナログで分散がゼロ**なので、**1列に潰すと2段が同じ札になる。** **時期の効果として読めない**(アナログ線画の3組はすべて 2024 系の札に固まっている)。**`device_note` の「未記載」は「iPad ではない」ではない**(作者は4組にだけ書いた)。**仕上げの列は無い** —— 「在れば備考に書く」約束だったが、どの行にも書かれていない',
    ),
    dict(
        source="`note_m_census_caveat_column_and_5_pages_20261031.md` + `note_m_census_caveat_is_permanent_20261101.md`",
        supplier="作者 —— Track M 経由(**発注は作者の直接の指示**)",
        as_of="2026-10-31",
        n_rows=5,
        note='**頁の但し書き5件** → `page_notes.csv`。**作者の指示**: 「**今回外した５点について、国勢調査にそれぞれの作品(頁)行に但し書きをつけるようにしてほしい(但し書きカラムがなければ追加するようにも依頼) セッションの記憶では今のような作品の事情はいずれ忘れてしまう**」。**翌日、一度きりの欄ではないと作者が決めた**: 「**そういった作品は見つけ次第但し書きで判断できるようにしていこう**」—— **常設の仕組みである。** **5件のうち3件は作者の裁定、2件は Track M の測定。** **内容はここで引き直していない**(条例5・条例6)。**頁のパスは `data/census_full/page_notes_subjects.csv`**',
    ),
]
