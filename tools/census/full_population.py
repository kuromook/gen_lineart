"""全数 437,561層についての表。**供給元が測った。ここでは測れない**(DB が手元に無い)。

出所: `inbox/reply_layer_type_bits_20261026.md`(抽出側、2026-10-26)。
母集団は構造 DB の全数 —— **9,653頁 = `.clip` 7,667 + `.lip` 1,986 / 437,561層**。
`.clip` と `.lip` で値の種類と振る舞いは同じ(供給元の明記)。

**固有名を含まない。** 層の名前(`正面`・`差分マスク` など)は**書かない** ——
段 A は層名を出さない。数と性質だけを持つ。
"""

SUPPLIER = "抽出側(um350)"
AS_OF = "2026-10-26"
N_LAYERS = 437561
N_PAGES = 9653

# ビットごとの意味。**不明は不明のまま持つ**(抽出側は「特殊な層と呼びたくなるが
# 確かめていない」と書いた)
BITS = [
    dict(bit=1, meaning="本体がラスタ(画素を描く層)", confidence="確か(全数で例外0)",
         evidence="1 / 3 は `v_layer_kind` が raster(ベクターの線を持つ237層を含む)。"
                  "0 / 2 はテキスト・画像素材・グラデーション・3D・塗り",
         not_answers="**画素の有無。** 「ラスタ」は名前の付け方である(抽出側の明記)"),
    dict(bit=2, meaning="層のマスクがある(`LayerLayerMaskMipmap`)",
         confidence="確か。**ただし1種類だけ外れる**",
         evidence="フォルダーも含めて一致する",
         not_answers="**コマ枠フォルダー 8,734個。** マスクがあっても 2 が立たない —— "
                     "例外ではなく**別の種類**。"
                     "**ただしマスクの向きは種類からは決まらない** —— "
                     "行ごとに `init` を読むこと(2026-10-26 に訂正)"),
    dict(bit=16, meaning="不明", confidence="不明",
         evidence="用紙(1584)にだけ立つ", not_answers="**1ビット単独の意味**"),
    dict(bit=32, meaning="不明", confidence="不明",
         evidence="ノンブル・作品情報(800)と用紙(1584)に立つ",
         not_answers="**1ビット単独の意味**"),
    dict(bit=256, meaning="不明", confidence="不明",
         evidence="根フォルダー(256)とノンブル・作品情報(800)に立つ",
         not_answers="**1ビット単独の意味**"),
    dict(bit=512, meaning="不明", confidence="不明",
         evidence="選択範囲(513)・ノンブル・作品情報(800)・用紙(1584)に立つ",
         not_answers="**1ビット単独の意味**"),
    dict(bit=1024, meaning="不明", confidence="不明",
         evidence="用紙(1584)にだけ立つ", not_answers="**1ビット単独の意味**"),
    dict(bit=4096, meaning="色調補正レイヤー(`FilterLayerInfo`)",
         confidence="確か(全数で例外0)",
         evidence="4096 / 4098 は全部 `has_filter` 真。他の値に `has_filter` 真は無い",
         not_answers="補正の中身"),
]

# 出てくる値は**この11種だけ**(上の位の組は固定で、自由に組まれていない)
VALUES = [
    dict(layer_type=0, bits="—", is_folder="no", n_layers=27283,
         content="テキスト 19,912 / 画像素材 5,538 / グラデーション 1,309 / "
                 "ベクター 379 / 3D 135 / その他 10"),
    dict(layer_type=0, bits="—", is_folder="yes", n_layers=97475,
         content="普通のフォルダー 88,741 / **コマ枠フォルダー 8,734**"),
    dict(layer_type=1, bits="1", is_folder="no", n_layers=158897,
         content="ラスタ(ベクターの線を持つ 231 を含む)"),
    dict(layer_type=2, bits="2", is_folder="no", n_layers=119619,
         content="塗り(トーン・ベタ)118,370 / 画像素材 799 / ベクター 437 / テキスト 13"),
    dict(layer_type=2, bits="2", is_folder="yes", n_layers=1741,
         content="マスクを持つ普通のフォルダー"),
    dict(layer_type=3, bits="1+2", is_folder="no", n_layers=3957,
         content="マスク付きのラスタ"),
    dict(layer_type=256, bits="256", is_folder="yes", n_layers=9649,
         content="根フォルダー。**1頁に必ず1個**(読めた頁 9,649)"),
    dict(layer_type=513, bits="512+1", is_folder="no", n_layers=93,
         content="選択範囲レイヤー"),
    dict(layer_type=800, bits="512+256+32", is_folder="no", n_layers=8097,
         content="ノンブル 6,633 / 作品情報 1,464"),
    dict(layer_type=1584, bits="1024+512+32+16", is_folder="no", n_layers=9625,
         content="用紙。1頁に1個"),
    dict(layer_type=4096, bits="4096", is_folder="no", n_layers=12,
         content="色調補正(マスクなし)"),
    dict(layer_type=4098, bits="4096+2", is_folder="no", n_layers=1113,
         content="色調補正(マスクあり)"),
]

# 2の位とマスクの全数(`layers` 全行)。**コマ枠は別の行**(混ぜると例外に見える)
MASK_BY_KIND = [
    dict(is_folder="no", has_frame_line="no", layer_type_and_2=0,
         has_mask="no", n_layers=204007),
    dict(is_folder="no", has_frame_line="no", layer_type_and_2=2,
         has_mask="yes", n_layers=124689),
    dict(is_folder="yes", has_frame_line="no", layer_type_and_2=0,
         has_mask="no", n_layers=98390),
    dict(is_folder="yes", has_frame_line="no", layer_type_and_2=2,
         has_mask="yes", n_layers=1741),
    dict(is_folder="yes", has_frame_line="yes", layer_type_and_2=0,
         has_mask="yes", n_layers=8734),
]

# `visibility` の2の位とマスクが食い違う層。**母集団ごとに別の数。どちらも持つ**
# (供給元は一度「置き換えてください」と書き、その後**取り消した**)
VISIBILITY_EXCEPTIONS = [
    dict(population="db_all_layers", direction="マスクがあるのに `visibility & 2` が立たない",
         n_layers=119, detail="フォルダー以外 85 / フォルダー 34",
         measured_by=SUPPLIER, as_of=AS_OF),
    dict(population="priority65", direction="`layer_type & 2` と `visibility & 2` が食い違うフォルダー",
         n_layers=245, detail="**全部コマ枠フォルダー**(名前で2群に分かれるが第3の種類ではない)",
         measured_by="抽出側 / Track L", as_of=AS_OF),
    dict(population="db_all_layers", direction="`layer_type & 2` だけ立つフォルダー",
         n_layers=34, detail="**作者が付けた層名は書かない**(段 A は層名を出さない)",
         measured_by=SUPPLIER, as_of=AS_OF),
]

# 本体の `InitColor` の納品(`body_initcolor_delivered_20261026.csv`、1層1行)。
# **これはここで測れる** —— `set` 列に作品名が入るので**段 A では畳む**
SET_LABELS = {
    "tone_v2": "トーン",
    "grad_v2": "グラデーション",
}
FILL_PREFIX = "fill_"
FILL_LABEL = "ベタ"
