"""遮蔽物の層(occluder)。**手元に層名が在る頁だけ。**

**なぜ要るか**(作者の言明、2026-10-26):

> **手前側に何かがあると、その後ろ側のレイヤでは描かないで済ますことが多い。**
> **しかし、遮蔽物レイヤがあるということがわからないと、
> なぜ下のレイヤで何かが途切れているかわからないということになる。**

**下の層の線が途切れているのは、描き手が省いたからで、欠陥ではない。**
**遮蔽物が分からないと、その途切れを「欠落」と読む。**

**作者が挙げた4種**: 3D / フキダシ(独立したフキダシレイヤ)/ ホワイト(`white` という名が多い、
との言明)/ 黒消し。

**作者の答(2026-10-27)**: **「ラスタライズする。印については分からない。」**

> **だから 3D の層数は「3D を使った頁の数」ではなく、その下限である。**
> **ラスタライズされた 3D は、種別では `raster` と見分けが付かない。**

**そしてこれは判定の形を変える**: **ラスタライズされても、その層は遮蔽物のままである。**
**名前と種別で見つかるのは遮蔽物の一部だけ** —— 一般の判定は
**「不透明な層が、上の重なりに在って、重なっているか」**で、
**名前ではなく順序と不透明度と重なり**で決まる(依頼8 がその材料)。

**印が残るかは作者も知らない** —— **「残らない」ではなく「不明」。**
**形式を知っているのは抽出側なので、そちらへ回す問いである。**

**この表が答えない**:
- **全数の傾向。** 層名が手元に在るのは納品の manifest が覆う頁だけ(ごく一部)
- **その層が何を隠しているか。** 名前だけでは分からない —— **重なり順が要る**
  (`parent_id` / `order_in_parent` / `ancestors`。依頼8でまとめて頼む)
- **語彙が網羅的か。** 名前の候補は**作者が選ぶ**(規律6)。
  **`white` は作者自身が「多いと思う」と言っている** —— 推測である

**層名は作者の作業名なので `census/` には出さない。** 公開するのは**種別ごとの件数**だけ。
"""
import collections
import glob
import json
import os
import re

ROOT_DEFAULT = "/home/sh1/deepl/lineart/data/incoming_manuscript"

# **3D は経路が2つある**(作者の言明、2026-10-27):
#
#   経路1  クリスタの 3D オブジェクト → **レンダリングして画像化して確定する。**
#          **印が残るかは不明。ファイル名が一つのサインになるかもしれない**(作者)
#   経路2  blender でレンダリングした画像のインポート → **既に画像化されている。**
#          **自動処理で取り込むためフォルダ名に `rendering` 等がつく**(作者)。
#          **ファイル名のパターンも決まっており、データも特徴的**
#
# **手元の15頁で経路2が見つかった** —— 層名 `rendering_base_middle0001` /
# `rendering_shadow_middle0001` / `rendering_lineart_middle0001` /
# `rendering_render_middle0001 3`、**全部 `layer_type = 1`(ラスタ)でフォルダー `/bg` の中。**
# **つまり種別では普通のラスタと見分けが付かず、名前でしか分からない。**
# **そして名前に出たのは層名で、フォルダー名ではなかった** —— **両方見る。**
#
# 経路1 の候補として、**既定の時刻名**(`2018年11月29日22時47分52秒0003`、
# `2018-12-11 00_32_39-CLIP STUDIO PAINT`)が在った。**ただしこれは取り込んだ画像一般の
# 既定名でもありうる** —— 手元の9件は `/参考` や `/下絵` の中で、**3D と断定できない。**
# **候補として数え、断定しない。**

# 遮蔽物の候補。**名前の規則は作者に未確認**(`white` は作者の「多いと思う」から)
CLASSES = (
    # **3D は下限** —— 作者が「ラスタライズする」と答えた(2026-10-27)。
    # ラスタライズされた 3D は種別が `raster` になり、この数に入らない
    ("3d_kind", "3D(種別で分かる。**使った頁の下限**)", None, "3d"),
    ("white", "ホワイト(白で隠す)", r"white|ホワイト|ﾎﾜｲﾄ", None),
    ("balloon", "フキダシ(独立した層)", r"フキダシ|ふきだし|吹き出し|バルーン|balloon", None),
    ("black_erase", "黒消し", r"黒消|黒ケシ|くろけし|黒けし", None),
    # **経路2(blender)**: 名前の規則で分かる。**種別はラスタなので種別では出ない**
    # **区切りを要求していたので 211層・13名を落としていた**(`rendering0001` の形)。
    # **供給元の素の `rendering` が 5,383 で、こちらが 5,172 だったので見つかった** ——
    # **供給元の数と自分の数を比べることが、ここでは点検になった。**
    # 直した後の 5,383 は先方の数と一致するが、**それは独立な確認ではなく一致である**(T132)
    ("blender_render", "blender の取り込み(**経路2**。名前の規則で分かる)",
     r"rendering|blender", None),
    # **経路1 の候補**: クリスタの既定の時刻名。**取り込んだ画像一般の既定名でもありうる** ——
    # **3D と断定しない。候補として数える**
    ("default_timestamp_name",
     "既定の時刻名(**経路1 の候補。断定しない**)",
     r"^\d{4}年\d{1,2}月\d{1,2}日\d{1,2}時\d{1,2}分\d{1,2}秒|"
     r"^\d{4}-\d{2}-\d{2} \d{2}[_:]\d{2}[_:]\d{2}", None),
)


def load_named_layers(root: str, drops):
    """納品の manifest から、**層名が在る頁**の層を集める。

    **落としたものを理由別に数える**(規律9)—— manifest の形が揃っていないので、
    名前の無い層・頁の鍵が無いものを黙って捨てない。
    """
    pages = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            drops.drop(f"manifest が読めない: {type(e).__name__}", path)
            continue
        if not isinstance(d, dict):
            drops.drop("manifest の最上位が dict でない", path)
            continue
        if isinstance(d.get("pages"), list):
            for p in d["pages"]:
                if isinstance(p, dict) and isinstance(p.get("layers"), list):
                    key = p.get("page") or p.get("source")
                    if not key:
                        drops.drop("頁の鍵(`page`/`source`)が無い", path)
                        continue
                    pages[key] += p["layers"]
                else:
                    drops.flag("`layers` を持たない頁の行", path)
        elif isinstance(d.get("layers"), list):
            pages[d.get("page") or os.path.basename(path)] += d["layers"]
        else:
            drops.flag("層の表を持たない manifest(別の用途)", path)
    return pages


def classify(pages, drops):
    """頁ごとに、どの遮蔽物の種別が在るか。**名前は返さない**(件数だけ)。"""
    per_class = collections.Counter()
    per_class_pages = collections.defaultdict(set)
    n_layers = n_named = 0
    pats = {k: (re.compile(p, re.I) if p else None) for k, _, p, _ in CLASSES}
    kinds = {k: v for k, _, _, v in CLASSES}
    for page, layers in pages.items():
        for L in layers:
            n_layers += 1
            name = (L.get("name") or "").strip()
            if name:
                n_named += 1
            else:
                drops.flag("層名が空(種別でしか見られない)")
            kind = (L.get("v_layer_kind") or "").strip().lower()
            # **層名とフォルダーの鎖の両方を見る** —— 作者は「フォルダ名に `rendering`」と
            # 述べたが、手元の15頁では**層名に出ていた。** 片方だけ見ると落ちる
            anc = str(L.get("ancestors") or "")
            for key, _, _, want_kind in CLASSES:
                hit = False
                if want_kind is not None:
                    hit = kind == want_kind
                elif pats[key] is not None:
                    hit = bool(name and pats[key].search(name)) or \
                          bool(anc and pats[key].search(anc))
                if hit:
                    per_class[key] += 1
                    per_class_pages[key].add(page)
    rows = []
    for key, label, pat, want_kind in CLASSES:
        rows.append([key, label,
                     "名前の規則(層名とフォルダーの鎖)" if want_kind is None else "種別",
                     per_class[key], len(per_class_pages[key])])
    return rows, len(pages), n_layers, n_named


# **層の見極めは片面だけで済ませない**(作者の指示、2026-10-27):
#
# > **名前で見つけて統計量で裏付ける・統計量で見つけて名前で裏付ける。**
# > **どちらの順でもいいが、片面だけで済ませない方がよい。**
#
# **これは 3D に限らない規則なので、表の構造に入れる** —— どの種別にも
# **名前側の数 / 統計量側の数 / 裏付けの状態**が付く。
#
# やり方: 名前で当たった層の**素性**(どの列がどの値だったか)を取り、
# **同じ素性を持つのに名前が当たらない層**を数える。
# **その数が大きければ、名前が取りこぼしている。**

# **2026-10-27 の測定結果: この素性は何も特徴づけていない。**
# `white` の名前側 10層に対し、**同じ素性の層が 824**(全1,013層中)——
# つまりこの列の組は「普通の層」と言っているだけ。
#
# **素性を、名前に合うまで調整しない** —— それは名前に素性を当てはめる作業で、
# **両面の裏付けではなく、片面を2回数えることになる。**
# **失敗を出したまま残し、何が要るかを書く。**
#
# **統計量側に要るのは画素の分布の列**(作者: 「データも特徴的なので分布でわかる特性が
# あると思われる」)—— 納品の manifest には `drawn_gray_mean` / `black_among_drawn` /
# `colours_q16` / `inside_ratio` が在る。**レンダリングは中間調を持ち、手描きの線画は
# 二値に近い**はずだが、**15頁では閾も帰無値も立てられない。未測定。**
# **そして `drawn_gray_mean` / `black_among_drawn` は下限になりうる列**(T60 拡張)なので、
# 使う前に `body_init` と保存の状態が要る(依頼3の3列)。
SIGNATURE_FIELDS = ("layer_type", "composite", "clip", "has_mask")


def signature(layer) -> tuple:
    """素性。**値そのものではなく、宣言した列の組**(列を増やすと素性が細くなる)。"""
    return tuple(str(layer.get(f)) for f in SIGNATURE_FIELDS)


def two_sided(pages, drops):
    """**両面の裏付け。** 名前側と統計量側を、互いの逆から数える。

    **この表が答えない**:
    - **素性が遮蔽物を特徴づけているか。** `SIGNATURE_FIELDS` はこちらが選んだ列で、
      **作者も抽出側も確かめていない**
    - **全数。** 層名が手元に在る頁だけ
    - **素性が一致することが同じ種類であること** —— 一致は「この列の組では分けられない」
      であって「同じもの」ではない
    """
    import collections as C
    pats = {k: (re.compile(p, re.I) if p else None) for k, _, p, _ in CLASSES}
    rows = []
    for key, label, pat, want_kind in CLASSES:
        named, sigs = [], C.Counter()
        for layers in pages.values():
            for L in layers:
                name = (L.get("name") or "").strip()
                anc = str(L.get("ancestors") or "")
                kind = (L.get("v_layer_kind") or "").strip().lower()
                hit = (kind == want_kind) if want_kind is not None else (
                    pats[key] is not None
                    and (bool(name and pats[key].search(name))
                         or bool(anc and pats[key].search(anc))))
                if hit:
                    named.append(L)
                    sigs[signature(L)] += 1
        if not named:
            rows.append([key, label, 0, 0, 0, "名前側が0件(裏付けられない)"])
            continue
        # 名前側で当たった素性を持つのに、名前が当たらない層
        unnamed = 0
        for layers in pages.values():
            for L in layers:
                if L in named:
                    continue
                if signature(L) in sigs:
                    unnamed += 1
        if unnamed == 0:
            state = "**両面一致**(この素性を持つ層はすべて名前も当たる)"
        elif unnamed <= len(named):
            state = f"**片面** 素性だけの層 {unnamed}(名前側と同じ桁)"
        else:
            state = (f"**名前が取りこぼしている可能性** 素性だけの層 {unnamed} "
                     f"対 名前側 {len(named)}")
        rows.append([key, label, len(named), len(sigs), unnamed, state])
    return rows


# ----------------------------------------------------------------- 仕組みの素性
#
# **上の `SIGNATURE_FIELDS` は失敗した** —— `layer_type`/`composite`/`clip`/`has_mask` は
# 「普通の層」と言っているだけで、何も特徴づけていない(`white` の名前側10層に対し
# 同じ素性が824層)。**失敗は上に残してある。**
#
# **ここはその代わりではなく、別の側である** —— 作者の言明:
#
# > **データも特徴的なので分布でわかる特性があるとおもわれる**(2026-10-27、3D について)
#
# **だから素性は画素の分布から取り、節ごとに「その層が物として何であるか」から導く。**
# **節の理由を書けない節は入れない** —— 理由の無い節は、名前に合わせて足した節と
# 区別が付かない(それが「片面を2回数える」ことになる)。
#
# **フキダシは何か**: **不透明な白い面と、その輪郭。** 下を隠すのはその白い面である。
#   → 白が最頻 / 無彩色 / 色の段が2つ(白と輪郭) / 最頻色の占有が1未満(輪郭が在る)
#   → 面は頁の一部(コマの中)
#
# **ホワイトは何か**: **白で消すだけの面。輪郭を持たない。**
#   → 白が最頻 / 無彩色 / 色の段が1つ / 最頻色の占有が1(輪郭が無い)
#
# **この2つは同じ道具ではなく、同じ列で分かれる** —— **分けているのは占有の節である。**
#
# **見つけた順を列に書く**(作者の規則はどちらの順でもよいが、**順は記録する**):
#   フキダシ = 名前で見つけて統計量で裏付けた
#   ホワイト = **フキダシの試験の近傍を見て統計量側から見つけ、名前で裏付けた**
#              (`white`・`断ち落とし枠(白)`・`bg base`・`high light` が同じ素性に並んだ)

def _q16_white(L):
    t = L.get("top_colour_q16")
    return None if t is None else tuple(t) == (15, 15, 15)


def _colours_le(n):
    def f(L):
        c = L.get("colours_q16")
        return None if c is None else c <= n
    return f


def _colours_eq(n):
    def f(L):
        c = L.get("colours_q16")
        return None if c is None else c == n
    return f


def _achromatic(L):
    s = L.get("sat_mean")
    return None if s is None else s == 0.0


def _share_band(lo, hi):
    def f(L):
        s = L.get("top_colour_share")
        return None if s is None else lo < s < hi
    return f


def _share_full(L):
    s = L.get("top_colour_share")
    return None if s is None else s >= 1.0


def _drawn_lt(x):
    def f(L):
        d = L.get("drawn_ratio")
        return None if d is None else d < x
    return f


# 節: (符号, 何を見るか, **なぜその節が在るか**(物の側の理由), 判定)
MECHANISM = {
    "balloon": (
        "フキダシ(**不透明な白い面と輪郭**)",
        "名前で見つけて統計量で裏付けた",
        (("white_top", "白が最頻",
          "下を隠すのは白い面なので、最も多い色は白である", _q16_white),
         ("colours_le2", "色の段が2以下",
          "面の白と輪郭の黒で、段は2つになる", _colours_le(2)),
         ("achromatic", "彩度の平均が0",
          "白と黒だけなら無彩色である", _achromatic),
         ("share_band", "最頻色の占有が 0.5〜0.99",
          "**輪郭が在るので白は全部ではない。** これがホワイトと分ける節",
          _share_band(0.5, 0.99)),
         ("local", "描画率が 0.25 未満",
          "フキダシはコマの中の一部で、頁を覆わない", _drawn_lt(0.25))),
    ),
    "white": (
        "ホワイト(**白で消す面。輪郭を持たない**)",
        "**統計量側から見つけて名前で裏付けた**(フキダシの試験の近傍)",
        (("white_top", "白が最頻",
          "白で消す面なので最頻色は白である", _q16_white),
         ("colours_eq1", "色の段が1つ",
          "**輪郭を持たないので段は1つ。** これがフキダシと分ける節", _colours_eq(1)),
         ("achromatic", "彩度の平均が0", "白だけなら無彩色である", _achromatic),
         ("share_full", "最頻色の占有が 1.0",
          "面が白一色であること。輪郭の黒が無い", _share_full)),
    ),
}

PROBE_PREFIX = "drop_"


def _name_hit(L, pat):
    name = (L.get("name") or "").strip()
    anc = str(L.get("ancestors") or "")
    return bool(name and pat.search(name)) or bool(anc and pat.search(anc))


def mechanism_two_sided(pages, drops):
    """**仕組みから導いた素性で、両面を数える。**

    返す行は1つの問いずつ:
      `name_side`     名前の規則で当たった層
      `feature_all`   全節を満たす層 ← **両面の判定はここ**
      `drop_<節>`     その節だけを外したときに増える層 = **その節が落としているもの**
      `undecidable`   素性の列が無くて判定できない層
      `control_<他>`  **もう一方の類の名前側が、この素性に入るか**(入ったら分けていない)

    **この表が答えない**:
    - **全数。** 素性の列が在るのは納品の manifest が覆う頁の一部
    - **閾の正しさ。** 節の**向き**は物の側から導いたが、
      **`0.5` と `0.25` の値はこちらが置いた数で、どちらも何も落としていない**
      (下の `drop_` の行がそれを印字する)
    - **一致が同じものであること。** 一致は「この列では分けられない」であって同一ではない
    """
    pats = {k: re.compile(p, re.I) for k, _, p, _ in CLASSES if p}
    layers = [(pg, L) for pg, ls in pages.items() for L in ls]
    rows = []
    for key, (label, order, clauses) in MECHANISM.items():
        pat = pats[key]
        name_side = {id(L) for _, L in layers if _name_hit(L, pat)}
        decided, undecidable = [], 0
        for pg, L in layers:
            vals = [c[3](L) for c in clauses]
            if any(v is None for v in vals):
                undecidable += 1
                continue
            decided.append((pg, L, vals))
        feat = {id(L) for _, L, v in decided if all(v)}
        feat_pages = {pg for pg, L, v in decided if all(v)}
        name_pages = {pg for pg, L in layers if id(L) in name_side}
        both = len(name_side & feat)
        f_only = len(feat - name_side)
        n_only = len(name_side - feat)
        rows.append([key, label, "name_side", f"名前の規則で当たった層({order})",
                     len(name_side), len(name_pages), both, "", n_only,
                     "**名前側が0件**" if not name_side else
                     f"うち素性も当たる {both} / 素性が当たらない {n_only}"])
        if not feat and not name_side:
            continue
        if f_only == 0 and n_only == 0:
            state = "**両面一致**(この頁の範囲では、名前側と素性側が同じ集合)"
        elif f_only > len(name_side):
            state = (f"**名前が取りこぼしている可能性** 素性だけの層 {f_only} "
                     f"対 名前側 {len(name_side)}")
        else:
            state = f"素性だけの層 {f_only} / 名前だけの層 {n_only}"
        rows.append([key, label, "feature_all", "全節を満たす層(**両面の判定はここ**)",
                     len(feat), len(feat_pages), both, f_only, n_only, state])
        # **どの節が働いているか。** 1つ外して増えた分が、その節が落としているもの
        for i, (code, what, why, _fn) in enumerate(clauses):
            sel = [(pg, L) for pg, L, v in decided
                   if all(c for j, c in enumerate(v) if j != i)]
            got = {id(L) for _, L in sel}
            added = len(got - feat)
            rows.append([key, label, PROBE_PREFIX + code,
                         f"「{what}」を外す —— {why}",
                         len(sel), len({pg for pg, _ in sel}),
                         len(got & name_side), len(got - name_side),
                         len(name_side - got),
                         f"**この節が落としている層 {added}**" if added else
                         "**この節は何も落としていない**(働いていない)"])
        rows.append([key, label, "undecidable", "素性の列が無くて判定できない層",
                     undecidable, "", "", "", "",
                     "**判定できない層は、素性側の母集団に入っていない**"])
    # 負の対照: 一方の名前側が、他方の素性に入るか
    for key, (label, _order, clauses) in MECHANISM.items():
        for other in MECHANISM:
            if other == key:
                continue
            pat = pats[other]
            n = 0
            for pg, L in layers:
                if not _name_hit(L, pat):
                    continue
                vals = [c[3](L) for c in clauses]
                if any(v is None for v in vals):
                    continue
                if all(vals):
                    n += 1
            rows.append([key, label, "control_" + other,
                         f"**負の対照** —— {other} の名前側が、この素性に入るか",
                         n, "", "", "", "",
                         "**0。この2類は素性で分かれている**" if n == 0 else
                         f"**{n} 層が入った。この素性は2類を分けていない**"])
    return rows


def probe_tokens():
    """`probe` 列の白名簿。**器具の宣言から作る**(手で並べると片方が古くなる)。"""
    out = {"probe", "name_side", "feature_all", "undecidable"}
    for key, (_l, _o, clauses) in MECHANISM.items():
        out |= {PROBE_PREFIX + c[0] for c in clauses}
        out |= {"control_" + k for k in MECHANISM if k != key}
    return out


# ------------------------------------------------- 全数の層名(依頼7、2026-10-27 着)
#
# **名前側はこれで全数に届いた** —— 437,561層・16,241 の異なる名前。
# **素性側は届いていない**(画素の分布の列が在るのは納品の manifest の8頁だけ)。
# **だから両面が一致したと言える範囲は8頁で、名前側の数は全数である。**
# **この2つを同じ行に並べない。**
#
# **供給元の言明**(添え状 §6):
# - **フォルダーの行も入っている**(`is_folder`)。**`黒消し` はフォルダーで 609、層は 0** ——
#   **黒消しは層ではなくフォルダーだった。** 遮蔽しているのはその中の層で、
#   **どの層かは `ancestors` が要る**(依頼8)
# - `ancestors` は層名ごとに1つに決まらないので入っていない。**親は `top_parent_name`
#   (最頻の直接の親)と `n_parent_names`(親の名前の異なり)だけ**
# - **`layer_name` が空の 9,652層**は頁の最上位フォルダー 9,649 と名無しの `other` 3
#
# **`top_parent_name` は作者の作業名なので `census/` には出さない。** 公開するのは
# **`n_parent_names`(異なり数)だけ。**
R7_DEFAULT = os.path.join("inbox", "census_r7_layer_names_20261027.csv")


def load_vocab_table(path: str, drops):
    """依頼7の層名の表。**行は読むが、名前は返さない関数を通して使う。**"""
    import csv
    if not os.path.exists(path):
        drops.flag("全数の層名の表が無い(依頼7が未着)", path)
        return []
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    need = {"layer_name", "is_folder", "n_layers", "n_pages",
            "first_year_label", "last_year_label", "kinds", "n_parent_names"}
    missing = need - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"依頼7の表に宣言した列が無い: {sorted(missing)}")
    return rows


def _is_folder(row) -> bool:
    return str(row.get("is_folder", "")).strip().lower() not in ("0", "false", "")


def vocab_full(rows, drops):
    """**全数の名前側。** 語彙に当たる名前を、層とフォルダーに分けて数える。

    **この表が答えない**:
    - **頁は足せない。** 同じ頁に2つの名前が在りうるので、名前ごとの `n_pages` の和は
      頁数ではない(T96 の形)。**だから最大の名前の頁数だけを書き、和は書かない**
    - **語彙が正しいか。** **どの名前が遮蔽物かは作者が選ぶ**(規律6)。
      ここに在るのは**こちらが書いた正規表現が当たった数**である
    - **素性側の裏付け。** 全数には画素の分布の列が無い —— **8頁でしか両面を見ていない**
    - **名前が物を同定するか。** 語彙は台詞の中の語にも当たる
      (`text` 種別の層が1つ入った)。**種別の内訳を同じ行に書いて、それを見せる**
    """
    out = []
    for key, label, pat, want_kind in CLASSES:
        if pat is None:
            out.append([key, label, "", "", "", "", "", "", "",
                        "**種別で見つける類なので、名前の表には出ない**"])
            continue
        p = re.compile(pat, re.I)
        hit = [r for r in rows if p.search(r["layer_name"])]
        layers = [r for r in hit if not _is_folder(r)]
        folders = [r for r in hit if _is_folder(r)]
        n_layers = sum(int(r["n_layers"]) for r in layers)
        n_folders = sum(int(r["n_layers"]) for r in folders)
        kinds = collections.Counter()
        for r in layers:
            for part in r["kinds"].split(";"):
                if ":" in part:
                    k, v = part.rsplit(":", 1)
                    kinds[k] += int(v)
        top = max(layers, key=lambda r: int(r["n_layers"]), default=None)
        if top is None:
            top_n_pages = top_first = top_last = top_parents = ""
            reading = ("**名前側は層では0件**" if folders else "**名前側が0件**")
            if folders:
                reading += f" —— **フォルダーで {n_folders}。遮蔽しているのは中の層で、"\
                           "どの層かは `ancestors` が要る(依頼8)**"
        else:
            top_n_pages = top["n_pages"]
            top_first, top_last = top["first_year_label"], top["last_year_label"]
            top_parents = top["n_parent_names"]
            reading = (f"最大の1名で層 {top['n_layers']} / 頁 {top_n_pages}。"
                       f"**頁は名前をまたいで足せない**(同じ頁に別の名前が在りうる)")
            if n_folders:
                reading += f" / フォルダーでも {n_folders}"
        out.append([key, label, len(hit), n_layers, n_folders,
                    top_n_pages, top_first, top_last, top_parents,
                    reading, ";".join(f"{k}:{v}" for k, v in kinds.most_common())])
    return out


def vocab_year_labels(rows):
    """**公開してよい年の札**を、依頼7の表から集める。

    **年の札は段 A で公開する決定が在る**(作者、2026-10-26)。だが**この関数が
    返した語はすべて白名簿に入る**ので、**年の形でない語が1つでも混ざれば
    点検に穴が開く。** だから形を強制し、外れたら走行を落とす。
    """
    labels = set()
    for r in rows:
        for col in ("first_year_label", "last_year_label"):
            v = (r.get(col) or "").strip()
            if not v:
                continue
            if not re.fullmatch(r"\d{4}[a-z]?", v):
                raise SystemExit(
                    f"**年の札でない値が年の列に在る**: {col}={v!r}\n"
                    "  **白名簿に入れない。** 年の列に別のものが入っている")
            labels.add(v)
    return labels


# --------------------------------------- 全数の重なり順(依頼8 の構造の表、2026-10-27 着)
#
# **遮蔽物であることは、名前でも種別でもなく順序の性質である**(こちらが書いた言い方)。
# 作者の言明に戻ると: **「手前側に何かがあると、その後ろ側のレイヤでは描かないで済ます」**
# —— **手前側であることが効いている。** だから名前で集めた層が**本当に手前に居るか**を
# 測る。これが**名前側(全数)に対する、もう一方の側**である。
#
# **素性側(画素の分布)は8頁しか届かない。** 構造の側は**全 437,561層に届く。**
# **だから両面のうち全数で引けるのは構造の側である。**
#
# **供給元の言明**(添え状 §1 と v2 の訂正):
# - `z_global` は**一番上のフォルダーを 0 とし、その下は下の層から数える。**
#   **フォルダー以外の層どうしでは、数の大きいほうが手前。**
#   v1 は向きを逆に書いていて、**v2 で文章だけを訂正した(CSV は正しかった)**
# - **`z_global` が空の 39層は頁の重なり順に入らない**(木から辿れない24層とその中の15層)
# - 隣の兄弟 335,454組すべてで `order_in_parent` と `z_global` の向きが一致(例外 0)
#
# **棒の前に帰無値を出す**(規律9)—— 「前方に寄っている」は、
# **同じ頁の層を1つ選んだときの前方割合**と比べて初めて言える。
R8_DEFAULT = os.path.join("inbox", "census_r8_layer_structure_20261027.csv")

# 便 n1(年 × 形式の頁数)と便 n2(形式ごとの層数)。
# **n2 は突き合わせにしか使わない** —— 行は依頼8 から出す(年の次元が在る)。
# **同じ DB の同じ走行なので、一致は一致であって独立の確認ではない**(T132)
N1_DEFAULT = os.path.join("inbox", "census_n1_pages_by_year_20261027.csv")
N2_DEFAULT = os.path.join("inbox", "census_n2_layers_by_format_20261027.csv")
N4_DEFAULT = os.path.join("inbox", "census_n4_lip_layers_20261027.csv")

BASELINE_KEY = "baseline"

# **帰無値の実測の形**(基盤が 2026-10-27 に足した)。
#
# 解析の帰無値(同じ群の層を1つ選んだときの前方割合の中央 = 0.500)は**定義から出る数**で、
# **定義が間違っていても 0.500 になる。** だから**同じ軸の上で、前後に寄る理由の無い名前**を
# 1つ測って並べる —— **線画は紙面の主題で、手前にも後ろにも寄る理由が無い。**
# **これが 0.500 付近に出れば、前方割合の軸そのものが素直だと分かる。**
#
# **これは遮蔽物ではない。** だから `CLASSES` に入れず、この表だけで使う
# (`CLASSES` に入れると、名前の表と素性の表にも遮蔽物として並ぶ)。
FRONT_CONTROLS = (
    ("control_lineart", "線画(**対照。前後に寄る理由の無い名前**)", r"^線画$"),
)
SCOPES = (("page", "頁の中(フォルダー以外の層すべて)"),
          ("parent", "同じ親の中(兄弟が2つ以上の親だけ)"))

# **「前方1割」の境界。**
#
# **これは列名が運んでいない量である** —— 基盤が同じ所を `<= 0.1` で測り、
# **story_info 7,405 対 7,440・text 6,538 対 6,728 とずれた**(2026-10-27)。
# **名前は同じで、境界だけが違った。** 差は 529層。
#
# **だから閾と向きを器具の中に名前付きで置き、母集団の欄に書く。**
# **どちらが正しいかではない** —— **どちらで測ったかが書かれていないことが誤りだった。**
# **手で描いた線ではない種類。** `text` は CSP の文字オブジェクト、
# `story_info` はノンブルや作品情報で、**どちらもアプリが置く役割の層**である。
# **この2つを束ねた数は、器具が数から出す** ——
# **印字された割合(小数3桁)を足して作らない。**
# `0.025 + 0.061 = 0.086` と書いたが、**数から出すと 0.0853 である**
# (2026-10-27、基盤が指摘。**丸めの和は、測った量ではない**)。
NON_PICTURE_KINDS = ("story_info", "text")

FRONT_DECILE = 0.1
FRONT_DECILE_RULE = f"前方割合 < {FRONT_DECILE}(**境界の値 {FRONT_DECILE} は含まない**)"


_STRUCT_CACHE = {}


# --------------------------------------- 頁のパスから取れる2つの次元(**名前は取らない**)
#
# **ここは遮蔽物の話ではない。** 構造の表を読む器具がここに在るので、ここに置いた。
#
# **`.lip` を母集団に入れる決定は作者のもの**(2026-10-27、作者: 「入れる」)。
# **決定の前から、依頼7と依頼8 の表には `.lip` が入っていた** ——
# 312,898(`.clip`)+ 124,663(`.lip`)= 437,561 で、**「全数」と書いた数は両形式の和だった。**
# **決定が変えたのは材料ではなく、書いてあることである**(T139 の形: 出力は作り直されるが
# 文は作り直されない)。だから**形式を次元として出す。**
PAGE_FORMATS = ("clip", "lip")
YEAR_SEGMENT = 2          # `F:\manuscripts\<年>\...` の3つめ。**外れたら数えて落とす**


def path_format(page_path: str) -> str:
    """頁のパスの拡張子。**宣言した2つ以外は空で返す**(呼び出し側が数えて印字する)。"""
    ext = page_path.rsplit(".", 1)[-1].lower() if "." in page_path else ""
    return ext if ext in PAGE_FORMATS else ""


def path_year(page_path: str) -> str:
    """頁のパスの年の区画。**位置で取る。**

    **形で探さない** —— 作品の名が年の形をしていた日に、作品名を年の札として
    白名簿に入れてしまう。**位置で取り、形も強制する**(ここで返した札は公開の
    白名簿に入るので、年の形でない語が1つ混ざれば点検に穴が開く ——
    `vocab_year_labels` と同じ理由)。
    """
    segs = page_path.split("\\")
    if len(segs) <= YEAR_SEGMENT:
        return ""
    v = segs[YEAR_SEGMENT]
    return v if re.fullmatch(r"\d{4}[a-z]?", v) else ""


def load_structure(path: str, drops):
    """構造の表を、**層名を持たない形に畳んで**読む。

    **名前はここで落とす** —— 返すのは「どの類に当たったか」の印だけ。
    **後ろの集計に名前を渡さないので、出力に名前が出る道が無い。**
    """
    import csv as _csv
    if not os.path.exists(path):
        drops.flag("全数の構造の表が無い(依頼8 が未着)", path)
        return {}, {}, 0
    # **52MB を段 A と段 C で2回読まない。** 同じ走行の同じファイルは同じ結果である
    if path in _STRUCT_CACHE:
        return _STRUCT_CACHE[path]
    pats = {k: re.compile(p, re.I) for k, _, p, _ in CLASSES if p}
    pats.update({k: re.compile(p, re.I) for k, _, p in FRONT_CONTROLS})
    pages = collections.defaultdict(list)
    folder_state = {}
    folder_child_rows = collections.Counter()
    folder_children = collections.Counter()
    folder_rows = collections.defaultdict(list)
    # **類ごとに、どの名前が何層か。** 名前はここから外に出さない ——
    # **最頻の1名の印に畳んでから返す**(下)
    names_per_class = collections.defaultdict(collections.Counter)
    matched_at = []
    n_rows = n_no_z = 0
    # **形式と年**(頁のパスから。名前は取らない)。**落ちた行は数えて返す**(規律9)
    page_dims = {}
    fmt_year = collections.defaultdict(collections.Counter)
    fmt_year_pages = collections.defaultdict(set)
    n_bad_fmt = n_bad_year = 0
    with open(path, encoding="utf-8", newline="") as fh:
        rd = _csv.reader(fh)
        head = next(rd)
        ix = {c: i for i, c in enumerate(head)}
        need = ("page_path", "layer_id", "layer_name", "v_layer_kind", "parent_id",
                "ancestors", "composite", "clip", "visibility", "opacity",
                "is_folder", "z_global")
        missing = [c for c in need if c not in ix]
        if missing:
            raise SystemExit(f"依頼8 の表に宣言した列が無い: {missing}")
        for row in rd:
            n_rows += 1
            pg_ = row[ix["page_path"]]
            dims = page_dims.get(pg_)
            if dims is None:
                dims = page_dims[pg_] = (path_format(pg_), path_year(pg_))
            fmt, yr = dims
            if not fmt:
                n_bad_fmt += 1
                drops.flag("頁のパスの拡張子が宣言した2形式でない(形式ごとの表)")
            if not yr:
                n_bad_year += 1
                drops.flag("頁のパスの3つめが年の札の形でない(形式ごとの表)")
            if fmt and yr:
                cell = fmt_year[(yr, fmt)]
                cell["sum_n_layers_incl_folders"] += 1
                if row[ix["is_folder"]] == "True":
                    cell["sum_n_folders"] += 1
                fmt_year_pages[(yr, fmt)].add(pg_)
            name = row[ix["layer_name"]]
            anc = row[ix["ancestors"]]
            by_name = frozenset(k for k, p in pats.items() if name and p.search(name))
            by_anc = frozenset(k for k, p in pats.items()
                               if anc and p.search(anc)) - by_name
            # **直下の行(フォルダーの行も含む)を、層とは別に数える**(T138)——
            # `folder_children` は**フォルダー以外の層**しか数えないので、
            # **中が部分フォルダーだけのフォルダーが「空」に見える**
            # (2026-10-27 に基盤と `white` で 61 対 71 として出た)。
            if row[ix["parent_id"]]:
                folder_child_rows[(row[ix["page_path"]], row[ix["parent_id"]])] += 1
            is_folder = row[ix["is_folder"]] == "True"
            if is_folder:
                for k in by_name:
                    folder_rows[k].append((row[ix["page_path"]], row[ix["layer_id"]]))
                # **祖先の表示を見るために、フォルダーの状態を覚える**(T143)
                folder_state[(row[ix["page_path"]], row[ix["layer_id"]])] = (
                    row[ix["parent_id"]], int(row[ix["visibility"]] or 0),
                    int(row[ix["opacity"]] or 0))
                continue
            pid = row[ix["parent_id"]]
            if pid:
                folder_children[(row[ix["page_path"]], pid)] += 1
            z = row[ix["z_global"]]
            if not z:
                n_no_z += 1
                drops.flag("`z_global` が空(頁の重なり順に入らない層)")
                continue
            pg = row[ix["page_path"]]
            pages[pg].append(
                (int(z), pid, by_name, by_anc,
                 int(row[ix["opacity"]] or 0), int(row[ix["visibility"]] or 0),
                 row[ix["composite"]], row[ix["v_layer_kind"]], frozenset(), False))
            if by_name:
                for k in by_name:
                    names_per_class[k][name] += 1
                matched_at.append((pg, len(pages[pg]) - 1, name, by_name))
    # **語彙の中で最頻の1名だけを別に数えられるようにする** ——
    # **語彙全体と1名は別の量**(2026-10-27 に基盤の再現で 0.152 対 0.143 として出た。
    # **正規化は同じで、違いは語彙の広さだった**)。**だから両方を行に出す。**
    top = {k: c.most_common(1)[0][0] for k, c in names_per_class.items() if c}
    for pg, i, name, by_name in matched_at:
        hit = frozenset(k for k in by_name if top.get(k) == name)
        if hit:
            x = pages[pg][i]
            pages[pg][i] = x[:8] + (hit,) + x[9:]
    # **祖先のフォルダーが非表示・不透明度0なら、その中の層は描かれない**(T143)——
    # **「層自身が可視」と「描かれる」は別の量である。** 作者の定義は「非表示レイヤーは
    # 含めない」で、CSP では**非表示フォルダーの中身は描かれない**(描画の事実)。
    # **層の1行だけで走査できる量を使うと、定義に従った実装の方が落ちる**
    # (基盤が 2026-10-27 に自分の点検で踏んで回送した。**点検の値は定義から導く**)。
    n_hidden_anc = 0
    # **2つの述語が一致する範囲を、走行が数える**(基盤の規則、2026-10-27)——
    # 「表の1列で済む走査」(自身だけ見る)と定義(祖先も見る)が**どこまで同じ答を出すか。**
    # **借りた割合を散文に書かない**: 母集団が違えば数が違う(先方は 90.8%、こちらは別)
    pred_n = collections.Counter()
    pred_diff = collections.Counter()
    for pg, ls in pages.items():
        fmt_ = page_dims.get(pg, ("", ""))[0]
        for i, x in enumerate(ls):
            hid = False
            parent = x[1]
            for _ in range(64):
                st = folder_state.get((pg, parent))
                if st is None:
                    break
                parent, pvis, pop = st
                if pvis % 2 == 0 or pop == 0:
                    hid = True
                    break
            pred_n[fmt_] += 1
            pred_n[""] += 1
            if hid:
                ls[i] = x[:9] + (True,)
                n_hidden_anc += 1
                # 自身は表示なのに祖先が非表示 = **2つの述語が違う答を出す層**
                if x[5] % 2 == 1 and x[4] > 0:
                    pred_diff[fmt_] += 1
                    pred_diff[""] += 1
    _STRUCT_CACHE[path] = (
        pages, {"folder_rows": folder_rows, "folder_children": folder_children,
                "n_no_z": n_no_z,
                "page_fmt": {pg: d[0] for pg, d in page_dims.items()},
                "fmt_year": fmt_year,
                "fmt_year_pages": {k: len(v) for k, v in fmt_year_pages.items()},
                "n_bad_fmt": n_bad_fmt, "n_bad_year": n_bad_year,
                "n_hidden_ancestor": n_hidden_anc,
                "pred_n": pred_n, "pred_diff": pred_diff,
                "folder_state": folder_state,
                "folder_child_rows": folder_child_rows}, n_rows)
    return _STRUCT_CACHE[path]


def load_pages_by_year(path: str, drops):
    """年 × 形式の**頁数**(便 n1)。

    **頁の数はこの便だけが持っている** —— 依頼8 の表は**層の在る頁**しか持たないので、
    **読めなかった4頁がそこには出てこない。**「層が在る頁」と「頁」は別の量である。
    """
    import csv as _csv
    if not os.path.exists(path):
        drops.flag("年ごとの頁数の表が無い(便 n1 が未着)", path)
        return {}
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in _csv.DictReader(fh):
            yr = (r.get("year_label") or "").strip()
            fmt = (r.get("format") or "").strip()
            # **年の札は白名簿に入る語**なので、形を強制して走行を落とす
            if not re.fullmatch(r"\d{4}[a-z]?", yr) or fmt not in PAGE_FORMATS:
                raise SystemExit(
                    f"**便 n1 に宣言外の値**: year_label={yr!r} format={fmt!r}\n"
                    "  **白名簿に入れない。** 年か形式の列に別のものが入っている")
            out[(yr, fmt)] = (int(r["n_pages"]), int(r["n_read_error"] or 0))
    return out


def population_by_format(aux, n1, drops, folded=None):
    """**母集団を形式ごとに出す表。** 作者の決定(2026-10-27): **`.lip` を入れる。**

    **2つの便を並べる**: 頁の数は便 n1、層とフォルダーの数は依頼8 の表。
    **どちらも同じ DB の同じ走行**なので、**一致は一致であって独立の確認ではない**(T132)。

    **この表が答えない**:
    - **`.lip` の画素。** 1頁も無い —— 便 n4 の `rendered` は全行が空で、抽出側の言明は
      **「描けなかったのではなく、描いていない」。** だから**素性の側(画素の分布)に
      `.lip` は 0 頁**である —— **0 と測ったのではなく、材料が無い**(T126)
    - **形式が年と独立であること。** **`.lip` は早い年に偏る**(行を見ること)。
      **だから形式ごとの差を世代の差と読み分けられない** —— この表では分離できない
    - **「層」が何を数えているか。** `sum_n_layers_incl_folders` は DB の `layers` の行で、
      **フォルダーを含む**(便 n2 の `n_layers` と同じ量)
    """
    rows = []
    keys = sorted(set(n1) | set(aux.get("fmt_year", {})))
    fy = aux.get("fmt_year", {})
    fyp = aux.get("fmt_year_pages", {})
    tot = collections.defaultdict(collections.Counter)
    for yr, fmt in keys:
        n_pages, n_err = n1.get((yr, fmt), ("", ""))
        c = fy.get((yr, fmt), collections.Counter())
        n_lay = c["sum_n_layers_incl_folders"]
        n_fol = c["sum_n_folders"]
        n_with = fyp.get((yr, fmt), 0)
        for k, v in (("n_pages", n_pages if n_pages != "" else 0),
                     ("n_read_error", n_err if n_err != "" else 0),
                     ("n_pages_with_layers", n_with),
                     ("sum_n_layers_incl_folders", n_lay),
                     ("sum_n_folders", n_fol)):
            tot[fmt][k] += v
            tot["both"][k] += v
        rows.append(["year", yr, fmt, n_pages, n_err, n_with, n_lay, n_fol, ""])
    # **年の札ごとに、どの形式が在るか。**
    # **「早い年に偏る」と散文で書かない** —— `.lip` は 2020 にも在り、
    # **2012〜2020 の6札に在って 2020b 以降に無い**。札を数えて書く
    all_years = sorted({yr for yr, _ in keys})
    years_of = {fmt: sorted({yr for yr, f in keys if f == fmt})
                for fmt in PAGE_FORMATS}
    # **年ごとの行の和は、この器具が計算する**(T140: 公開した行を読み手が足さない)
    for fmt in list(PAGE_FORMATS) + ["both"]:
        c = tot[fmt]
        if not c:
            continue
        # **分母の種別を散文にも書く** —— 基盤が同じ割合を**非フォルダーの分母**で出して
        # 66.3% 対 70.0% にずれた(2026-10-27)。**どちらも正しく、どちらも「層」と
        # 書いていた。** 列名(`sum_n_layers_incl_folders`)は運んでいたが、
        # **`reading` の散文が運んでいなかった** —— T138 の家族が分母の軸で出た形
        share = (f"全層(**フォルダー込み**)の "
                 f"{c['sum_n_layers_incl_folders'] / tot['both']['sum_n_layers_incl_folders']:.3f}"
                 f" / 全頁の {c['n_pages'] / tot['both']['n_pages']:.3f}"
                 f" / この形式が在る年の札 {len(years_of[fmt])}(全 {len(all_years)} 札): "
                 + "・".join(years_of[fmt])
                 if fmt != "both" and tot["both"]["sum_n_layers_incl_folders"] else
                 f"**この2形式で母集団の全部**(作者の決定 2026-10-27)。"
                 f"年の札は全 {len(all_years)}")
        rows.append(["format_total", "", fmt, c["n_pages"], c["n_read_error"],
                     c["n_pages_with_layers"], c["sum_n_layers_incl_folders"],
                     c["sum_n_folders"], share])
    # 年の行の読み: **その年の層のうちこの形式が占める割合**(年の中で形式が混ざる年が在る)
    by_year = collections.Counter()
    for r in rows:
        if r[0] == "year":
            by_year[r[1]] += r[6]
    for r in rows:
        if r[0] == "year" and by_year[r[1]]:
            r[8] = (f"この年の層(**フォルダー込み**)の "
                    f"{r[6] / by_year[r[1]]:.3f} がこの形式")
    # **畳んだ後の頁数を、畳む前の隣に置く**(T140: 読み手に足させない・引かせない)。
    # **畳む規則は作者の裁定**で、述語は `hashes.folded_by_year` に書いてある。
    # **畳む前と畳んだ後は同じ母集団の別の量** —— **差は量ではなく手続きの結果**
    # (余り 2,235 は「複製されたファイルの数」ではない。T96 の家族)。
    # **この列が無い間、年ごとの傾向は読めなかった** —— 池は一様に縮まないので
    # (2023b 0.480 / 2024c 1.000)、**畳む前の数で年を比べると複製の多い年ほど
    # 大きく見える**(Track M の指摘、2026-10-28)
    if folded:
        fsum = collections.Counter()
        for (yr, fmt), (_before, after) in folded.items():
            fsum[("year", yr, fmt)] += after
            fsum[("format_total", "", fmt)] += after
            fsum[("format_total", "", "both")] += after
        out = []
        for r in rows:
            n_after = fsum.get((r[0], r[1], r[2]), "")
            n_before = r[3] if isinstance(r[3], int) else 0
            kept = f"{n_after / n_before:.3f}" if n_after != "" and n_before else ""
            out.append(r[:8] + [n_after, kept] + r[8:])
        rows = out
    return rows


def check_formats(aux, n2_path: str, n4_path: str, drops):
    """**3つの便が同じ数を言っているか。**

    - 依頼8 の表から数えた形式ごとの層数 ↔ 便 n2 の `n_layers` / `n_folders`
    - 依頼8 の `.lip` の層数 ↔ 便 n4 の行数
    - **便 n4 の `rendered` が本当に全行空か**(抽出側の言明: 「描いていない」)

    **届く**: こちらの読み取りの誤り(拡張子の取り方・年の位置)、便どうしの食い違い、
    言明と中身の食い違い。
    **届かない**: DB そのもの。**3便とも同じ DB の同じ走行から出ている**ので、
    **一致は一致であって、独立な確認ではない**(T132)。
    """
    import csv as _csv
    out = []
    fy = aux.get("fmt_year", {})
    mine = collections.defaultdict(collections.Counter)
    for (yr, fmt), c in fy.items():
        mine[fmt]["n_layers"] += c["sum_n_layers_incl_folders"]
        mine[fmt]["n_folders"] += c["sum_n_folders"]
    if os.path.exists(n2_path):
        with open(n2_path, encoding="utf-8", newline="") as fh:
            for r in _csv.DictReader(fh):
                fmt = (r.get("format") or "").strip()
                if fmt not in PAGE_FORMATS:
                    continue
                for col in ("n_layers", "n_folders"):
                    out.append((f"依頼8 から数えた {fmt} の {col}",
                                mine[fmt][col], int(r[col])))
    else:
        drops.flag("便 n2 が無い(形式ごとの層数の突き合わせをしていない)", n2_path)
    if os.path.exists(n4_path):
        n4_rows = 0
        n4_rendered = 0
        with open(n4_path, encoding="utf-8", newline="") as fh:
            for r in _csv.DictReader(fh):
                n4_rows += 1
                if (r.get("rendered") or "").strip():
                    n4_rendered += 1
        out.append(("便 n4 の行数(`.lip` の層)", mine["lip"]["n_layers"], n4_rows))
        out.append(("便 n4 の `rendered` が空でない行"
                    "(**抽出側の言明: 1枚も描いていない**)", 0, n4_rendered))
    else:
        drops.flag("便 n4 が無い(`.lip` の層の表の突き合わせをしていない)", n4_path)
    return out


def _front_shares(groups):
    """各層の**前方割合** = 自分より手前に在る層の数 ÷ (その群の層数 − 1)。

    **0 が最前、1 が最後。** 群が1層だけの親は**比が定義できないので入れない**
    (落とした数は呼び出し側が印字する)。
    """
    import bisect
    out = []
    skipped = 0
    for ls in groups:
        if len(ls) < 2:
            skipped += len(ls)
            continue
        zs = sorted(x[1][0] for x in ls)
        m = len(ls) - 1
        for pg, x in ls:
            out.append(((len(zs) - bisect.bisect_right(zs, x[0])) / m, pg, x))
    return out, skipped


def _ratio_word(got: float, base: float) -> str:
    """**文言を比から導く。** 「寄っている」と書いてから数を見ない。"""
    if base == 0:
        return "帰無値が 0 なので倍率を書けない"
    r = got / base
    if r >= 1.5:
        return f"前方1割は**帰無値の {r:.1f} 倍**"
    if r <= 1 / 1.5:
        return f"前方1割は**帰無値の {r:.2f} 倍**"
    return f"前方1割は帰無値の {r:.2f} 倍(**変わらない**)"


def _median_word(med: float, base: float) -> str:
    """中央の向き。**前方割合は 0 が最前なので、小さいほど手前である。**

    **2つの統計量が食い違うことが在る**(フキダシの親の中: 中央は後ろ寄り、
    前方1割は帰無値どおり)。**片方だけ書くと、書いた側の話になる。**
    """
    d = med - base
    if d <= -0.1:
        return f"中央 {med:.3f} < 帰無値 {base:.3f} —— **手前寄り**"
    if d >= 0.1:
        return f"中央 {med:.3f} > 帰無値 {base:.3f} —— **後ろ寄り**"
    return f"中央 {med:.3f} ≒ 帰無値 {base:.3f} —— **寄りなし**"


def front_position(pages, aux, drops):
    """**全数の構造の側。** 名前で集めた層が、本当に手前に居るか。

    **この表が答えない**:
    - **何を隠しているか。** 手前に在ることは重なっていることではない
      (**画素が要る。保留中**)。小さい層・透明な層も手前に在る
    - **前方割合が1つの量であること。** **頁の中と親の中で別の量になる**
      (フキダシは**頁では手前、親の中では後ろ** —— 文字がその上に乗るから)。
      **同じ列名で2つを書かない**(T138)。`scope` の列で分ける
    - **向き。** `z_global` の向きは**供給元の v2 の訂正に依っている。**
      こちらでは兄弟の `order_in_parent` との一致でしか確かめていない
    - **不透明度が遮蔽の強さであること。** `opacity = 256` は層の設定で、
      **画素のアルファではない**(T87 の形: 設定は中身を含意しない)
    - **可視であることが、描かれていることであること。** `share_visible_bit` と
      `share_opacity_full` は**層自身の値**で、**祖先のフォルダーが非表示なら
      その層は描かれない**(T143)。**だから隣に `share_ancestor_hidden` を出す** ——
      **可視の列だけを読むと、描かれない層を遮蔽物に数える**
    """
    import statistics
    rows = []
    pf = aux.get("page_fmt", {})

    def split(pgs):
        """**形式ごとの層数。** `.lip` を母集団に入れたのは作者の決定(2026-10-27)で、
        **入れた以上、どの行が何形式の層で出来ているかを書く**(`.lip` は早い年に偏るので、
        形式の内訳は**世代の内訳でもある**)。"""
        c = collections.Counter(pf.get(pg, "") for pg in pgs)
        return c["clip"], c["lip"]

    by_page = [[(pg, x) for x in ls] for pg, ls in pages.items()]
    by_parent = collections.defaultdict(list)
    for pg, ls in pages.items():
        for x in ls:
            by_parent[(pg, x[1])].append((pg, x))
    groups = {"page": by_page, "parent": list(by_parent.values())}
    for scope, scope_label in SCOPES:
        shares, skipped = _front_shares(groups[scope])
        base = [s for s, _, _ in shares]
        base_med = statistics.median(base)
        base_dec = sum(1 for s in base if s < FRONT_DECILE) / len(base)
        rows.append([BASELINE_KEY, "**帰無値**(同じ群の層を1つ選んだとき)", "baseline",
                     scope, len(base), len(pages),
                     *split([pg for _, pg, _ in shares]),
                     f"{base_med:.3f}", f"{base_dec:.3f}", "", "", "",
                     f"{sum(1 for _, _, x in shares if x[9]) / len(base):.3f}",
                     f"{scope_label}。**比が定義できない群の層 {skipped} は"
                     f"入れていない**(兄弟が自分だけの親)。"
                     f"**自身の可視・不透明度の3列は群では別の量なので空にしてある** ——"
                     f"**祖先が非表示の割合だけは、棒の前に出す帰無値である**"])
        classes = [(k, l, p) for k, l, p, _w in CLASSES if p is not None]
        classes += [(k, l, p) for k, l, p in FRONT_CONTROLS]
        for key, label, pat in classes:
            for matched_on, pick in (
                    ("layer_name", lambda x: key in x[2]),
                    ("top_name", lambda x: key in x[8]),
                    ("other_names", lambda x: key in x[2] and key not in x[8]),
                    ("ancestors", lambda x: key in x[3])):
                sel = [(s, pg, x) for s, pg, x in shares if pick(x)]
                if not sel:
                    continue
                v = [s for s, _, _ in sel]
                n = len(v)
                med = statistics.median(v)
                dec = sum(1 for s in v if s < FRONT_DECILE) / n
                reading = (f"{_median_word(med, base_med)} / "
                           f"{_ratio_word(dec, base_dec)}")
                rows.append([
                    key, label, matched_on, scope, n,
                    len({pg for _, pg, _ in sel}),
                    *split([pg for _, pg, _ in sel]),
                    f"{med:.3f}", f"{dec:.3f}",
                    f"{sum(1 for _, _, x in sel if x[4] == 256) / n:.3f}",
                    f"{sum(1 for _, _, x in sel if x[5] % 2 == 1) / n:.3f}",
                    f"{sum(1 for _, _, x in sel if x[6] == '0') / n:.3f}",
                    f"{sum(1 for _, _, x in sel if x[9]) / n:.3f}",
                    reading])
    return rows


def front_decile_kinds(pages, aux, drops):
    """**軸の端を、どの種類が占めているか。**

    **なぜ要るか**(基盤、2026-10-27): 対照(線画)は**中央では帰無値どおり(0.508)**だが、
    **前方1割では帰無値を下回る(0.048 対 0.112)。**
    **対照が保証するのは、比べた点の軸だけである。**

    **その理由をここで数にする** —— 前方1割を占めているのが何かを種類ごとに出す。
    **物語情報と文字がそこに居るなら、絵の層は最前に出てこない。**

    **この表が答えない**: **その層が何かを隠しているか**(手前であることは重なりではない)。
    **種類が役割であること**(`story_info` はノンブルや作品情報で、紙面の主題ではない)。
    """
    import bisect
    dec = collections.Counter()
    allk = collections.Counter()
    dec_fmt = collections.Counter()      # (種類, 形式) —— **前方1割の中の内訳**
    pf = aux.get("page_fmt", {})
    for pg, ls in pages.items():
        if len(ls) < 2:
            drops.flag("群が1層で前方割合が定義できない(前方1割の表)")
            continue
        fmt = pf.get(pg, "")
        zs = sorted(x[0] for x in ls)
        m = len(ls) - 1
        for x in ls:
            share = (len(zs) - bisect.bisect_right(zs, x[0])) / m
            allk[x[7]] += 1
            if share < FRONT_DECILE:
                dec[x[7]] += 1
                dec_fmt[(x[7], fmt)] += 1
    n_dec = sum(dec.values())
    n_all = sum(allk.values())
    # **束ねた数は数から出す**(上の `NON_PICTURE_KINDS` の注)
    np_dec = sum(dec[k] for k in NON_PICTURE_KINDS)
    np_all = sum(allk[k] for k in NON_PICTURE_KINDS)
    combined = dict(kinds=NON_PICTURE_KINDS, n_in_decile=np_dec, n_in_all=np_all,
                    share_of_decile=np_dec / n_dec if n_dec else 0.0,
                    share_of_all=np_all / n_all if n_all else 0.0)
    rows = []
    for kind, n in dec.most_common():
        rows.append([kind, n, dec_fmt[(kind, "clip")], dec_fmt[(kind, "lip")],
                     f"{n / n_dec:.3f}", f"{n / allk[kind]:.3f}",
                     f"{allk[kind] / n_all:.3f}",
                     "**この種類のほとんどが前方1割に在る**"
                     if n / allk[kind] >= 0.5 else
                     ("**前方1割に偏って多い**"
                      if n / n_dec > allk[kind] / n_all * 1.5 else
                      ("**前方1割では少ない**"
                       if n / n_dec * 1.5 < allk[kind] / n_all else "偏りなし"))])
    return rows, n_dec, n_all, combined


# **作者の判断**(2026-10-27)。**測定ではない。**
#
# 空の `黒消し` フォルダー 531/609 を「数として出し、結論は書かない」(規律6)として
# 作者に問うた答:
#
# > **空であればテンプレで作られて使用しなかったものだろう。一時期そういう工程だった。**
#
# **これは作者のものなので、こちらの数で裏付けたとは書かない。** 数の側から言えるのは
# **空のフォルダーが在る年の札が限られている**ことだけで、**「一時期」と一致して見える**のは
# 一致であって確認ではない(作者はもう答を持っているので、裏付けの要る主張ではない)。
AUTHOR_READING = {
    "black_erase": "**作者の判断(2026-10-27、測定ではない)**: "
                  "**空であればテンプレで作られて使用しなかったものだろう。"
                  "一時期そういう工程だった。**",
}


def folder_contents(aux):
    """**名前がフォルダーに付いている類**(黒消し)の中身。

    **`黒消し` は層では 0件、フォルダーで 609。** だから**遮蔽しているのは
    その中の層**で、**フォルダーの数を遮蔽物の数と呼べない。**
    **中が空のフォルダーも数える**(規律9: 落としたものを理由別に印字する)。
    """
    rows = []
    pf = aux.get("page_fmt", {})
    py = {pg: path_year(pg) for pg in pf}
    for key, label, pat, want_kind in CLASSES:
        if pat is None:
            continue
        fr = aux["folder_rows"].get(key, [])
        if not fr:
            continue
        kids = aux["folder_children"]
        with_kids = sum(1 for pp, lid in fr if kids.get((pp, lid), 0))
        n_kids = sum(kids.get((pp, lid), 0) for pp, lid in fr)
        fc = collections.Counter(pf.get(pp, "") for pp, _ in fr)
        # **フォルダー自身が非表示なら、中の層は描かれない**(T143)。
        # **「中身が在る」と「描かれる」は別の量である。**
        st = aux.get("folder_state", {})
        hidden = {(pp, lid) for pp, lid in fr
                  if (lambda v: v is None or v[1] % 2 == 0 or v[2] == 0)(
                      st.get((pp, lid)))}
        nonempty = {(pp, lid) for pp, lid in fr if kids.get((pp, lid), 0)}
        rowkids = aux.get("folder_child_rows", {})
        with_rows = sum(1 for pp, lid in fr if rowkids.get((pp, lid), 0))
        if hidden == nonempty:
            vis_read = (f"**直下に層が在るフォルダー {len(nonempty)} と、"
                        f"非表示のフォルダー {len(hidden)} は同じ集合である"
                        f"(例外0)** —— **中身が在るものは、全部消してある。**"
                        f" **答えないもの**: 消した理由 / 一度使って後で消したのか")
        else:
            rows_vis = sum(1 for pp, lid in fr
                           if rowkids.get((pp, lid), 0) and (pp, lid) not in hidden)
            vis_read = (f"非表示のフォルダー {len(hidden)} / 直下に層が在る {len(nonempty)}。"
                        f"**層が在って表示 {len(nonempty - hidden)}** —— "
                        f"**同じ問いを行で数えると {with_rows} のうち表示 {rows_vis}。"
                        f"数えるものが違えば数は違う**(T138)")
        years = sorted({py.get(pp, "") for pp, lid in fr
                        if not kids.get((pp, lid), 0)} - {""})
        rows.append([key, label, len(fr), fc["clip"], fc["lip"],
                     with_kids, with_rows, len(fr) - with_kids, len(hidden), n_kids,
                     len({pp for pp, _ in fr}),
                     AUTHOR_READING.get(key, "")
                     + (f" 空のフォルダーが在る年の札: {'・'.join(years)}"
                        if years and key in AUTHOR_READING else ""),
                     "**フォルダーの数を遮蔽物の層の数と呼べない** —— "
                     f"直下の層 {n_kids}。"
                     f"**直下に層(フォルダー以外)が無いフォルダーが {len(fr) - with_kids}"
                     f" —— ただし直下に行が在るものは {with_rows} で、"
                     f"差の {with_rows - with_kids} は中が部分フォルダーだけのもの**。"
                     + vis_read])
    return rows
