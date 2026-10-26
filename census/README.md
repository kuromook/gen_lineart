# `census/` —— 原稿の国勢調査(**公開してよい集計だけ**)

**このフォルダーは追跡される。** `origin` は**匿名で読める公開リポジトリ**なので、
**ここに置いたものは全世界から読める。**

## 規則

1. **段 A だけ**(ユーザーの決定 2026-10-26)—— **年・形式・種別ごとの件数と分布。**
   **作品名も層名も頁のパスも置かない。**
2. **原稿由来のラスタ・プレビュー・サムネイルを置かない。** 画像は1枚も置かない
3. **固有名を含む全量は `data/census_full/`**(`.gitignore` 配下。追跡されない)
4. **器具が「固有名を含む列は0」と印字しない走行の出力を commit しない**
5. **各数の隣に「その数が答えない問い」を置く** —— `doc/census_spec.md` §3-3。
   **これが無い集計は、重さを減らさない**(この集計が存在する理由そのもの)

## 仕様と作り手

- 仕様: `doc/census_spec.md`
- 作るのは **track `lineart-census`**。**基盤は実装しない**(規律5)

## 中身(2026-10-26 に入った)

**器具**: `tools/census/build_census.py` —— **1本の走行が段 A と段 C を同じ集計から出す。**

| ファイル | 中身 |
|---|---|
| `populations.csv` | **母集団の表。** 各行に述語・測った者・不足している材料。**ここから読む** |
| `pages_by_year.csv` | 年ごとの頁数・作品数・作品フォルダーの無い頁数 |
| `layers_by_year_kind_type.csv` | 年 × 種別 × `layer_type` の層数・描画・`drawn_ratio` の桶 |
| `layer_type_values.csv` | **全数 437,561層**の `layer_type` 全11種(抽出側が測った) |
| `layer_type_bits.csv` | ビットごとの意味。**確かなのは 1・2・4096 だけ。残りは「不明」** |
| `mask_by_kind.csv` | 2の位とマスク。**コマ枠フォルダー 8,734 は別の行** |
| `visibility_exceptions.csv` | 食い違いの数。**全数 119 と優先65頁 245 を母集団つきで両方** |
| `body_initcolor.csv` | 本体の `InitColor`(**ここで行から数え直した**)。種類 × `init` の交差表 |
| `caveats.csv` | **各列の「答えない問い」。** この集計の本体 |
| `provenance.csv` | 受領 CSV の sha256 と、**値が動く予定の列** |

**読み方の順序は `populations.csv` → `caveats.csv` → 数の表。**
数の表だけを読むと、この集計が止めようとしている誤りがそのまま起きる。

**空欄は 0 ではない** —— 「その母集団では測っていない」か「集計に入れない列」。
どちらかは `caveats.csv` と `provenance.csv` に書いてある。

**`measured_by` を読むこと。** 全数 437,561層の表は**抽出側が測った数**で、
ここに DB は無い。合うのは**こちらの転記と向こうの足し算**だけで、
**DB が壊れていても合う。**

**マスクの向きを層の種類から決めてはいけない**(2026-10-26 の訂正)——
普通のフォルダーにも `init 0` のマスクが在る。`body_initcolor.csv` の交差表で読む。

## 点検(**走らせずに commit しない**)

```bash
./venv/bin/python tools/census/build_census.py              # 指摘 0 でなければ commit しない
./venv/bin/python tools/census/control_name_leak.py census  # 点検が本当に鳴るか(C1〜C5)
./venv/bin/python tools/census/known_cell_supplier_table.py # 供給元の手書きの表と突き合わせ
./venv/bin/python tools/census/build_ui.py                  # 画面(集計の後。これも点検つき)
```
