# MoE Plan

## 目的

カテゴリの異なるペアを無差別に混ぜることで、別カテゴリの推論品質を落とす問題を避ける。
まずは full MoE 実装ではなく、カテゴリ分離されたデータ・評価・ルーティングの入口を作る。

## 分類軸

### Dataset/source 軸

既存の原稿・抽出由来の違いを表す。

- `housei`
- `lineart`
- `ako5`
- `kurip`
- `unknown`

この軸は、スキャン品質、ラフ濃度、線画の癖、ペア抽出誤差の出方に影響する。

### Content 軸

絵として何を描いているかを表す。

- `character_bust`
- `character_fullbody`
- `natural_scenery`
- `manmade_object`
- `panel_or_background`
- `line_fragment`
- `unknown`

この軸は、将来の expert 分割候補。今回の refined kurip では `line_fragment` が多く、
キャラクター部位の学習に混ぜると悪影響が出やすい可能性がある。

## 当面の方針

1. 既存ペアにカテゴリ metadata を付ける
2. dataset/source 軸と content 軸を分けて扱う
3. 学習リストをカテゴリ別に生成できるようにする
4. カテゴリごとの固定評価セットを作る
5. まずは single model のカテゴリ混合比を制御して干渉を測る
6. 干渉が確認できたカテゴリだけ specialist / router / MoE に進める

## 実装ステップ

### Step 1: Pair metadata schema

CSV/JSONL でペアごとの metadata を作る。

候補ファイル:

- `dataset/pairs_480/pair_metadata.csv`

最低限の列:

- `name`
- `split`
- `rough_path`
- `line_path`
- `dataset_source`
- `content_category`
- `pair_quality`
- `alignment_quality`
- `line_ink`
- `rough_std`
- `notes`

`pair_quality` はまず以下でよい。

- `high`
- `medium`
- `low`
- `reject`

`alignment_quality` は以下でよい。

- `same_coordinate`
- `locally_refined`
- `rough_shifted`
- `uncertain`

### Step 2: Initial auto-labeling

まず filename と既存リストから機械的にラベルを付ける。

- `housei_*` -> `housei`
- `lineart_*` -> `lineart`
- `ako5r_*` -> `ako5`
- `kurip*` / `kuripr_*` -> `kurip`

content は自動判定が難しいため、初期値は控えめにする。

- refined kurip の細い断片中心 -> `line_fragment`
- 明確な顔・上半身 -> `character_bust`
- 全身が入るもの -> `character_fullbody`
- それ以外 -> `unknown`

初期は手動レビュー用 CSV を出し、人間が必要分だけ修正する。

### Step 3: Category-aware list builder

metadata から学習リストを作る。

例:

- base general only
- base + kurip refined line_fragment 5%
- base + kurip refined character_bust 10%
- base + ako5 character_bust

出力:

- `dataset/pairs_480/lists/*.txt`
- `dataset/pairs_480/line_mix/<experiment_name>/`

現行 `scripts/train.py` は単一 `line_dir` 前提なので、当面は mixed line dir を生成して使う。

### Step 4: Category evaluation sets

固定8サンプルだけでは干渉を見落とす。
カテゴリ別の小さい評価セットを作る。

- `eval_fixed_general`
- `eval_housei_character`
- `eval_lineart_character`
- `eval_ako5_regions`
- `eval_kurip_refined`

各セットで見る指標:

- F1@2px
- chamfer
- ink_ratio
- 目視 montage

### Step 5: Interference experiments

まず full MoE ではなく、single model の混合比で干渉を見る。

候補:

1. `shape1` baseline
2. `shape1 + kurip_refined line_fragment 5%`
3. `shape1 + kurip_refined line_fragment 10%`
4. `shape1 + kurip_refined character_bust only`
5. `shape1 + kurip_refined all`

評価観点:

- general fixed が悪化しないか
- kurip refined が改善するか
- line_fragment が character 出力を汚さないか
- 黒太り・網点・吸われる現象が増えないか

### Step 6: Router / expert entry

干渉が確認できたら、MoE 入口に進む。

初期案:

- dataset/source gate は既存 `lineart/dataset_gate.py` 系を継続利用
- content gate は metadata と VLM/軽量分類器の両方を検討
- まずは hard routing で specialist checkpoint を切り替える
- full MoE layer は最後に検討する

優先順位:

1. category-aware dataset
2. category-aware evaluation
3. hard routed specialists
4. learned gate
5. full MoE architecture

## 直近の次アクション

1. `pair_metadata.csv` 生成スクリプトを作る
2. 既存リストから dataset/source と alignment_quality を自動付与する
3. refined kurip 123件を `line_fragment` / `character_*` / `unknown` に分けるレビューCSVを作る
4. metadata から mixed list と line dir を生成する list builder を作る
5. refined kurip のカテゴリ別 montage を出して、手動ラベルの基準を固める
