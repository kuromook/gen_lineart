# Badrough Lucy-Thin Threshold Output Notes

Date: 2026-07-24

## 位置づけ

このメモは `work_log.md` とは別に、ako5/ako6 系の bad-rough 混入対応から派生した
`lucy_thin` 系調整の成果と限界を記録するもの。

今回の調整は MoE 本体の実装というより、MoE / router の前提になるデータ健全化、
および ako 系 bad-rough 混入後のモデル挙動確認に属する。

## 成果

`ako5_` 由来の解釈不能 rough を除外した clean list で再学習し、さらに
`lucy_thin` 系に対して以下を試した。

- ink / width 制約
- relaxed lucy_thin 再学習
- output threshold 後処理
- threshold-aware differentiable loss

その結果、ざらつきは残るものの、以下の性質を持つ出力を捉えた。

- halo がかなり少ない
- 背景 haze が少ない
- gray stroke ではなく白黒出力に寄せられる
- clean retrain の過剰インクよりは制御された出力になる

特に参考候補:

- soft output:
  `badrough_lucy_thin_threshold_e3_lucy_thin_thresh_mid`
- black/white output:
  `badrough_lucy_thin_threshold_e3_lucy_thin_thresh_mid_post_threshold52`

主要 metric:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| old lucy_thin | 0.4095 | 4.536 | 1.252 | 0.3198 | 0.5962 |
| clean lucy_thin | 0.4227 | 4.457 | 2.789 | 0.3005 | 0.7272 |
| relaxed_b | 0.4163 | 4.553 | 2.236 | 0.3021 | 0.6821 |
| relaxed_b threshold52 | 0.4096 | 4.667 | 1.917 | 0.3024 | 0.6466 |
| thresh_mid | 0.4211 | 4.471 | 2.204 | 0.3041 | 0.6965 |
| thresh_mid threshold52 | 0.4172 | 4.551 | 1.922 | 0.3064 | 0.6652 |

成果物:

- `results/compare_badrough_lucy_thin_threshold_e3.png`
- `results/fixed_output_metrics_badrough_lucy_thin_threshold_e3_compare.csv`
- `results/haze_uncertainty_metrics_badrough_lucy_thin_threshold_e3_compare.csv`

## 限界

実用線画としてはまだ遠い。

最大の問題は threshold 後のざらつき。調査では、threshold 後に大量の微小連結成分が
出ることが確認された。

例:

- `mid_t52` の connected components 平均: 約 2673 個
- そのうち 12 px 未満: 約 2584 個

これは、soft 出力上では灰色の不確実なストロークとして見えていたものが、
threshold 後に点・短い傷・粒状断片として現れるため。

単純な小連結成分除去も試したが、有効な細線まで消えて recall / F1 が落ちた。
したがって、単純な postprocess だけでは根本解決しない。

## 解釈

今回の成果は「完成した線画モデル」ではなく、以下を確認したことに価値がある。

- bad-rough 除外後の clean retrain は挙動を変える
- `lucy_thin` は halo 抑制と白黒出力の方向では最も見込みがある
- threshold-aware loss は、threshold 後の metric を改善できる
- ただし、下絵の揺らぎ・重ね描き線をそのまま候補線として拾うため、
  threshold 後にざらつく

## 次の方針

この系統の loss 微調整はいったん止める。

次にリソースを振るべき方向:

- preprocess 研究
  - 下絵からノイズを減らす
  - gray / haze を抑える
  - 重ね描き線・揺らぎを減らす
  - モデルが一本線として解釈しやすい rough 入力に変換する
- dataset 拡張
  - clean で rough/line 一致度の高いペア数を増やす
  - old `ako5_` の同座標グリッド抽出ではなく、region-matched /
    alignment-aware な抽出を優先する
  - badrough / black-fill / metric-special-case QC を継続する

現時点では、`thresh_mid` / `thresh_mid threshold52` は参考成果として保持するが、
production 昇格はしない。
