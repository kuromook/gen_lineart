# Track B: SDXL路線 — 下絵との乖離(条件忠実度)

作成: 2026-09-06 JST / ブランチ: `controlnet-sdxl-fidelity`
前身: `lineart-controlnet-realpairs`(クロスハッチ脱却を達成して終了)
起案: `doc/track_proposal_20260906.md`(このtrackにも同梱)
前身の全経緯: `doc/track_controlnet_realpairs_work_log.md`

このファイルは「現状・次の一手・運用ルール」だけを短く保つ。
実験の時系列は`doc/work_log.md`に追記すること。

## 課題

SDXLは**クロスハッチを出さない**——`sdxl_trained`の`orientation_entropy`
0.6366は全11モデル中最小で、GTの0.7117よりさらに低い(ストローク方向が
揃っている)。SD1.5路線が苦労して手に入れた性質を最初から持っている。

しかし**下絵との乖離が激しい**(gt_bsds_f1 0.0945で全11モデル中最下位)。
さらにSD1.5系と異なり**csを上げると悪化する**(cs1.0で0.0945 →
cs2.0で0.0581)ため、SD1.5路線で効いたcsレバーが効かない。
クロスハッチとは**別種の失敗**。

## 最優先で確認すべきこと — 解像度の不一致

前trackのSDXL実験は、**学習・推論とも512×512で実行されていた**:

- `scripts/train_controlnet_sdxl.py`: `--resolution` 既定値 **512**
- `scripts/infer_controlnet_sdxl.py`: `--resolution` 既定値 **512**
- `experiments/run_controlnet_lora_sdxl_20260829.sh` および
  `..._sdxl_manga_20260830.sh`: どちらも`--resolution`を**上書きしていない**

これらの既定値はSD1.5時代のスクリプトからの引き継ぎだが、ベースの
`animagine-xl-3.1`は**1024ネイティブのSDXL**。SDXLを512で回すと
ベースの学習分布から外れ、SDXL固有のmicro-conditioning
(`original_size`/`target_size`/`crops_coords_top_left`の埋め込み)とも
噛み合わず、出力が大きく劣化することが広く知られている。

**つまり、これまでのSDXLの評価はSDXLの実力を測れていない可能性が高い。**
「SDXLは下絵から乖離する」という上記の観測は**保留扱い**とし、
まず1024で土俵を揃えるところから始めること。

## 次の一手(優先度順)

1. **1024で再学習・再推論**(最優先)。VRAM 12GBで1024のSDXL ControlNet
   LoRAが回るかは要検証(512で24.5s/stepだった)。勾配チェックポイント・
   バッチサイズ・grad accumの調整。1024学習が非現実的なら
   「学習512/推論1024」など段階的な切り分けも検討。
2. 土俵を揃えたうえで条件忠実度を`condition_roundtrip_fidelity.py`の
   `roundtrip_ssim`/`roundtrip_bsds_f1`で測り、乖離が残るか再判定。
3. 乖離が残る場合、`Eugeoter/noob-sdxl-controlnet-*`の適性
   (どんなデータで学習されたControlNetか)を疑う。

## 参照

- ベース: `animagine-xl-3.1`
  (`/home/sh1/.cache/huggingface/hub/models--cagliostrolab--animagine-xl-3.1/...`)
- ControlNet init: `Eugeoter/noob-sdxl-controlnet-lineart_anime` および
  `...-manga_line`
- 既存チェックポイント(512学習、比較の基準として):
  `../lineart-controlnet-realpairs/checkpoints/controlnet_lora_sdxl_20260829/final`、
  `..._sdxl_manga_20260830/final`
- データ: `../lineart-controlnet-realpairs/data/`(約1GB、`.gitignore`対象)。
  ペアデータ一式(`train_list.txt` 8,467件、`line/`、`rough_manga_line/`、
  `captions.csv`ほか)。**このtreeにはまだ複製していない**——このtrackで
  実作業を始める時点で、参照のままにするか`cp -a`で複製するかを判断する
  こと(ユーザー判断、2026-09-06)。前trackは終了済みでいずれ整理される
  可能性があるので、本格的に学習を回すなら複製を推奨。
  Track A(`../lineart-controlnet-sd15-refine/data/`)は複製済み。
  なお共通基盤側にある新しいv3プール(8,798タイル)には**移行しない**方針
  (v2との差は再ベースライン化に見合わないというユーザー判断、2026-09-06)。

## 運用ルール

- 大きな学習・抽出は必ずスモークテストしてから本番投入
- バックグラウンド実行は`nohup ... & disown`+PPID=1確認で完全に切り離す
- 数値指標だけで判断しない、必ず目視モンタージュを作り、**その画像の
  正確なパスを数値と一緒に併記する**
- **モデル評価を単一の`controlnet_conditioning_scale`だけで行わない**
  (SD1.5系では既定値1.0がハッチ支配領域でモデル間の差を潰していた。
  SDXLはcs上昇で悪化する挙動なので、低め側も含めて振ること)
- **`orientation_entropy`単体で判断しない**。ハッチ網目と滑らかなベタ塗りの
  境界を区別できない。`line_width_p50`(GT≈3.7)と`ink_ratio`(GT≈0.035)を
  必ず併記する。**SDXLは特に注意**——`sdxl_trained`は
  `line_width_p50` 20.64(GTの5.5倍)でベタ塗り寄りの出力になっている
- 実験の結論が出たら、その場で`results/`の生成物を要否判断する
