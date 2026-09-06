# Track B: SDXL路線 — 下絵との乖離(条件忠実度)

作成: 2026-09-06 JST / 更新: 2026-09-06 / ブランチ: `controlnet-sdxl-fidelity`
前身: `lineart-controlnet-realpairs`(クロスハッチ脱却を達成して終了)
起案: `doc/track_proposal_20260906.md`(このtrackにも同梱)
前身の全経緯: `doc/track_controlnet_realpairs_work_log.md`

このファイルは「現状・次の一手・運用ルール」だけを短く保つ。
実験の時系列は`doc/work_log.md`に追記すること。

## 現状 (2026-09-06 時点)

**起案時の課題設定は、解像度スイープでほぼ全面的に書き換わった。**
以下は測定済みの事実(`doc/work_log.md` 2026-09-06 の各項、
`results/resolution_sweep_20260906/`、`results/grey_source_cross_20260906/`)。

### 起案時の前提のうち、覆ったもの

- 「**SDXLはcsを上げると悪化する**(csレバーが効かない)」は**誤り**。
  それは`sdxl_trained`(=512学習のanime LoRA)1本の性質だった。素の
  ControlNetはcs2.0〜3.0で最良になる。csレバーはSDXLでも生きている。
- 「**SDXLは下絵との乖離が激しい**(f1 0.0945で最下位)」も**保留のまま**では
  なく**否定**。乖離していたのは512で学習したLoRAであって、素の
  ControlNetは1024で最良 **f1 0.2582** に達する。これは前trackの
  11モデル最良(`manga_consistency` 0.2337)を上回る。

### 新たに確定した事実

- **解像度とcsは交互作用する。単独では効かない。**
  `anime_base`のf1は cs1.0 では 512→1024 でほぼ横ばい(0.2263→0.2341)、
  cs2.0 では単調改善(0.2416→0.2568)。紙の白も同様で、1024かつcs2.0が
  揃って初めて`near_white`が3.0%→81.5%に跳ぶ。
- **グレー背景は`lineart_anime` ControlNet固有**(2×2交差で帰属確定。
  条件画像を替えても解像度を変えても動かない)。1024×高csでのみ外れる。
- **`manga_line` ControlNetは使えない**。512/1024とも縞状・ブロック状に
  破綻し、剣タイルで剣が出ない。`near_white` 0.878〜0.914は「きれいな白紙」
  ではなく「ほぼ空白のページ」。
- **512学習のLoRA2本は、全解像度・全csで素のControlNetに劣る。**
- **12GBで1024学習は可能**。`scripts/cache_sdxl_conditioning.py`で
  VAE/テキストエンコーダを事前計算する構成で **8.05GiB / 9.35s/step**
  (素の512は9.79GiB / 9.85s/step)。10エポック≒55時間。

### 現時点の最良構成 (fine-tuneなし)

`Eugeoter/noob-sdxl-controlnet-lineart_anime` を **LoRAなし・1024・cs2.5**で
推論。f1 0.2582 / ink 0.0447 / 線幅 3.06 / near_white 77.6%
(GT: 0.0353 / 3.72 / 94.8%)。目視 `results/resolution_sweep_20260906/
montage_anime_base_cs_plateau.png`。

**留保**: この出力は大部分が**条件画像の描き直し**である(条件画像の時点で
すでに線が引かれている)。f1 0.2582は「線画を生成できた」ではなく
「条件に忠実である」ことを示す。GTとの残差は忠実度ではなく
**線の取捨選択と強弱づけ**という別の仕事。

## 次の一手(優先度順)

1. **月曜(09-07)投入: 1024でのLoRA再学習**。
   `experiments/run_controlnet_lora_sdxl_1024_20260907.sh`(既定
   `VARIANT=anime`)。キャッシュ生成→スモーク→10エポック→csラダー評価まで
   連結済み。素のControlNetを同一ジョブ内で再測定して並走比較する。
   **問いは「1024で学習し直せば、素のControlNet(0.2582)を超えられるか」**。
   超えられなければ「このデータとレシピでは素のControlNetを改善しない」が
   結論になり、それも成果。
2. 上の結果次第で、残差である**線の取捨選択・強弱づけ**に課題を移す。
   条件忠実度は`condition_roundtrip_fidelity.py`の
   `roundtrip_ssim`/`roundtrip_bsds_f1`で測る(このtreeのデータ配置で
   動くよう`--rough-dir`/`--line-dir`/`--line-prefix`を追加済み)。
3. 条件画像側の伸びしろ。2×2で**`cnAnime` × `condManga`の1024が f1 0.2438**と、
   同ControlNetにcoarseを与えた0.2341を上回った。前trackは各ControlNetに
   その前処理を組み合わせる流儀だったので、この交差は未探索。

## 参照

- ベース: `animagine-xl-3.1`
  (`/home/sh1/.cache/huggingface/hub/models--cagliostrolab--animagine-xl-3.1/...`)
- ControlNet init: `Eugeoter/noob-sdxl-controlnet-lineart_anime`(採用) および
  `...-manga_line`(不採用、上記)
- 既存チェックポイント(512学習、劣ることが確定済み。歴史的アンカー用):
  `checkpoints/controlnet_lora_sdxl_20260829/final`、
  `..._sdxl_manga_20260830/final`
- データ: `data/`(約1GB、`.gitignore`対象)。前trackから`cp -a`で複製済み
  (2026-09-06)。`train_list.txt` 8,467件、`line/`、`rough_lineart_coarse/`、
  `rough_manga_line/`、`captions.csv`ほか。
  v3プール(8,798タイル)には**移行しない**方針(ユーザー判断、2026-09-06)。
- 1024学習用キャッシュ: `data/cache_sdxl_1024/`(約4.9GB、`.gitignore`対象)。
  `data/line`のlatentとテキスト埋め込みのみ。条件画像に依存しないので
  anime/manga両variantで共用できる。

## 運用ルール

- 大きな学習・抽出は必ずスモークテストしてから本番投入
  (数日規模のバッチが引数ミスで落ちるのが最も高くつく失敗)
- **長時間GPUジョブは月〜木の4日連続バッチが標準枠**(ユーザーは平日ほぼ
  在室しない)。週末は対話的な作業——レビュー・分析・短い実験・次の
  長時間バッチを決めること——に充てる。数日かかること自体は問題ない。
  チェックポイントは3日目のクラッシュで全損しない頻度で取る
- バックグラウンド実行は`setsid`で完全に切り離す(PPID=1を確認)。
  `pkill -f <スクリプト名>`は**自分のシェルのコマンドラインにも一致して
  自滅する**ので、PIDを特定して`kill`すること(2026-09-06に踏んだ)
- 数値指標だけで判断しない、必ず目視モンタージュを作り、**その画像の
  正確なパスを数値と一緒に併記する**。画像は`/tmp`ではなく
  **作業フォルダ(`results/`)に置く**(ユーザー要望、2026-09-06)
- **モデル評価を単一の`controlnet_conditioning_scale`だけで行わない**。
  さらに**解像度とcsは交互作用する**ので、片方だけ振っても効果は見えない
- **`orientation_entropy`単体で判断しない**。`line_width_p50`(GT≈3.7)と
  `ink_ratio`(GT≈0.035)を必ず併記する
- **`gt_bsds_f1`単体でも判断しない**。紙の白さを一切見ていない。
  `bg_mode`/`near_white_frac`/`midtone_frac`(GT: 255 / 94.8% / 1.8%)を
  必ず併記する。実例2件——灰色一色の`anime_base` 512/cs1.0が
  スイープ最良のf1 0.2263を白3.1%で取り、ほぼ空白の`manga_base`
  1024/cs1.0が0.2121を白87.8%で取った
- **`near_white`単体でも判断しない**。「きれいだから白い」のか
  「描けていないから白い」のかを区別できない。線幅と目視で裏を取る
- 実験の結論が出たら、その場で`results/`の生成物を要否判断する
