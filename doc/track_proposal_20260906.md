# Track Proposal: SD1.5 Refinement and SDXL Condition-Fidelity (2026-09-06)

起案日: 2026-09-06 JST
起案元: `lineart-controlnet-realpairs` track(目的達成につき一区切り)
経緯の詳細: `../lineart-controlnet-realpairs/doc/work_log.md`

## 背景 — 終了するtrackの結論

`lineart-controlnet-realpairs`は、公開ControlNet
(`control_v11p_sd15s2_lineart_anime`)のLoRAファインチューンが、GTの
クリーンな線画ではなく**密なクロスハッチ**を生成する問題を追っていた。

**結論: 主因は学習側ではなく、推論時の`controlnet_conditioning_scale`
(以下cs)が既定値1.0のままだったこと**だった。

| | cs=1.0(旧既定) | cs=3.5 | GT参照 |
|---|---|---|---|
| gt_bsds_f1 | 0.1411 | **0.2337** | — |
| ink_ratio | 0.2992 | **0.0772** | 0.0353 |
| line_width_p50 | 3.94 | **3.34** | 3.72 |

到達構成: `lineart-controlnet-realpairs/checkpoints/controlnet_lora_manga_consistency_20260904/final`
を`--controlnet-conditioning-scale 3.5`で推論。

**このプロジェクト全体に効く教訓が2つある**:

1. **cs=1.0での比較は信用できない**。そこはハルシネーションが支配的な
   領域で、全11モデルがf1 0.13〜0.15に潰れて**モデル間の差が消える**。
   実際、cs=1.0では`sd15_lineart`(0.1558)が1位で`manga_trained`(0.1293)は
   下位だったが、各モデルの最適csで測り直すと順位が大きく入れ替わった
   (`lineart-controlnet-realpairs/results/cs_reeval_20260906/scores.csv`、
   11モデル×5スケール。cs=1.0列は過去の値と完全一致することを
   アンカー検証済み)。**ControlNet系の評価は必ず複数csで行うこと**。
2. **`orientation_entropy`単体で判断しない**。この指標は「ハッチ網目」と
   「滑らかなベタ塗りの境界」を区別できない。実例として、再評価で
   `coarse_trained`はf1 0.2101と好成績に見えたが`line_width_p50`が40.92
   (GTの11倍)で、実体は線ではなくベタ塗りだった。
   `line_width_p50`(GT≈3.7)と`ink_ratio`(GT≈0.035)を必ず併記する。

上記のcs問題は解決済みだが、**性質の異なる2つの課題が残った**。両者は
原因も対処も別系統なので、それぞれ独立trackとして起案する。

---

## Track A: SD1.5路線 — 最良構成からの詰め

**課題**: 上記の最良構成でも、GTの白背景・黒線に対し**背景がグレー・線も
グレー寄り**になる。cs4.0以上に上げると線自体が薄れて消える(cs5.0で
剣がほぼ消失)ため、csをさらに上げる方向では埋まらない。

**出発点**: `controlnet_lora_manga_consistency_20260904` @ cs3.5。
これは`scripts/train_controlnet_consistency.py`(x0推定をVAEデコードして
GT画像とのSobelエッジ一致度L1損失を補助項に追加)で学習したもの。

**優先度順の調査方向**:

1. **consistency損失のハイパラスイープ**(最有力)。効果があることは
   確定したが、`consistency_weight=0.1`・`consistency_max_timestep=200`の
   **1点しか試していない**。重みとtimestep範囲を振る余地が大きい。
2. **InnerControl方式への拡張**。文献
   ([arxiv 2507.02321](https://arxiv.org/abs/2507.02321)、
   [github.com/ControlGenAI/InnerControl](https://github.com/ControlGenAI/InnerControl))は
   ControlNet++系の一致度損失が「最終デノイズステップのみ」に適用される
   限界を指摘し、全timestepの中間UNet特徴からの条件再構成に拡張して改善を
   報告している。実装済みのconsistency損失は`max_timestep=200`=まさに
   その「最終ステップ付近のみ」版に相当するので、直接の発展形になる。
3. **出力の二値性/コントラストへの介入**。グレー残差そのものへの対処。
   サンプラー・CFG・あるいは学習側の目的関数で扱う。

**やってはいけないこと(検証済み)**:
- **ベースUNetにLoRAを足さない**。仮説6として検証したが、ベースライン
  より全csで劣ったうえ、**csレバー自体を壊した**(他のmanga系と違い
  csを上げると単調悪化)。UNet凍結は維持する。
- **LoRA rankを上げない**。rank32は唯一cs上昇で単調悪化し、最適域で
  最下位級だった。
- **ネガティブプロンプトで"hatching"等を禁止しない**。逆効果で、
  ハッチの代わりにベタ塗りへ逃げる(`line_width_p50` 14.85)。

---

## Track B: SDXL路線 — 下絵との乖離

**課題**: SDXLは**クロスハッチを出さない**(`sdxl_trained`の
`orientation_entropy` 0.6366は全11モデル中最小で、GTの0.7117よりさらに
低い=ストローク方向が揃っている)。しかし**下絵との乖離が激しい**
(f1 0.0945で最下位)。さらにSD1.5系と異なり**csを上げると悪化する**
(cs1.0で0.0945 → cs2.0で0.0581)ため、Track Aで効いたcsレバーが効かない。
クロスハッチとは**別種の失敗**として切り離して扱う。

**最優先で確認すべき仮説 — 解像度の不一致**:

`lineart-controlnet-realpairs`のSDXL実験は、**学習・推論とも512×512で
実行されていた**:

- `scripts/train_controlnet_sdxl.py`: `--resolution` 既定値 **512**
- `scripts/infer_controlnet_sdxl.py`: `--resolution` 既定値 **512**
- `experiments/run_controlnet_lora_sdxl_20260829.sh` および
  `..._sdxl_manga_20260830.sh`: どちらも`--resolution`を**上書きしていない**

これらの既定値はSD1.5時代のスクリプトから引き継いだものだが、
ベースの`animagine-xl-3.1`は**1024ネイティブのSDXL**である。SDXLを512で
回すと、ベースの学習分布から外れるうえ、SDXL固有のmicro-conditioning
(`original_size`/`target_size`/`crops_coords_top_left`の埋め込み)とも
噛み合わず、出力が大きく劣化することが広く知られている。

**つまり、これまでのSDXLの評価はSDXLの実力を測れていない可能性が高い。**
Track Bは、まず1024で学習・推論をやり直して土俵を揃えるところから始める
べきで、それ以前の「SDXLは下絵から乖離する」という観測は保留扱いにする。

**調査方向**:

1. **1024で再学習・再推論**(最優先、上記の理由)。VRAM 12GBで1024の
   SDXL ControlNet LoRAが回るかは要検証(512で24.5s/stepだった)。
   勾配チェックポイント・バッチサイズ・grad accumの調整、または
   1024学習が非現実的なら「学習512/推論1024」など段階的な切り分けも検討。
2. 土俵を揃えたうえで、条件忠実度を`condition_roundtrip_fidelity.py`の
   `roundtrip_ssim`/`roundtrip_bsds_f1`で測り、乖離が残るかを再判定する。
3. 乖離が残る場合、`Eugeoter/noob-sdxl-controlnet-*`の適性
   (どんなデータで学習されたControlNetか)を疑う。

**SDXLに期待する理由**: クロスハッチを出さないという性質は、Track Aが
苦労して手に入れたものを最初から持っているということでもある。条件忠実度
さえ確保できれば、SD1.5路線より筋が良い可能性がある。

---

## 運用面の提案

- **worktree運用にするか**: 既存の`lineart-cleanup-refiner` /
  `lineart-halo-loss` / `lineart-router-moe`はgit worktreeだが、
  `lineart-controlnet-realpairs`は**worktreeではない素のディレクトリ**
  として作られていた(ペアデータとチェックポイントを実体コピーする設計に
  したため)。新2trackをどちらの方式にするかは要判断。
  なお`doc/worktree_policy.md`は2026-07-20時点の記述で、統合ブランチを
  `MoE`としているが現在の共通基盤は`diffusion`ブランチ。更新が必要。
- **引き継ぎ文書の構成**: `lineart-controlnet-realpairs`では
  `inbox/initial_notice.md`(現状・次の一手・運用ルールのみ、117行)と
  `doc/work_log.md`(時系列ログ、857行)に分離する形に落ち着いた。
  引き継ぎ文書が作業ログを兼ねると900行超まで肥大化して機能しなくなった
  ためで、新trackでも最初からこの2本立てにすることを推奨する。
- **`doc/CURRENT.md`の扱い**: 共通基盤の`doc/CURRENT.md`は2026-07-31が
  最終更新で、Active Goalが生データ抽出のままになっている。今回の2track
  起案を反映するかどうかは要判断(このファイルは共通基盤の権威ある
  ステータス文書なので、起案側で勝手に書き換えていない)。
