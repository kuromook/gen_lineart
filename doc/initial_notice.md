# Track E: 「線画らしさ」の審判を作る

作成: 2026-09-16 JST / ブランチ: `aesthetic-judge`
起案: `../lineart-pair-signal/doc/track_proposal_aesthetic_judge_20260916.md`
前身: Track D `../lineart-pair-signal`(診断完了。ペアデータが寄与しない理由は
生成の目的関数にある、と結論。残件として規模・潜在感度・manga_lineの空白化を調査中)
並走: Track C `../lineart-stroke-selection`(ストローク選択、進行中)

このファイルは「現状・次の一手・運用ルール」だけを短く保つ。
実験の時系列はこのtreeの`doc/work_log.md`に追記すること。
このtreeはTrack Dのコミットから分岐しているので、`doc/work_log.md`と`doc/CURRENT.md`には
Track Dまでの履歴が入っている。**このtrackの記録は末尾に追記する。**

## この track が存在する理由

Track D が測った通り、**画素の一致を目的にする生成の学習は、条件画像の再現に収束する**。
位置の合ったペアだけで学習しても、条件画像に無い GT の線を描く割合は対照と同じ約10%だった。

一方で、出力の線の **27〜50% は GT にも条件画像にも近くない**。凍結 UNet の癖(クロスハッチ、
描き足された髪の線)である。**「ベースモデルの美」はすでに出力に混ざっているが、誰も制御して
いない。** 本プロジェクトはそれを一貫して「幻覚」として扱ってきたが、常に悪いのかは確かめて
いない。

GT と条件画像が食い違う場所は、どの目的関数も定義していない空白地帯である。ここを
「人が見て良いと思う方」に倒せるなら、Track C の選択にも、生成の微調整にも使える。

## この track の問い

**「線画として良い」を、人の判断を再現する形で採点できるか。**

できたら接続先は3つ(この順に安い):
1. Track C の同点決着(GT が答えを持たない場所の、残す/消すの判断)
2. 推論時の空間制御(条件画像が当てにならない領域だけ ControlNet 強度を下げる)
3. 報酬による微調整(DRaFT/ReFL 系)

## 【2026-09-17】収集を中断している — 材料がまだ存在しない

比較UIを公開し、ユーザーが実際に触ったうえでの指摘: **壊れていて何だか分からないものを
「美しい」とは判断できない。** それは美の軸ではなく「壊れている/解釈不能」の軸であり、
そこを美の軸として扱おうとしても成立しない。意味のない単語を吐き続けるGPTの出力を見ても
何も言えないのと同じ。

材料の側にも証拠がある。段階1のために用意した3候補のうち、線画として成立しているのは
**前処理器単体だけ**で、残りは学習済みモデルの出力(中間調の割合 `preproc` 0.072 に対し
`control` 0.503、`w02_1000` 0.259、モンタージュでも灰色の浮き彫り)。つまり300組の大半は
「きれいな線画 対 壊れた何か」であり、そこから得られるのは既存の軸で計算できる答えだけ。

**診断: 軸が悪いのではなく、比較に足る材料がまだ無い。** 解釈可能な出力を出せるモデルが
一つも無い以上、この track の段階1は先に進めない。UIと300組(`results/comparison_pairs_20260916/`、
https://claude.ai/code/artifact/b8f99c23-0980-42a4-b4ab-6e8c7c1ad1fb )はそのまま残す。
**再開の条件: 両方が解釈可能な出力の組が作れるようになったとき。**

引き継ぎ先: `../lineart-stroke-grammar`(線を単位にして「この配置に馴染むか」を判別する)。
そちらが解釈可能性を測れるようになれば、この track の材料も揃う。

## 次の一手(順序つき) — 【上記により保留中】

### 1. 判定データを集める(最初の一手)

素材はすでにある。Track D のスナップショット出力、前処理器単体、削除オラクル、GT。

- 比較2枚組を作る。**同一タイル**の異なる出力同士を並べ、どちらが線画として良いかを選ぶ
- 比較 UI は Artifact(保存機能つき)。1回数秒、目標 300〜500 組
- **同じ組を再提示して本人内の一貫性を測る。** 採点器の上限はここで決まる
- 「どちらとも言えない」を選べるようにし、曖昧な組は学習から外す
- 組の作り方に偏りを入れない。時点・群・タイルを層化して選ぶ(片方が常に白い等の手がかりで
  選べてしまうと、採点器は質感だけを覚える)

### 2. 採点器を作り、検証する

- まず既存の線画らしさ軸(`near_white_frac`/`line_width_p50`/`fill_ratio`/連続性/交点統計)を
  特徴量にした軽い線形モデル。次に小さな CNN か CLIP 特徴＋線形
- 評価は **hold-out の人の判断をどれだけ再現するか**だけ
- **合格条件(先に決めてある)**: hold-out 一致率 ≥ 本人内一貫性 × 0.85、かつ既存の単独指標
  (f1 / near_white / fill)より明確に高いこと
- **落ちたら 3 に進まない。** 指標に3回裏切られたプロジェクトなので、ここで止まれることが設計の一部

### 3. 用途に接続する(合格後)

Track C の同点決着 → 推論時の空間制御 → 報酬微調整。学習が要るのは最後だけ。

## この track の範囲外

- **採点器が段階2に合格するまで、生成モデルの再学習はしない**
- GT の置き換えではない。GT が答えを持つ場所は今まで通り GT に従う
- 「美」の一般論ではなく、**このデータセットに対するこのユーザーの判断**の再現。
  判定者が1人である以上、汎化は主張しない
- Track D の残件(規模の曲線・潜在感度・manga_lineの空白化)は Track D 側で進行中。
  こちらでは触らない

## 素材の所在

- 比較2枚組の材料(すべて 480px、同一の192タイルに揃っている):
  - `../lineart-pair-signal/results/h34_alignment_probe_20260915/{aligned,control}/step_*/`(5時点×2群)
  - `../lineart-controlnet-sd15-refine/results/controlnet_lora_manga_consistency_w0.2_snapshot_probe_20260914/step_*/`(11時点)
- GT: `../lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line/`(192)
- 条件画像: `../lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/conditioning/`(lineart_coarse)、
  `../lineart-controlnet-sd15-refine/data/holdout_lineart_family_rough_manga_line/`(manga_line)
- 削除オラクル: `../lineart-stroke-selection/results/oracle_visual_check_20260913/oracle_renders/`
- 測定道具(この tree にも同じものがある): `tools/evaluation/` の
  `stroke_churn.py`、`output_vs_condition_proximity.py`、`pair_alignment_strata.py`、
  `fixed_t_validation_loss.py`、`measure_lineart_profile.py`

## 運用ルール(Track A〜D で確立、引き継ぎ)

- **数値だけで判断しない。** 必ず目視モンタージュを作り、画像の正確なパスを数値と併記する。
  画像は`/tmp`ではなく`results/`配下
- **`gt_bsds_f1`を報告するときは条件画像自体のスコアを基準線として併記**(共通基盤 lesson 6)
- **プールを跨いで平均しない**(`lineart`系と`housei`系は別タスク)
- **学習損失で進捗を判断しない**(lesson 7)。スナップショットごとに hold-out を採点する
- **条件画像は学習用ペアでも中身を測る**(lesson 8)。評価セットだけ見ると分布のずれを見落とす
- 大きな学習・抽出は必ずスモークテストしてから本番投入
- 長時間 GPU ジョブは月〜木の4日連続バッチが標準枠
- バックグラウンド実行は`setsid`で完全に切り離す(PPID=1を確認)。`pkill -f <スクリプト名>`は
  自分のシェルにも一致して自滅するので PID を特定して`kill`する
- 線を単位にする処理で共有の`evaluate_stroke_stability.skeletonize()`を使わない
  (2px幅が残り交点判定が壊れる)。skimage の1px細線化＋crossing number を使う
- `bipartite_match_f1` は特定のタイルで数分〜12分以上かかることがある。多数タイルに一括適用
  するときはタイルごとにサブプロセス化してタイムアウトを設ける
  (`tools/evaluation/vae_roundtrip_score.py` の方式)
