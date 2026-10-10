# Track J (ベタ塗り) 作業ログ

## 2026-10-09: track 開設

共通基盤セッションが、ユーザー決定(2026-10-09)により開設。`doc/CURRENT.md` の
Next Actions 項目5(2026-09-13 に「今はやらない」と判断されたベタ塗り)の再開にあたる。

経緯: GPU が空き(RTX 3060、158MiB、0%)、保留中の実験に GPU 律速のものが無いことを
共通基盤が確認したうえで、「GPU が構造的に有利な残差はベタ塗りだけ」という理由で選択された。
前処理器の `fill_ratio` は 0.0%、GT は 24.5%。

開設時点の確認済みの事実(共通基盤が実測):

- `dataset/pairs_480/train/line` の出典別枚数: `housei` 5,070、`ako5` 6,380、`ako5r` 418、
  `kurip` 2,280、`lineart` 168。`test/line`: `housei` 1,596、`lineart` 24
- したがって**ベタのプール 11,868 枚は未学習・未評価**で、ホールドアウトは `housei` test 1,596 が
  既に別出典として分離されている。`ako5` にテスト分割は無い
- プロファイル(`pool_inventory.csv` 由来): `ako5` ベタ25.6%/ink 0.0920/空白tile 33.3%、
  `housei` 28.8%/0.0744/26.7%、`lineart` 6.5%/0.0475/0.0%

**次の一手**: `doc/initial_notice.md` の第1段(ベースライン測定)の事前登録。走らせる前に書く。

---

## 2026-10-09: 第1段「ベースライン測定」事前登録(走らせる前に記載)

`doc/initial_notice.md` 第1段の事前登録。**以下はすべて数値を一切見る前に確定したもの**で、
合否基準の事後緩和はしない。読んだもの: 共通基盤 `../lineart/doc/CURRENT.md` の
lesson 5/6/8/9/10(および 11/12/13)、Known Tool Traps 全件、Next Actions 項目5、
`../lineart-controlnet-sdxl-fidelity/doc/pool_inventory.md` と接触シート
`results/pool_inventory_20260911/contact/shared_train_{ako5,housei}.png`。

### 0. 接触シートから先に言っておくこと(数値には出ない)

`ako5` と `housei` は「ベタが多い」で一括りにできない。**ベタの形が違う**:

- `ako5`: 大きな黒面。**タイル全体がベタの内側に入っている例が複数ある**
  (`ako5_030_12_03`, `ako5_054_06_02` など)。この手のタイルは 480px の窓の中に
  境界が無く、ラフを見ても「ここが黒いか」はタイル内の情報では原理的に決まらない
- `housei`: 髪の毛・影の鋭い黒楔が主で、面は小さい。加えて**グレーの濃淡**を含むタイルがある
  (`housei_011_12_15`, `housei_008_22_14`)。128 の二値化でインクに落ちるかは微妙

この2点は第2段の設計に効く(タイル内に手がかりの無いタイルが一定数ある)。第1段では
**層別の第3軸として「GTベタがタイル面積の90%超」を別に数える**ことで記録だけ残す。

### 1. 用語の確定 — `fill_ratio` は面積率ではない

共通基盤 `measure_lineart_profile.fill_ratio` は **インク画素のうち** 距離変換 > 4.0px
(=幅8px超)に属する割合。`ako5` の 25.6% は「ページの25.6%が黒」ではない。
`ink_ratio` 0.0920 と掛けて**面積では約2.4%**。`doc/initial_notice.md` と
共通基盤 Next Actions 項目5 の「GT は 24.5%」はこのインク正規化値であり、
**面積率と混同してはならない**。以後、区別して両方報告する:

- `fill_ratio` (インク正規化、既存指標、棚卸しと比較可能)
- `fill_area_frac` (タイル面積に対する割合、**新規**。IoU の分母側の感覚を与える)

### 2. ベタマスクの定義(主)と感度(副)

グレースケール 480x480 に対し:

```
ink  = gray < 128                          # プロジェクト共通の閾値
dist = cv2.distanceTransform(ink, DIST_L2, 3)
core = ink & (dist > 4.0)                  # fill_ratio が数えている集合そのもの
fill = binary_dilation(core, disk(4)) & ink   # 主マスク = ink の disk(4) オープニング
```

`core` だけだとベタの外周4pxが欠けるため、**採点は `fill`**(黒面の実体)で行う。
感度として半径 3 と 6 も併記する。結論が半径4に依存していないことを示すため。

### 3. 評価セット — 結果を見る前に機械的に固定

`ako5` にテスト分割が無いので、**ページ単位**で切る。規則(今ここで確定):

```
held_out(pool, page) := int(sha1(f"{pool}_{page}").hexdigest()[:8], 16) % 5 == 0
```

結果(この規則から機械的に出たもの。ページ名だけで決まり、画像を一切見ていない):

| プール | 全tile | ページ数 | held ページ | held tile |
|---|---:|---:|---|---:|
| `ako5` | 5,962 | 53 | 011 017 021 026 043 064 | **689** (11.6%) |
| `ako5r` | 418 | 34 | 0047 0049 0060 | **23** (5.5%) |
| `housei` | test 1,596 | 001 004 009 | (既に別分割) | **1,596** |

**第1段の採点対象 = housei test 1,596 + ako5 held 689 + ako5r held 23 = 2,308 枚。**
`ako5r` は n=23 なので**参考値としてのみ報告し、主張には使わない**(事前に明言)。
この分割は第2段でもそのまま使う。`housei` train は 002-018 の15ページで test と
ページが重ならないことを確認済み。分割の解決は `tools/evaluation/gt_paths.resolve()`。

別に、**モデルを使わない記述統計**として GT 側のベタ分布を 11,868 枚全件で測る
(これは仮説を持たない記述で、採点ではない)。

### 4. 腕(arm)

| arm | 中身 |
|---|---|
| `blank` | 何も塗らない。**禁止事項3の自明ベースライン**。IoU は構成上 0 |
| `all_black` | 全面黒。過剰塗りの上限側。器具確認 IC2 を兼ねる |
| `preproc` | `controlnet_aux.LineartAnimeDetector`(= 本プロジェクトの前処理器)をラフに通し、**白線黒地を反転してから**マスク抽出 |
| `msgan` | `checkpoints/combined_koma_lucy_mild_msgan_20260729`。`in_channels=2` を確認済みで、aux 無しでは動かない。`experiments/run_combined_koma_lucy_mild_msgan_20260729.sh` と同じ3段(atari aux → `preprocess_atari_aux.py --mode lucy_mild` → `--aux-dir` 付き推論、すべて `--autocontrast`)で再現する |

`msgan` を aux 無し1chで回すのは**別のモデルを測ることになる**ので禁止。

### 5. 器具確認 — 既知の値のセル。これが通るまで他は一切読まない

**IC1(既知の値の再現。アンカーは自分のパイプラインが作っていない他 track の CSV)**:
`../lineart-controlnet-sdxl-fidelity/results/pool_inventory_20260911/inventory.csv` の
`fill_ratio` を自分の実装で再現する。対象7セル(shared プールのみ):
`train:ako5` 0.2564 / `train:housei` 0.2881 / `test:housei` 0.2240 /
`train:kurip` 0.2030 / `train:ako5r` 0.2650 / `train:lineart` 0.0650 / `test:lineart` 0.0330。

- 第1脚(配管の検査): 当該スクリプトと同じ 120枚シード抽出を再現し、同じ cv2 プリミティブ
  (`DIST_L2`, maskSize 3)で計算。**合格 = 7セル中5セル以上が ±0.002 以内**
- シード抽出が再現できない場合(`iterdir()` 順に依存するため)のフォールバックは事前に決めておく:
  **全件平均を出し、棚卸し値が自分の 120枚相当の95%区間 (±2 SE) に入ること**。
  どちらに倒れたかは結果と一緒に明記する
- 第2脚(実装非依存性の感度): scipy の厳密 EDT で同じタイルを計算し、第1脚と **±0.02 以内**で
  一致すること。cv2 の 3x3 近似の産物でないことの確認。これは感度であって合否ではない

**IC2(解析的に答えが決まっているセル)**: `all_black` の予測マスクはタイル全面になるので
`IoU = |G| / |P|` が恒等的に成り立つはず。**合格 = 全タイルで |IoU − |G|/|P|| ≤ 0.01**。

**IC3**: `blank` は全タイルで `pred_fill_frac == 0.0` かつ `IoU == 0.0`(厳密に)。

### 6. 陰性対照 — 「向き」ではなく、起こりうる具体的な誤りを1つずつ

Known Tool Traps の2026-10-09の項(Track G が落ちた件)に従い、**実在の呼び出し側が犯しうる
混同を1行につき1つ**、それぞれ期待値を付けて実装する。

| # | 名指しした誤り | 期待値(事前) |
|---|---|---|
| NC1 | **前処理器の極性を反転し忘れる**。`LineartAnimeDetector` は白線黒地を返すので、そのまま `gray<128` すると背景がインクになる(共通基盤に実例: 8,451/8,467 = 99.8% が ink_ratio 0.95 超) | `pred_fill_frac ≥ 0.90`、かつ IoU が `all_black` と区別できない(差 ≤ 0.02) |
| NC2 | **出力と GT の対応を1つずらす**(ソート済みリストで i 番の出力を i+1 番の GT に当てる、巡回) | 全 arm で micro IoU ≤ 0.10、median IoU ≤ 0.05。**各 arm の正対応 IoU が自分自身の NC2 値を明確に上回らない限り、その IoU は位置合わせの証拠ではない** |
| NC3 | **予測マスクを転置する**(lesson 12 の形。タイルは正方形なので黙って通る) | micro IoU が NC2 の水準に落ちる(NC2 との差 ≤ 0.03) |
| NC4 | **50px ずらす**(位置特定の感度。合否バーは置かない。記録のみ) | NC2 と正対応の中間 |

### 7. 合否基準 — 第1段の結論を事前に固定する

主値は **`housei` test と `ako5` held をそれぞれ別に**、**「通常層 かつ GT にベタがある」タイル**
(GT `ink_ratio ≥ 0.002` かつ `gt_fill_area_frac > 0.001`)での micro IoU(画素プール)。
macro(タイル平均)と全体平均も併記するが**主値にしない**。プール間の平均は取らない(lesson 6)。

- **C1(この track の前提の検査)**: 「前処理器は原理上塗れない」。
  期待どおり = `preproc` の micro IoU < 0.05 **かつ** 平均 `pred_fill_area_frac` が GT の 0.1倍未満。
  **`preproc` の micro IoU が 0.20 以上なら、この track の前提そのものが誤りで、それを見出しに報告し、
  第2段は登録しない**(ユーザー判断に戻す)
- **C2(既存モデルの検査)**: `msgan`。
  - micro IoU ≥ 0.20 → 「既に部分的に解けている」。第2段の比較対象は `blank` ではなく `msgan` になる
  - micro IoU < 0.05 → 「何もしていない」。第2段のベースライン3種は notice 記載のまま
  - 0.05–0.20 → **判定保留**とだけ報告する。目視を見てから第2段を登録する(数字を見てから基準を動かさないため、
    どちらに転んだかの解釈だけは目視に委ねると今ここで決めておく)
- **C3**: すべての数値は `blank`(IoU 0、`fill_area_frac` 誤差 = GT 全量)と `preproc` の両方を
  同じ表に併記する。片方でも欠けた表は報告しない(禁止事項3)
- **C4**: 自分の陰性対照に落ちた arm は**一切読まない**

### 8. 副次(主張には使わない)

- 線側: `tools/evaluation/stroke_decomposition.py` を `preproc` を条件画像として
  `b_recall` / `c_survival` / `neither_ink` で。lesson 9 のとおり `b_recall` 単独では読まない
- `bsds`: **主値にしない**。走らせる場合は必ず `tools/evaluation/bsds_guard.py` 経由、
  300s キャップ、対象は 200枚までの部分抽出、タイムアウト件数を必ず併記する
- 目視図: `results/baseline_fill_20261009/` 配下(`/tmp` は使わない)。
  (a) 固定シードで無作為抽出した層別コンタクトシート(数値を見る前に抽出枚数と seed を固定: seed=20261009、各層8枚)、
  (b) IoU の上下端サンプル(**「選んだ」ことを図のタイトルに明記する**)

### 9. この第1段でやらないこと

- diffusion / ControlNet でベタを描かせる試み(禁止事項1)。反証器具の登録もまだしない
- CNN+GAN の再学習(禁止事項2)。`msgan` は**既存チェックポイントの推論のみ**
- 学習は一切しない。GPU 推論と CPU 集計だけ

### 補遺 (2026-10-09、スモークテスト24枚のみを見た時点。本走の数値は未取得)

スモークで**器具の問題が1件**見つかったので、事前登録を**厳しくする方向に**1点修正する。
緩和ではない。見出しの結論を自分に有利にしない方向の変更であることを明記しておく。

**問題**: `LineartAnimeDetector` の出力は白線黒地で、しかも**線が薄い**。ako5 の内容のある
タイルで `>128` の画素は 0.01% 程度しかない(反転して `gray<128` でインク判定すると
インクがほぼゼロになる)。これは本プロジェクトが既に持っている条件画像
(`dataset/pairs_480/train/rough_lineart_anime_20260824/`)と同じ分布で、**検出器の仕様**であって
不具合ではない。共通基盤の `stroke_decomposition.py` が固定閾値でなく `edge_map()` を
使っているのも同じ理由と思われる。

**何が問題か**: このまま閾値128で `preproc` 腕のベタマスクを作ると、**空になるのは当たり前**で、
C1(「前処理器は原理上塗れない」)が**同語反復になる**。それは前処理器を測っておらず、
自分の閾値を測っている(「新しい数値が出たら、これは道具を測っていないか」)。

**修正(追加する二値化の梯子)**: `preproc` 腕を次の4通りで採点し、**全部を報告する**。

| 変種 | 中身 |
|---|---|
| `preproc@128` | 当初登録どおり(プロジェクト共通の閾値)。参考値に降格 |
| `preproc@32` | Known Tool Traps にある「素朴な >32」。はるかに寛大 |
| `preproc@otsu` | タイルごとの大津閾値 |
| `preproc@budget` | **最も寛大**。出力の明るい順に、**GT の `ink_ratio` と同じ枚数**の画素をインクと認める。つまり「置き場所だけ検出器に決めさせ、インク量は正解を渡す」 |

**C1 の判定は `preproc` の4変種のうち最良のもので行う。** `@budget` は検出器に正しいインク量を
渡したうえで「そのインクが 8px 幅の面を作るか」を問うので、「エッジ検出器は原理上塗れない」の
真の検査になる。どれか1つでも micro IoU 0.20 以上なら、前提は誤りとして見出しに出す。

他の腕・陰性対照・合否基準は登録どおりで変更しない。`msgan` の出力は濃い二値なので 128 のまま。

### 中断時点の状態 (2026-10-09、利用上限により中断。**採点値はまだ1つも出ていない**)

事前登録と器具は完成、腕の実体化は走行中。**合否判定はまだ出していない。**

**完了**

- **IC1 合格(7/7)**: `results/baseline_fill_20261009/ic1_fill_ratio.csv`。棚卸し CSV の
  `fill_ratio` 7セル全部を自分の実装で小数4桁まで再現(ako5 0.2564、housei 0.2879 など)。
  第2脚も確認: cv2 の 3x3 近似と scipy の厳密 EDT は**距離そのものは最大 8.56px 食い違う**が、
  4px 閾値を跨ぐ画素は 87万インク画素中 **0個**(閾値 ±0.5px の帯にいるのはインクの 3.13%、
  そこでの誤差は最大 0.18px)。つまり `fill_ratio` はこの近似に依存していない
- **評価リスト確定**: `results/baseline_fill_20261009/lists/`。登録どおり
  housei_test 1,596 / ako5_held 689 / ako5r_held 23 = **2,308 枚**。規則と held ページは
  `lists/SPLIT.md`
- **GT 記述統計(モデル非依存、13,046枚)**: `results/baseline_fill_20261009/gt_fill_profile.csv`

  | プール | n | ink | fill_ratio | **fill面積率** | near-blank% | ベタ有り% | 全面ベタ% |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | `ako5` | 5,962 | 0.0845 | 0.2759 | **0.0757** | 31.3% | 64.4% | 0.5% |
  | `housei` train | 5,070 | 0.0523 | 0.2223 | **0.0454** | 26.6% | 53.3% | 0.1% |
  | `housei` test | 1,596 | 0.0534 | 0.2174 | **0.0460** | 41.1% | 49.3% | 0.0% |
  | `ako5r` | 418 | 0.0628 | 0.2617 | **0.0404** | 0.0% | 100.0% | 0.0% |

  読み方: 「GT は 24.5%」はインク正規化値で、**面積では 4〜8%**。`housei` test は
  **41.1% が near-blank**(棚卸しの 38.3% より高い)で、「ベタを探す対象がある」タイルは
  約半分しかない。層別が必須なのはこの数字のため
- **GT ベタマスクの目視確認**: `results/baseline_fill_20261009/montage_gtcheck_housei_normal_fill.png`
  (seed 20261009 の無作為8枚、normal_fill 層)。赤のオーバーレイは髪のベタに正確に載っており、
  **マスク定義は妥当**。同じ図で前処理器の列はほぼ空(予告どおりだが、採点値はまだ出していない)
- **道具一式**: `tools/evaluation/{fill_mask,check_fill_instrument,build_fill_eval_lists,`
  `profile_fill_pools,score_fill_baselines,report_fill_baselines,montage_fill}.py`、
  `experiments/run_fill_baselines_20261009.sh`
- **msgan 腕の再現方法を確定**: チェックポイントは `in_channels=2` で、aux 無しでは別のモデルになる。
  atari aux → `preprocess_atari_aux --mode lucy_mild` → `--aux-dir` 付き推論の3段を
  `run_combined_koma_lucy_mild_msgan_20260729.sh` どおりに再現(24枚でスモーク済み)

**走行中(中断時点)**

`experiments/run_fill_baselines_20261009.sh`(ログ `logs/fill_baselines_20261009.log`)。
housei_test の preproc 1,596 と atari 1,596 は完了、**`lucy_mild` が律速**(約0.5s/枚、
CPU単スレッド)。残り: housei の lucy と msgan、ako5_held 689、ako5r_held 23。見積り約1時間。
**次セッションはまずログで完了を確認すること。**

**次の一手(順に)**

1. `score_fill_baselines.py --list <各リスト> --split <test|train>` を3本
2. `report_fill_baselines.py` — IC2/IC3 と NC1-NC4 を先に印字し、落ちた腕は読まない。
   そのうえで C1/C2 の判定が自動で出る(基準は登録済みで、コードに埋めてある)
3. `montage_fill.py --mode random`(各プール × 3層)と `--mode extreme`
4. 共通基盤へ通達 `../lineart/inbox/`。第2段の登録は第1段の数値が出てから

**事実の訂正1件(数値を見る前の、数え直しによるもの)**: briefing と共通基盤 Next Actions 項目5 の
**「11,868 枚」は `ako5r` 418 を `ako5` の 6,380 の中で二重計上し、`housei` test 1,596 を
落としている**。実数は train 側 11,450(ako5 5,962 + ako5r 418 + housei 5,070)、
housei test 1,596 を足して **13,046 枚**。未学習・未評価という主張自体は変わらない。

---

## 2026-10-10: 第1段「ベースライン測定」結果

全数値: `results/baseline_fill_20261009/report.txt`、per-tile は同ディレクトリの
`per_tile_{housei_test,ako5_held,ako5r_held}.csv`。器具確認と陰性対照を先に通し、
登録どおり**落ちた腕は読んでいない**。

### 器具確認・陰性対照(主結果より先に)

- **IC1 合格 7/7**(前セッション記載)。**IC2 合格**: `all_black` の IoU は全 2,308 枚で
  解析解 `|G|/|P|` と 0.01 以内。**IC3 合格**: `blank` は全枚 `pred_px=0`
- **NC1(極性の反転忘れ)**: 予測どおり壊れる。fill 面積 0.9989〜0.9999、`all_black` との
  IoU 差 0.0001 以下。この腕は `all_black` と区別できない
- **NC2(1つずらした対応)で `msgan` が落ちた**。housei 真値 0.0163 に対し NC2 0.0123、
  **ako5 は真値 0.0152 に対し NC2 0.0161 で逆転**。つまり `msgan` のベタ重なりは
  位置合わせの証拠ではない。**登録 C4 により、この腕の IoU は読まない**
- **NC3(転置)が `preproc@budget` を落とした。これが今回いちばん重要な器具所見。**
  `@budget` は唯一 NC2 を超えた前処理器変種だった(housei 0.0860 対 0.0592、
  ako5 0.1618 対 0.0615)。ところが**転置してもスコアがほぼ動かない**
  (housei 0.0854、ako5 0.1606、真値との差 0.0006 / 0.0012)。
  転置で失われるのは位置情報だけなので、**この重なりは「どこを塗るか」ではなく
  「どれだけ塗るか」で稼いでいる**。しかもその「どれだけ」は `@budget` の定義上
  GT から渡した量である。macro IoU が micro のはるかに下(0.0108 / 0.0230)なのも同じ所見で、
  値は少数の大面積タイルが作っている。**よって `@budget` も読まない**

陰性対照を通過した前処理器変種は**1つも無い**。

### 主結果(normal層 かつ GT にベタ有り、micro IoU、半径4、プール別)

| arm | housei_test (n=782) | ako5_held (n=474) |
|---|---:|---:|
| **`blank`(塗らない)** | **0.0000** | **0.0000** |
| `preproc@128` | 0.0000 | 0.0000 |
| `preproc@32` | 0.0002 | 0.0001 |
| `preproc@otsu` | 0.0002 | 0.0003 |
| `preproc@budget`(陰性対照に落ちた) | (0.0860) | (0.1618) |
| `msgan`(陰性対照に落ちた) | (0.0163) | (0.0152) |
| `all_black`(全面塗り) | 0.0939 | 0.1201 |
| GT ベタ面積率 | 0.0939 | 0.1201 |

`ako5r_held` は n=23 の参考値(登録どおり主張には使わない): `msgan` 0.0664 が
NC2 0.0179 を超え転置でも落ちる唯一のセルだが、23枚では何も言えない。

### 登録どおりの判定

- **C1 は予期どおり**。前処理器は4変種とも**塗らない**。陰性対照を通った最良変種で
  micro IoU 0.0002、ベタ面積は GT の 0〜2%。**インク量を GT から渡しても(`@budget`)
  場所の情報は出てこない**(転置で落ちない)。「エッジ検出器は原理上塗れない」は
  同語反復ではなく、**寛大な4通りの二値化と GT 由来のインク予算を与えたうえで確認された**
- **C2: `msgan` は何もしていない**(両プールとも 0.05 未満、かつ NC2 に落ちている)。
  lineart プールで学習した現行の採用モデルを ベタのプールに通すと、**線を太らせるだけで
  面は塗らない**(目視: `montage_ako5_held_normal_fill_random.png` の msgan 列)。
  **第2段のベースライン3種は notice 記載のまま**
- **C3**: 全表に `blank` を併記済み。`blank` の IoU は 0、ベタ面積誤差は GT 全量
  (housei 0.0939 / ako5 0.1201)

### この track の前提について

**前提は維持された。** 前処理器の天井は IoU 0.0002 で、`blank` と区別がつかない。
同時に、**上限側の自明解 `all_black` が 0.0939 / 0.1201 を取る**ことも記録しておく。
これは「GT のベタ面積率そのもの」で、**面積率を当てるだけの腕が取れる値**である。
第2段で `all_black` を超えられない数値は、場所を当てていない。

### 半径感度(結論は半径4の産物ではない)

GT ベタ面積は r=3 → r=4 → r=6 で housei 0.0975 / 0.0939 / 0.0879、
ako5 0.1250 / 0.1201 / 0.1109。**4→6 で 93.6% / 92.3% が残る**ので、
マスクの大半は 12px より太いインクで、半径の選び方に依存していない。
前処理器の値はどの半径でも 0.0006 以下(`report.txt` の感度表)。

### 目視で分かって数値に出ないこと(第2段に効く)

`results/baseline_fill_20261009/` の接触シート(seed 20261009 の無作為抽出、
`montage_{housei_test,ako5_held}_{normal_fill,near_blank}_random.png`、
および選抜であることを図中に明記した `*_msgan_extreme.png`):

- **半径4のベタマスクには「太い輪郭線」が相当量入る。** ako5 の無作為6枚のうち3枚は
  赤が面ではなく太い線に載っている。幅では面と太線を分けられない
  (`fill_ratio` が `line_width_p50` を補う指標として導入されたのと同じ限界が、
  一段上の粒度で再発している)。**ベタを「面」の単位で扱う必要があるという
  2026-10-10 のユーザー所見と整合する**
- `msgan` の出力は**線が太るだけ**で、閉領域が黒くならない。ベタ列の赤は
  太った線の断片であって面ではない
- 前処理器の出力はラフの薄い鉛筆線をそのまま拾っており、面の内側は白いまま

### 残課題(第1段の範囲で未実施)

副次として登録した `stroke_decomposition`(線側)と `bsds`(部分抽出)は未実施。
主結果と判定には不要で、第2段の登録前に必要なら回す。

## 2026-10-10 Design check before stage 2: does a fill region close inside the input window?

User, on reading the stage-1 montage finding: "パネル単位からコマ単位にしよう、ということだね?"
That is the right reading of the problem, so it was measured rather than
agreed to. `tools/evaluation/check_region_closure.py`, GT line art only, no
model, no arm: fill-mask connected components of >=200px, counting those that
touch the window border -- a region whose own boundary is outside the input is
one the model is never shown whole.

| set | n | median min side | stroke px | components | border-cut | fill AREA in cut comps |
|---|---:|---:|---:|---:|---:|---:|
| housei 480 grid tile | 1596 | 480 | 1.91 | 2547 | 47.3% | 87.6% |
| housei koma panel | 74 | 1876 | 1.91 | 2119 | 5.6% | 26.4% |
| ako5 480 grid tile | 689 | 480 | 2.74 | 1631 | 52.3% | 92.9% |
| ako5ver2 koma panel | 171 | 1576 | 2.74 | 8247 | 8.9% | 48.5% |

The two units are at the *same drawing scale* -- the stroke widths match to
two decimals per source, which is what makes the rows comparable. Nothing was
resized.

`stroke_px` had to be measured over ink *outside* the fill mask. Over all ink
it reads 27.15px on an ako5 koma panel against 5.48px on a 480 tile, which
looks like a 5x scale difference and is not one: the median is sitting in the
middle of the solid areas, i.e. it is reporting the thing under test. The
first version of this check did exactly that and the scale-free argument built
on it was wrong. Same family as the transpose control: a statistic that moves
with the quantity being measured is not a control on it.

Caveat, stated rather than resolved: the koma window is both *larger* (3-4x per
side) and *aligned to the drawing's own boundaries*. This measurement does not
separate the two. It does not need to for the decision -- the koma gives both
-- but no claim is made here that alignment alone is what does it.

Not in the numbers, from the sheets: on the 480 grid the red (closed) mask
lands on eye outlines and a pair of glasses -- thick *lines* at that zoom, not
areas -- while the hair mass is blue in every tile that contains one. On the
koma panels the red regions are hair and clothing, which is what a person
means by ベタ. Visual:
  results/region_closure_20261010/montage_ako5_koma_panels.png
  results/region_closure_20261010/montage_ako5_480_grid.png
  results/region_closure_20261010/montage_housei_koma_panels.png
  results/region_closure_20261010/montage_housei_480_grid.png

Consequences for stage 2, to be settled in its registration, not here:

1. Stage 1's `all_black` 0.0939 / 0.1201 is a property of the 480 window. On a
   koma panel, with its blank margins, it will be lower. Baselines must be
   re-measured at whatever unit stage 2 adopts; they do not carry across.
2. Paired koma panels already exist, no extraction needed: ako5ver2 171,
   4th 155, fitness 109, housei 74, hamlabi 58, gakuen 46, clip_pairs_v3 5018
   (+ psd_line 756 line-only, unusable as pairs).
3. The koma pipeline deliberately went the *other* way in 2026-07-26 -- panel
   -> subregion -> per-subregion alignment refinement, 58 -> 75 -> 85 tiles for
   housei -- because `ink_range` rejected 80.9% of panel candidates and
   panel-level alignment was too loose for line position. Both gates are line
   gates. A mostly-blank panel is a valid "do not paint" label, and a few px of
   misalignment moves a region's boundary without changing which region is
   black. Whether that tolerance claim holds is itself measurable and should be
   registered as an instrument check, not assumed.

## 2026-10-10 Stage 2 pre-registration: the region model

Registered after stage 1's numbers existed, as the initial notice requires,
and written before any stage-2 arm has been run. The criteria below are fixed
and are not to be loosened afterwards; if one of them turns out to have been
the wrong criterion, the registered outcome is "failed C_n, and here is why
the criterion was wrong", not a new threshold.

### What was measured first, and what it removed from the design

The user's reading of the stage-1 montage finding -- "パネル単位からコマ単位に" --
is adopted: **the unit of input is the koma panel, not the 480px window.**
The closure table in the previous entry is the evidence (88-93% of fill area
on a 480 grid tile belongs to a region whose boundary is outside the window,
against 26-49% on a koma panel, at matched drawing scale).

A second measurement, made before writing this, removed the architecture that
"the unit of judgement is the 面" most obviously suggests. If a 面 is an
enclosed area, it should be recoverable by taking the strokes and flood-filling
what they enclose. It is not:

    line_ink = ink & ~fill ; regions = components of ~line_ink (4-conn)

    ako5ver2 koma   171 panels    202 regions >=500px    1.2 per panel
    housei koma      74 panels     90 regions >=500px    1.2 per panel
    clip_pairs_v3   400 panels   2295 regions >=500px    5.7 per panel

and of those, **0 / 0 / 17 label as filled**, with 100% / 100% / 95% of the
fill area landing outside any filled region. The reason is visible per panel:
the largest background component is **100.0% of the background in every one of
six panels inspected** -- the strokes do not close. Even keeping the fills as
ink, the largest component is 82-100%.

So a region proposer has no ground truth to be scored against on the GT side,
before one even asks whether the rough could produce one. **Consequence: the
面 is the fill connected component itself** (8-connected, 28-289 per ako5
panel, 5-71 of them >=500px), the model stays pixel-wise, and only the *metric*
is region-wise. This is the opposite of what stage 2 would have been designed
as yesterday, and it cost one afternoon instead of the GPU slot.
Tool: `tools/evaluation/region_decompose.py`, CSVs in
`results/region_closure_20261010/`.

### Corpus and split, fixed now

Corpus: `dataset/regions_clip_pairs_v3_koma_panels_20260826`, **5018 paired koma
panels, 57 work_ids, 1122 work-pages**, native scale, stroke width 1.91px
(identical to the 480 pools, so the radius-4 fill definition carries over
unchanged and stage 1's masks remain comparable).

The six small per-source panel sets (ako5ver2 171, 4th 155, fitness 109,
housei 74, hamlabi 58, gakuen 46) are **excluded**: clip_pairs_v3 contains
`011_2017_ako5`, `066_2024_housei`, `069_2024b_fitness`, `063_2023b_hamlabi`
and three `4th` works, so they are duplicate material under a different
extraction and mixing them would put the same drawing on both sides.

Split, mechanical, on the **work family**, not the page:

    family(work_id) := third underscore field, lowercased, "-..." and a
                       trailing "_color" dropped, trailing digits (plus one
                       optional letter) stripped; anything starting with
                       "ako" collapses to "ako"
    held_out(family) := int(sha1(f"fillv2_{family}").hexdigest()[:8], 16) % 5 == 0

33 families; **9 held out = 969 panels (19.3%), 189 work-pages**; 4049 train.
Held: avbu 42, ben 89, china 69, gakusai 383, hero 93, housei 45, pink 107,
poolsideidle 79, stoptime 62.

Why the family and why the ako collapse: **whether a character's hair is
blacked in is a property of that character**, so the leak that would inflate
this task is character identity, not page adjacency. `ako3/ako5/.../ako10`,
`akogoods`, `akokate`, `akocult` are one series and go to one side. Noted
rather than hidden: that puts 1825 panels (36%) in train and makes the held
side smaller than a page-level split would. The rule was chosen on panel
counts only; no score has been computed on either side.

### Arms

Non-learned, all mandatory, all reported beside every model number:

  `blank`            paint nothing
  `all_black`        paint everything
  `preproc@{128,32,otsu,budget}`  LineartAnimeDetector, the stage-1 ladder.
                     Its stage-1 ceiling was micro IoU 0.0002; it is here
                     because the initial notice forbids a number without it,
                     and because the koma unit is a new unit for it too.
  `rough_dark`       the rough's own dark ink, through `fill_mask`. **This is
                     the arm that matters.** If roughs already mark hair with
                     a dark scribble, the task is mostly solved with no model
                     and the honest report says so.
  `rough_dark@budget` the same, handed the GT's fill pixel count -- the
                     placement-only reading, as in stage 1.
  `freq_table`       lesson 14's required control: no training, a frequency
                     table over one cheap descriptor of each `rough_dark`
                     component (16 bins of normalised y-centre x 4 bins of
                     log area), fitted on train, predicting filled/not.

Learned, the GPU slot:

  `M0`  window-only. 480px native-scale crop of the rough -> fill mask.
        The current unit, carried forward so the koma unit has something to be
        better than.
  `M1`  window + context. The same crop, plus the whole koma downscaled to
        480 as additional input channels and the crop's normalised position.
        This is the registered experiment: **does seeing the whole 面 help
        predict it?** M1 - M0 is the answer, and NC4 below is what says
        whether the answer came from the context stream.

Both are plain supervised segmentation (U-Net, BCE + Dice, no adversarial
term, `scripts/train_fill_seg.py`, to be written). Outputs are stitched back
to the panel at native scale and scored there; the window is an implementation
detail of the network, not the unit of judgement.

**On prohibition 2 (do not return to CNN+GAN).** Read as: do not return to the
`cleanup`/msgan family -- adversarial line *generation* judged on line metrics,
which is what lesson 5 closed. A non-adversarial mask predictor for a
different target, judged region-wise, is a different thing. The falsifier is
stated so this cannot drift: **if an adversarial term is added, or if the
output is ever read as line art rather than as a mask, the prohibition has
been violated.** Flagged for the user rather than decided quietly.

Prohibition 1 stands untouched: no diffusion and no ControlNet arm appears
here, and none may be added until an instrument that would falsify Track D's
hypothesis 4 / lesson 8 is built and registered first.

### Instrument checks (run and reported before any arm is read)

  IC1  Re-score stage 1's `housei_test` and `ako5_held` 480 tiles through the
       new scorer. The pixel micro-IoU must reproduce
       0.0000 / 0.0002 / 0.0163 / 0.0939 (housei) and
       0.0000 / 0.0001 / 0.0152 / 0.1201 (ako5) for blank / best preproc /
       msgan / all_black, exactly. A new scorer that cannot reproduce the old
       numbers where the two overlap is not measuring the same thing.
  IC2  `fill_ratio` on clip_pairs_v3 panels against
       `../lineart-controlnet-sdxl-fidelity` pool_inventory's published value
       for the ako5 work -- one leg anchored outside this pipeline (lesson 12).
  IC3  GT fed in as the prediction: region-F1 1.0, pixel IoU 1.0, by both the
       matcher and the stitcher.
  IC4  **The stitching check.** Cut a panel into windows, reassemble, and
       compare against the panel PNG *on disk* -- not against my own tiling.
       Must be bit-identical. This is the check lesson 12 says to anchor
       outside; an off-by-one in the window grid is invisible in every metric.
  IC5  The region matcher on a hand-computed synthetic case: two discs, one
       displaced by a known amount, expected matches written down first.

### Negative controls -- one named error each, with its expected score

  NC1  polarity: the rough read as white-on-black. Expected: fill area ->
       ~1.0, indistinguishable from `all_black`, as in stage 1.
  NC2  off-by-one: panel i's prediction scored against panel i+1's GT.
       Expected: every arm strictly above it. An arm that is not is not read.
  NC3  **horizontal flip**, not transpose. Panels are not square, so a
       transpose cannot pass silently here -- but a left-right flip can, and
       it is the error this material actually invites, since manga pages are
       read right-to-left and flipped copies exist. Expected: a localised arm
       loses most of its score; an arm that does not is painting the right
       *amount* in the wrong *place*, which is what caught `preproc@budget`.
  NC4  **context shuffle**: M1 run with another panel's global context.
       No pass bar on its own; it is the reading rule for C2.
  NC5  50px shift. Sensitivity only, no bar.

### Pass criteria, fixed

  C1  **Does anything predict fills at all?** The best arm's region-F1 must
      exceed `blank`, `all_black` and its own NC2 by >= 0.05 absolute, on the
      held-out side, reported per family group, never pooled into one average.
  C2  **Does the koma unit earn its cost?** M1 - M0 >= 0.03 region-F1 on the
      held-out side AND M1's NC4 (shuffled context) must drop by >= 0.03.
      Both, or the context stream is not what produced the difference.
  C3  **Is a model needed?** If the best non-learned arm is within 0.05
      region-F1 of the best model, the registered conclusion is *"no learned
      model is needed for this"* and that is the headline, not a footnote.
      Given lesson 6 this is the outcome to expect, and C3 exists so that
      expecting it does not become a reason to avoid measuring it.
  C4  Any arm failing NC2 or NC3 is excluded from every verdict, as
      `preproc@budget` and `msgan` were in stage 1.
  C5  Pixel micro-IoU is reported beside every region-F1, and the near-blank
      and normal strata separately. No overall average is a headline number.

Stop rules: if IC4 fails, nothing is trained. If `rough_dark` alone satisfies
C1 and C3, the GPU slot is released rather than spent confirming it.

Training discipline (lesson 7): `--eval-snapshot-steps` on from the first run,
holdout scored at every snapshot, `PYTHONUNBUFFERED=1`. Training loss is not
read as evidence of anything. Slot: the next 月-木 window, 2026-10-12 to
2026-10-15.

### What stage 2 will not do

Line refinement. Diffusion or ControlNet in any role. Any claim about ツヤベタ
(the texture case in the user's 2026-10-10 observation) -- the fill mask at
radius 4 does not separate it from plain ベタ, and no arm here is asked to.
Any number without `blank` and the preprocessor beside it.

## 2026-10-10 (amendment, before any stage-2 arm has run) User's correction, and the two proposals measured

Three things from the user, in the order they change the registration.

### 1. Prohibition 2 lifted for this stage

"いったん禁止を解除する承認をするのは問題ないつもりだ". The learned arms M0/M1
stand as registered. The falsifier written yesterday stays: an adversarial
term, or reading the output as line art, puts it back inside the prohibition.

### 2. `rough_dark` is demoted from contender to ceiling -- the user's account, measured

The user's account: solid fills concentrate in 髪 / 衣類(トップス) /
タイツ(スパッツ); those are decided before the rough is drawn, so **the artist
does not mark them**. A fill-like mark in a rough indicates *indecision*,
which is the opposite of what my arm assumed.

Measured on the aligned koma pairs, `tools/evaluation/rough_fill_signal.py`:

| source | base fill rate | rough mark | its area | **GT fill covered** | precision | lift |
|---|---:|---|---:|---:|---:|---:|
| ako5ver2 (167 panels) | 10.28% | rough<128 | 2.11% | **4.2%** | 20.4% | 1.98 |
| | | rough<64 | 0.15% | 0.3% | 21.2% | 2.06 |
| | | area-like | 0.57% | 1.2% | 21.7% | 2.11 |
| housei (73 panels) | 3.21% | rough<128 | 0.60% | **1.6%** | 8.4% | 2.62 |
| | | area-like | 0.09% | 0.2% | 7.7% | 2.40 |

**96% of the solid-fill area has no dark mark under it in the rough**, and the
strictest reading leaves 99.7%. So the recall ceiling of any "the rough
already has it" arm is 4.2%, and the arm I registered yesterday as the one
most likely to win cannot win. It stays in, reported as that ceiling.

Visible directly in `results/region_closure_20261010/pair_check.png`: in all
four pairs the hair and clothing that are solid black in the line art are
drawn in the rough as **an outline with white inside**.

What is *not* settled: the user's claim has a sign -- a fill-like mark should
mark indecision, so fill rate under such a mark should be *below* base, lift
< 1. Measured lift is about 2. That does not refute the claim, because my
proxy for "a sign" is any dark area in the rough, which is mostly ordinary
contour ink sitting where the drawing is dense, and fills are also where the
drawing is dense. **Registered as open**, with the instrument it needs: a
mark is a *sign* only if it is an area the artist filled in the rough and did
not have to, which requires the 面 of section 3 to even be stated. Not
claimed either way until then.

### 3. "縮小して閉じて見えるなら面" -- measured, and it does not recover the 面

Two readings of the proposal, both run against the thing it is for: does a GT
fill component come out as its own region?

**(a) Downscale** (`tools/evaluation/closure_by_downscale.py`). Barrier from
the rough, scales 1 to 1/16, two ink-coverage thresholds:
**enclosed 0.0% at every setting**, with the matched region a median 93-98% of
the panel. Barrier taken from the de-filled GT instead: also 0.0%.
The diagnostic says why -- `results/region_closure_20261010/downscale_diagnostic.png`:
INTER_AREA downscaling of a 2.7px stroke gives ~0.17 coverage per cell, so the
stroke **breaks into dashes** as the image shrinks. Shrinking makes the barrier
worse, not better. The eye bridges the gap; this operation widens it.

**(b) Morphological closing** of the rough's ink, which is the reading that
actually bridges gaps (`tools/evaluation/closure_by_closing.py`, 59 panels,
1,602 fill regions >=500px):

| closing radius | barrier cost | enclosed (fill area) | median tightness |
|---:|---:|---:|---:|
| 0 | 2.1% | 0.0% | 0.167 |
| 4 | 2.6% | 0.1% | 0.178 |
| 8 | 3.7% | 0.3% | 0.178 |
| 16 | 6.6% | **9.3%** | 0.085 |
| 32 | 12.5% | **15.4%** | 0.033 |

Closing does something a shrink cannot, and it is still not enough: at best
15% of the fill area ends up inside a sub-panel region, and that region is
**30x the size of the fill**. "Enclosed" is being bought by swallowing the
neighbourhood, which `tightness` is there to catch.

**Consequence**: the mechanical readings of "treat an ambiguously closed
outline as a 面" do not deliver a 面. The requirement the user states is
right; cheap morphology does not meet it. This *supports* the registered
design rather than changing it -- the model stays pixel-wise and the metric
stays region-wise, and no region proposer is built.

### 4. What replaces `rough_dark` as the thing to beat

If the rough does not mark fills, then what predicts them is **what the region
is** -- hair, top, tights -- and the user has already said where those sit.
That is the same shape as lesson 13 (*the word does not carry what the thing
is; position and size inside a detected frame do*) and it makes lesson 14's
mandatory no-training control into this stage's main hypothesis rather than a
formality.

Registered as stage 2a, no GPU, before the slot:

  2a-1  Run `deepghs/anime_face_detection` (ONNX, the detector Track H
        registered in H2) over the koma panels. Instrument check: zero
        detections on a blank image, and the box-orientation check in
        `../lineart/tools/evaluation/face_box_check.py` -- boxes read with x
        and y exchanged leave the panel, which is the lesson-12 anchor here.
  2a-2  Describe every GT fill component >=500px in **face-relative**
        coordinates: offset and size in units of the nearest face box.
        The user's claim predicts three clusters -- above/around the face
        (hair), below it at roughly torso offset (tops), further below
        (tights) -- and that is a prediction about a histogram, falsifiable
        before anything is trained.
  2a-3  The frequency table: predict filled/not from face-relative position
        and size alone, no training, fitted on the train families. **This is
        now the baseline the GPU arms must beat**, in place of `rough_dark`.

Pass criteria C1-C5 are unchanged, and C3 now bites harder: if the frequency
table comes within 0.05 region-F1 of M0/M1, the headline is that a histogram
over face-relative position does the job.

### 5. 枝落とし -- recorded, with what it would take

The user: a line running continuously out of a ベタ面 has to be cut off like a
branch, and the way to find it is that it does not fit the curve the stroke
was following. This is a real defect in the current mask -- stage 1's montage
showed eye outlines and a pair of glasses inside the radius-4 fill mask.

Measured bound on how much is at stake, from stage 1's radius ladder: 93.6%
(housei) and 92.3% (ako5) of the fill area survives going from radius 4 to 6,
so roughly 7% of the mask is narrow enough to be branch rather than 面. That
is the budget this would recover, and it is small enough that it is **not**
registered as part of stage 2's scoring. What is registered: the radius
ladder is reported, so no verdict rests on the 7%.

The stronger method the user describes -- fit the stroke's own curve and cut
where the ink stops following it -- is a separate instrument, and the honest
note is that it needs the stroke decomposition the project already has
(`tools/evaluation/stroke_decomposition.py`) pointed at a new question. Not
started, not registered, recorded here so it is not lost.
