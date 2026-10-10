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

## 2026-10-10 (correction) I misstated the user's claim about rough marks

The amendment above wrote that the user's account "has a sign ... so fill rate
under such a mark should be *below* base, lift < 1". That is not what was
said. The user, correcting it:

> サインがあるのは迷いだから ベタになるかもしれないしならないかもしれないが、
> 何もない真っ白よりなりにくいわけではない

So the claim is that a mark means **undecided** -- it may or may not become a
fill -- and explicitly **not** that a marked area is less likely than blank
paper. The measured lift of about 2 is therefore *consistent* with the
account, not in tension with it, and there is no open sign question to
resolve. What stands unchanged is the part that actually governs the design:
**96% of the fill area has no mark at all**, so `rough_dark` is a ceiling of
4.2% recall and not a contender.

Noted as an error of mine in reading, not a change of position by the user.

## 2026-10-10 Plan for the stroke-equation instrument (items 2 and 4) -- DRAFT, nothing run

User: items 2 (recovering a 面 from outlines that only look closed) and 4
(cutting a line that runs out of a ベタ面 as a branch) are both to be taken
up, and item 4 comes with a hypothesis:

> おそらく大半の線は放物線状のアーチか直線なので線からその線がもつ方程式を
> 見いだせると思う（あるいは途中の点から別の曲線に変化するものもあるかも）

### The reason to do them together: they are one tool

Item 4 needs to know what curve a stroke is following, to say where the ink
stops following it. Item 2 needs to bridge a gap the eye bridges -- and the
measured reason the cheap methods failed is that both are **isotropic**:
closing by radius 32 swallows the neighbourhood (tightness 0.033) because it
grows in every direction, and downscaling breaks strokes instead of joining
them. Extending a stroke **along its own fitted curve** is anisotropic by
construction: it adds ink only where the drawing was already going. So the
same fit that cuts a branch in item 4 is what closes a gap in item 2, and
item 4 must be built first.

### The instrument

`tools/evaluation/stroke_curves.py` (new, no GPU):

  1. skeletonize the ink (skimage 0.25.2 is present)
  2. split the skeleton at junctions (degree >= 3) into polyline branches
  3. per branch, fit in its own PCA frame: **line** (degree 1) and
     **parabola** (degree 2), keeping residual RMS for each
  4. piecewise: search one breakpoint, fit two parabolas, accept the split
     only if the residual drop beats a fixed penalty -- the user's "途中の点
     から別の曲線に変化するもの"
  5. emit per branch: class (line / arc / split / unexplained), coefficients,
     residual, length, endpoints with tangents

### Hypothesis H-curve, stated so it can fail

**At least 70% of skeleton branch length is explained as a straight line or a
single parabola at residual RMS <= 1.0px**, with the split class allowed to
carry a further share reported separately. Below 70%, the hypothesis that
strokes have a recoverable equation is not supported on this material and
item 2's bridging loses its basis -- that is a real possible outcome and the
number is fixed here before the first fit is run.

### Instrument checks

  IC-c1  synthetic strokes with known equations -- a straight segment, a
         parabola of known curvature, a circular arc -- rendered as ink at
         the corpus's own stroke width and recovered. The line and parabola
         must come back within a tolerance written down before the run; the
         **circular arc must come back as a worse parabola fit than the
         parabola does**, or the classifier is not distinguishing shapes.
  IC-c2  the skeleton overlaid on the line PNG **on disk**, not on anything
         this pipeline produced (lesson 12's anchor), and a branch count on
         one panel counted by hand first.
  IC-c3  frame invariance: rotate the input 90 degrees; every residual must
         be unchanged. A fit done in image axes rather than the stroke's own
         frame fails this and would otherwise pass everything else.

### Negative controls, one named error each

  NC-c1  random polylines with the same length and endpoint distribution as
         the real branches. If real strokes do not fit much better than
         these, "most lines are arcs or straight" is unsupported and the
         measured 70% was a property of short segments, which fit anything.
  NC-c2  **shuffled tangents** for the bridging step: extend each endpoint in
         a random direction instead of along its fitted curve. Isotropic
         closing already reaches 15.4% enclosed, so bridging must beat its
         own shuffled version, or it is dilation wearing a hat. This is the
         control the whole of item 2 rests on.
  NC-c3  branch pruning applied to a panel with no solid fill at all: it must
         remove nothing.

### Pass criteria for item 2, against what is already measured

Isotropic closing's best, on 59 ako5 koma panels, is **enclosed 15.4% of fill
area at tightness 0.033** (radius 32). Curve-based bridging passes only if it
reaches **enclosed >= 25% at tightness >= 0.30** -- both together, since
either alone is reachable by swallowing or by doing nothing. NC-c2 must fall
at least 10 points below it.

### Order and cost

Item 4's fit first, then item 2's bridging on top of it; item 2 cannot be
evaluated before IC-c1 and NC-c1 have been read. No GPU, so this does not
compete with the 月-木 slot. The honest estimate is that this is the larger
piece of work of the two stages, and the user has said as much
("かなりコストをかけて道具を準備しなければならない").

### Item 3 stays on hold, with one exception worth naming

The user: 髪/トップス/タイツ の単語化 has not started, so stage 2a waits.
Noted that **2a-1 and 2a-2 need no vocabulary at all** -- a face box is a
detector output and the histogram of fill positions relative to it is
label-free; only the *naming* of the three peaks touches the word work. The
histogram could therefore be run now and would either show three clusters or
not. Held anyway, since the user's call was to wait.

## 2026-10-10 Item 4, first result: the fit works, and the unit it works on is not a stroke

`tools/evaluation/stroke_curves.py` and `check_stroke_curves.py` built and run
on the ako5ver2 koma panels. Instrument checks first, as registered.

### Instrument checks

**IC-c1 synthetic strokes, PASS.** Rendered at the corpus's own 3px width and
put through the same skeleton -> branch -> fit path as real data:

| case | class | line rms | parabola rms | recovered a |
|---|---|---:|---:|---:|
| straight segment | `line` | 0.000 | 0.000 | - |
| parabola, true a = 0.004 | `arc` | 27.399 | 0.322 | **0.00402** (0.5% off) |
| 120-degree circular arc, R=200 | `unexplained` | 29.873 | 2.346 | - |

The arc fits the parabola model **7.3x worse** than a parabola does, against a
bar of 3x fixed before the run. Worth carrying: a 120-degree circular arc is
*not* within 1.0px of a parabola, so `arc` in this tool means "parabola-like",
and a real circular arc lands in `split` or `unexplained`. That is a property
of the model, now measured rather than assumed.

**IC-c2 anchor outside the pipeline, PASS.**
`results/stroke_curves_20261010/ic_c2_skeleton_overlay.png` draws the branches
over the **line PNG on disk**, which nothing in this chain produces. The
skeleton sits on the ink.

**IC-c3 amended, then PASS.** As registered it rotated the *image*, which
compares 11,166 branches against 11,313 -- skeletonization is not exactly
rotation-equivariant, so a 0.2-point difference says nothing about the fit.
Rewritten to rotate the branch **coordinates** (same points, new frame) by 90
and 37 degrees: 1,495 branches, worst residual difference **6.5e-13**. An
image-axis fit would fail this and pass everything else. Amendment recorded
because the check as first written could not do its job, not because it was
inconvenient.

**NC-c1 random walks, PASS and decisively.** Matched to the real branch-length
distribution:

| | real | random walk |
|---|---:|---:|
| explained, all branches >=10px | 97.0% | 45.0% |
| explained, branches >=30px | **95.7%** | **15.8%** |

### H-curve: passes as registered

Registered bar: at least 70% of skeleton branch length explained as a straight
line or a single parabola at RMS <= 1.0px, with `split` reported separately.

| image | branches | subset | line | arc | split | **explained** |
|---|---:|---|---:|---:|---:|---:|
| rough | 22,894 | >=30px | 61.6% | 22.5% | 11.5% | **95.7%** |
| line art | 20,091 | >=30px | 35.3% | 22.5% | 27.3% | **85.1%** |

Both clear 70% and the control is 80 points below. **But the number must not
be read as the user's hypothesis confirmed**, for the reason below.

### What the number is actually about, and it is lesson 11 again

The unit here is the **junction-to-junction fragment**, not the drawn stroke.
Median branch length is **16px in the rough and 31px in the line art**, and
the median end-to-end distance divided by path length is **1.00-1.02** -- a
typical branch is a short, nearly straight piece. A short straight piece fits
a straight line by construction, which is most of the 61.6%.

Visible in the IC-c2 sheet: one drawn eyebrow is cut into several red
fragments where other strokes cross it, and a single continuous eyelid arc
comes out `unexplained` because it was cut mid-curve.

So the honest statement is: **real ink is locally smooth in a way a random
walk is not (NC-c1 settles that), and whether a drawn stroke has a recoverable
equation is still untested**, because no drawn stroke has yet been assembled.
This is lesson 11's warning arriving on schedule -- per-pixel, per-stroke and
junction-chained are three different units and they do not agree.

### A second finding, which decides how item 4 has to work

In the line art, **48.6% of long-branch length lies inside the solid-fill
mask**: it is the fill's medial axis, not a drawn stroke. In the rough the
same figure is 10.6%, as expected, since the rough has few fills.

And the fit does not see the difference:

| line art, branches >=30px | explained |
|---|---:|
| outside the fill mask | 85.4% |
| inside the fill mask | **84.9%** |

A fill's spine is as parabola-like as a stroke. **Curvature therefore cannot
be the thing that separates a line from a fill**; width has to be, with the
curve used afterwards to decide where to cut. Item 4's design changes
accordingly, before anything was built on the wrong assumption.

### Next step, registered

Chain fragments through junctions by curve continuity -- choose, at each
junction, the continuation whose tangent and curvature agree -- and re-run
H-curve at the stroke unit. This is required twice over: it is the only way
to test the hypothesis the user actually stated, and it is the same operation
item 2 needs, since continuing a stroke across a junction and continuing it
across a gap are one mechanism. The item-2 pass criteria (enclosed >= 25% at
tightness >= 0.30, NC-c2 at least 10 points below) are unchanged.

## 2026-10-10 Item 4 step 2 pre-registration: chaining fragments through junctions

Written before a line of the chainer existed. The user asked how the join is
actually decided, so the mechanism is spelled out here in full; the thresholds
below are the registered ones and are not to be moved after a result is seen.

### The graph

Junction pixels (skeleton, 8-neighbour degree >= 3) are not single pixels --
an X crossing yields a small cluster of them. So:

  node  = a connected component of the junction-pixel set (8-conn)
  arm   = (fragment, which end), for every fragment end 8-adjacent to a node
  the chainer's only job: at each node, decide which arms continue each other

Unlike the H-curve run, **short fragments are kept** (no MIN_BRANCH_PX gate at
this stage). Dropping a 4px fragment would sever a stroke that runs through it.

### What is measured at an arm

Not the fragment's global PCA -- a long curved fragment's overall direction is
not its direction at the tip. A local window of the last `TAN_WIN = 12` px at
that end, read inside -> outward, with the frame built explicitly rather than
from SVD sign luck:

  u = unit tangent from the window's PCA, flipped to point OUTWARD
  v = rot90(u)                     (so the handedness is fixed, not inherited)
  t = a s^2 + b s + c              fitted in (s = along u, t = along v)

At the tip (s = s_e): slope m = 2 a s_e + b, so

  outward tangent  T = (u + m v) / sqrt(1 + m^2)
  signed curvature K = 2a / (1 + m^2)^{3/2}

A fragment shorter than 3px has no curvature: T = tip - other end, K = 0.

### The cost of joining arm i to arm j

Traversing i outward and then into j means entering j against its outward
direction. A plane curve's signed curvature flips under reversal, so
continuity is `K_i ~ -K_j`, not `K_i ~ K_j`. Three terms, each in its own
physical unit, then normalised:

  angle      theta = angle(T_i, -T_j)                    [degrees]
  curvature  dK    = |K_i + K_j| / (|K_i| + |K_j| + 1/KAPPA_REF)
  offset     d     = mean of the two perpendicular distances from one arm's
                     extrapolated parabola to the other arm's tip   [px]

  cost = theta/THETA0 + W_KAPPA * dK + d/D0

  THETA0 = 30 deg   D0 = 1.5 px   W_KAPPA = 1.0   KAPPA_REF = 200 px
  hard gates: theta <= 60 deg, d <= 3.0 px  (a wider pair is never joined)

The offset term is what separates a real continuation from two strokes that
pass a junction side by side with parallel tangents. The KAPPA_REF floor is
what stops two nearly-straight ends being punished for noisy curvature.

### The decision at a node

A **maximum-weight matching over the node's arms**, not greedy pairing, and
**arms are allowed to stay unmatched** -- that is the whole point at a T,
where the bar continues and the stem genuinely stops. Unmatched costs
`C_UNMATCHED = 1.0` per arm, so a pair is only made when its cost is under
2.0; a pure 60 deg kink costs exactly 2.0, which is why the gate and the
break-even coincide. All matchings are enumerated exhaustively for nodes with
up to `MAX_ARMS = 8` arms; a larger node leaves every arm unmatched and is
counted in the report rather than handled silently.

Every arm is used at most once, so the result is automatically a set of
disjoint simple paths and cycles -- no stroke ever branches.

### Instrument checks

- **IC-j1 synthetic crossing.** A straight line and a parabola a=0.004 drawn
  crossing on a blank canvas. PASS = exactly 2 chained strokes, each one
  classed `line`/`arc` (NOT `split`), and the parabola's coefficient
  recovered within the IC-c1 tolerance (20% relative, rms <= 0.5). A wrong
  pairing produces two strokes with a kink, which the fit calls split or
  unexplained -- so this check is sensitive to exactly the failure it is for.
- **IC-j2 T-junction.** A straight bar with a stem meeting it at 90 deg.
  PASS = the bar is one stroke and the stem is a separate stroke; 3 arms ->
  1 pair + 1 unmatched. This is the check that fails if matching were greedy.
- **IC-j3 rotation.** IC-j1 rotated 37 deg in COORDINATES (the IC-c3
  amendment). PASS = identical pairing, cost deltas < 1e-9.
- **IC-j4 external anchor (lesson 12).** Each recovered stroke drawn in its
  own colour over the on-disk line PNG, under `results/`. Judged by eye
  against strokes a person would have drawn.

### Negative controls, one named error each

- **NC-j1 shuffled geometry.** At each node, the arms' (T, K) are replaced by
  those of a different randomly chosen node with the same arm count, seed
  20261010. Named error: *the matcher is not using geometry at all, and any
  matching of the right shape would score the same.*
- **NC-j2 random valid matching.** At each node, a uniformly random matching
  with no gates and no cost. Named error: *the gain comes merely from making
  strokes longer, since a longer stroke is a different unit, not from making
  the right ones.*
- **NC-j3 length fairness.** Not a pass/fail arm but a condition on reading
  NC-j1/NC-j2: their stroke-length medians must be within 20% of the real
  chainer's. If they are not, the explained share is reported as a curve
  against length rather than as a single number, because a longer stroke is
  harder to explain and the comparison would otherwise be rigged.

### H-stroke pass criterion, fixed now

Same model and same thresholds as H-curve, so the two units are directly
comparable: **>= 70% of stroke length in the line art, strokes >= 30px and
`in_fill < 0.5`, explained as line / single parabola / one-breakpoint split
at RMS <= 1.0px**, with NC-j1 and NC-j2 both at least 10 points below. The
rough is reported beside it, never pooled with it. Inside-fill strokes are
reported separately, as in H-curve.

A length-scaled breakpoint budget (floor(len/60), max 3) is computed and
reported as **exploratory only**; it cannot move the criterion above. The
controls get the same budget whenever it is quoted.

Cycles (closed chains) cannot be fitted in a PCA frame and are counted and
reported, not dropped quietly.

## 2026-10-10 Chainer: three amendments, each forced by a check that failed

Recorded before the H-stroke number was read, in the order the failures came.

### IC-j1 failed, and the reason is topological, not a bug

A straight line crossing a parabola gave **3 strokes, two of them `split`**,
not 2. The skeleton of an X is not two crossing paths: skeletonizing produced
**two** degree-3 clusters bridged by a **1.0px stub**, so the crossing has 6
arms across 2 nodes, and the stub belongs to *both* drawn curves. No matching
that uses each arm at most once can recover both strokes while the stub is a
fragment.

**Amendment:** junction clusters joined by a fragment of at most
`MERGE_PX = 6.0` are merged into one node and the stub is absorbed into it.
The stub's ink then joins no stroke, so the absorbed count and absorbed
length are printed in every run rather than disappearing. 6.0px is about
twice the measured stroke width on this material (2.74px over `ink & ~fill`),
which is the scale the stub's length is set by.

After the amendment IC-j1 PASSES: 4 arms, 2 pairs, 2 strokes, `line` +
`arc`, with **a = -0.004011 recovered against a true 0.004** (0.3% error).

### IC-j3 failed on a real panel, which the synthetic case could not show

On the synthetic crossing IC-j3 passed at 4.9e-13. On a real koma panel it
failed with a **worst term delta of exactly 37** -- the rotation angle. Cause:
a 1-pixel fragment has no direction, and the code fell back to a fixed
`[1, 0]`, i.e. to the image axes. This is the same class of error IC-c3 was
written to catch, and it only appeared because the check was pointed at real
ink.

**Amendment:** a directionless arm now returns no tangent and is never
matched. A fixed axis is never substituted for a measurement that does not
exist.

### IC-j3's tolerance was itself mis-registered

With that fixed, the real panel gave matching-identical = True and a worst
delta of **1.21e-06**, against a registered bar of 1e-9. The bar was wrong,
not the code: `theta = degrees(arccos(T_i . -T_j))` and arccos has an infinite
derivative at +-1, so for two collinear arms a 1e-12 rounding difference in
the dot product becomes `sqrt(2e-12)` rad = 8e-5 deg. Measured worst case is
**theta = 180.0 vs 179.99999879**, with dK at 2.3e-11 and d at 7.7e-13 --
conditioning, not an axis dependence.

**Amendment, with the reason above and not after seeing a result:** per-term
tolerances, cost 1e-6 / theta 1e-3 deg / dK 1e-6 / d 1e-6, and the criterion
that actually carries the check is left exact and unchanged -- **the chosen
matching must be identical**. Real panel now: matching identical, cost 4e-08,
theta 1.2e-06, dK 2.3e-11, d 7.6e-13. PASS.

### The first real run showed two of the registered constants are wrong

40 ako5ver2 koma panels, line art: 4,368 arms but only **537 joins**, median
chain 14px, **median 1 fragment per chain**. The chainer was barely chaining.
Per-candidate-pair diagnosis on one panel (409 candidate pairs, 102 nodes):

| gate verdict | pairs |
|---|---:|
| rejected by theta | 188 |
| rejected by offset d | 35 |
| passed gates but cost >= 2 | 97 |
| no direction | 37 |
| **accepted** | **52** |

| term | p10 | p25 | median | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| theta (deg) | 10.5 | 32.1 | **61.8** | 91.1 | 133.0 |
| offset d (px) | 0.55 | 1.15 | 1.83 | 2.83 | 5.94 |
| dK | 0.00 | 0.21 | **0.73** | 0.91 | 0.95 |
| cost | 1.75 | 2.70 | 4.23 | 5.87 | 7.88 |

Two defects, both in numbers I set by eyeball:

1. **The tangent is read where it is most deformed.** Skeletonizing pulls ink
   toward a junction's centre, so the last pixels of a fragment bend into the
   junction. At a 3-arm node only one of the three candidate pairs is a real
   continuation, so about a third of candidates should be good -- but p25 of
   theta is already 32 deg and the median 62 deg, so the 60 deg gate is
   discarding real continuations. Added `TAN_SKIP`: a collar is skipped at
   the tip, the parabola is fitted behind it, and the tangent is **evaluated
   at the tip by extrapolation**.
2. **Curvature over a 12px window is mostly quantisation noise.** `dK` has
   median 0.73 of the 2.0 budget, so the curvature term is spending the
   budget on noise. `KAPPA_REF = 200px` is far too small a floor: a 0.5px
   jitter over a 12px window is already a curvature of ~0.03, six times the
   1/200 floor.

### How those constants get fixed without fitting them to the hypothesis

Calibrating them on the explained share would be fitting the instrument to
the thing being tested. So a **labelled target that is not the hypothesis**,
on real ink (lesson 12):

**IC-j6, cut and rejoin.** Take a real line-art panel; crop a window centred
on a long fragment F; draw a straight occluding stroke of 36px across F at a
random angle, only where the occluder stays 2px clear of all other ink so the
ground truth stays known; re-skeletonize, which cuts F in two at a new
junction. The two parts are one stroke **by construction**.

  rejoin = share of trials where both parts end in one chain   (recall)
  wrong  = share of trials where a part of F is chained to a piece of the
           occluder instead                                   (precision)
  score  = rejoin - wrong

Panels are split mechanically: the H-stroke number is read from panels
**[0:40]**, calibration uses **[40:171]**, so nothing is tuned on the images
the number comes from. The angle histogram is always reported beside the
score, because sites are only usable where the occluder is clear of other ink
and that biases which angles get tested.

Swept: TAN_SKIP, TAN_WIN, KAPPA_REF, THETA_MAX, D_MAX, W_KAPPA. The selection
rule, fixed before the sweep was read: **the highest `rejoin - wrong`**, with
ties broken toward the smaller parameter change from the registered values.

Pre-calibration score on 74 trials from panels [100:171]: rejoin 0.486,
wrong 0.027, **score 0.459**.

## 2026-10-10 IC-j6 calibration result, and H-stroke: FAILS as registered

All numbers from `results/stroke_chain_20261010/`
(`icj6_grid.csv`, `icj6_tanwin_extension.txt`, `hstroke_line.txt`,
`hstroke_rough.txt`, `strokes_ako5_*.csv`, montages `icj4_*.png`).

### IC-j6: the calibration, and what it says about my own description

309 labelled cut-and-rejoin trials, panels [40:171], disjoint from the [0:40]
the H-stroke number is read from. Best of each axis over the registered
972-point grid:

| axis | best value | score | rejoin | wrong |
|---|---|---:|---:|---:|
| W_KAPPA | **0.0** (0.5 -> 0.602, 1.0 -> 0.592) | 0.615 | 0.654 | 0.039 |
| TAN_SKIP | **2** (0 -> 0.553, 4 -> 0.447) | 0.615 | 0.654 | 0.039 |
| TAN_WIN | **20** (12 -> 0.515, 8 -> 0.385) | 0.615 | 0.654 | 0.039 |
| THETA_MAX | no effect (45/60/75/90 all 0.615) | | | |
| D_MAX | 3.0 = 5.0 (2.0 -> 0.576) | | | |
| KAPPA_REF | no effect, because W_KAPPA is 0 | | | |

Registered constants scored **0.362**; calibrated **0.615** (rejoin 0.654,
wrong 0.039). Selection rule was fixed beforehand and not touched.

Three things worth keeping:

- **Curvature agreement is worth nothing at a junction.** Every combination
  in the top band has `W_KAPPA = 0`, and the three KAPPA_REF values tie
  because the term is switched off. **My own description of the mechanism to
  the user -- "the continuation whose tangent AND curvature agree" -- is half
  wrong.** Curvature read over a 20px skeleton window is quantisation noise.
  Only the tangent and the sideways offset carry anything.
- **The collar is a precision fix, not a recall fix.** TAN_SKIP 0 -> 2 leaves
  rejoin almost unchanged (0.657 -> 0.654) and drops wrong joins from 0.104
  to 0.039. The deformed last pixels at a junction were making the *wrong*
  arms look collinear.
- **TAN_WIN = 20 sat on the grid edge**, so that axis was extended beyond the
  registered grid (24 -> 0.583, 32 -> 0.579, 48 -> 0.560, 64 -> 0.547). It is
  a real interior optimum; the grid did bracket it.

With the calibrated constants IC-j1, IC-j2 and IC-j3 all still PASS (IC-j1
recovers a = -0.004011 against a true 0.004).

### H-stroke: the bar is cleared, the control condition is not

Registered: **>= 70% of stroke length explained, strokes >= 30px with
in_fill < 0.5, AND NC-j1 and NC-j2 both at least 10 points below.**

| | line art | rough |
|---|---:|---:|
| real | **74.8%** | **92.9%** |
| NC-j1 shuffled tangents | 72.9% (**-1.9**) | 91.6% (**-1.3**) |
| NC-j2 random matching | 68.7% (**-6.1**) | 88.8% (**-4.1**) |

**H-stroke FAILS.** The 70% bar is cleared on both, and the registered
10-point control gap is met on neither. The bar is not moved; this is a fail
and is recorded as one.

### Why it fails, which is a defect in the criterion I wrote

The chainer joins only **7,513 of 40,212 arms (18.7%)** in the line art and
**6,812 of 54,573 (12.5%)** in the rough. So the registered subset is mostly
chains of a single fragment, on which the real arm and both controls are
**identical by construction**:

| line art, >=30px outside fill | real | NC-j1 | NC-j2 | share of px |
|---|---:|---:|---:|---:|
| single fragment | 88.0% | 88.2% | 86.8% | 57% |
| chained, >= 2 fragments | 56.8% | 50.9% | 43.8% | 43% |
| chained, >= 3 fragments | 40.0% | 30.0% | 22.0% | 20% |

| rough, >=30px outside fill | real | NC-j1 | NC-j2 | share of px |
|---|---:|---:|---:|---:|
| single fragment | 95.9% | 96.1% | 95.7% | 59% |
| chained, >= 2 fragments | 88.6% | 82.9% | 76.1% | 41% |
| chained, >= 3 fragments | 81.2% | 71.6% | 58.9% | 15% |

I registered a criterion on a population the treatment barely touches. That
is my error in the registration, not a reason to rescore: the criterion
stands failed, and the cut below is reported as a diagnostic beside it, never
in place of it.

Where the arms genuinely differ the chainer does beat both controls, by
5.7-9.6 points against the shuffled tangents and 12.6-22.3 against random
matching -- so **the joins it makes are the right ones more often than
chance**, consistent with IC-j6's rejoin 0.654 / wrong 0.039.

### The finding that actually answers the user's hypothesis

In the line art, explained share falls monotonically as a stroke is actually
assembled: **88.0% at one fragment, 56.8% at two or more, 40.0% at three or
more.** The longer the drawn stroke, the worse a single line or a single
parabola describes it. So the hypothesis as stated -- a drawn stroke carries
one equation, straight or a parabolic arch -- **is not supported at the
stroke unit on the line art**, and the 95.7%/85.1% from the fragment unit was
indeed the artefact lesson 11 warned it was.

The rough behaves differently and much better: 95.9% / 88.6% / **81.2%**. A
rough stroke really does mostly follow one curve; a finished line does not.

The exploratory piecewise fit (at most `floor(len/60)` and no more than 3
breakpoints, each piece in its own PCA frame) explains **89.0%** of the
line art's >=3-fragment chains against controls at 70.8% and 62.2%, and
94.7% of the rough's against 83.2% / 69.0%. Registered as exploratory, so it
cannot move the bar -- but it says where the description that does fit lives:
**a drawn stroke is a short chain of parabolas, not one parabola.**

### IC-j4, by eye

`icj4_line_koma_0002_page0001_p3_zoom_line.png`: chains run the length of
real contours in one colour -- a jaw line, a hem, a long hair sweep -- and
the colour changes at junctions the chainer declined, which matches a 65%
rejoin rate. `..._zoom_fill.png` is the same panel's solid fill: a lattice of
long smooth chains that are the fill's medial axis and not drawn strokes at
all. The two sheets side by side are the clearest statement of why the
in_fill split has to stay in every table.

### Registered next steps

1. **The join rate is the binding constraint, not the curve model.** 18.7% of
   arms joined caps everything downstream. The next measurement is on IC-j6,
   not on the hypothesis: of the 34.6% of trials the chainer fails to rejoin,
   how many are refused by the cost break-even (2.0) versus matched to the
   wrong arm. Those are different repairs.
2. The single-parabola model is now known to be the wrong description above
   one fragment. Before item 2 is run, the model used for "does this
   continue" should be re-registered as **piecewise**, with the breakpoint
   budget fixed by length as above and the controls given the same budget.
3. Item 2's own pass criteria (enclosed >= 25% at tightness >= 0.30, NC-c2 at
   least 10 points below) are unchanged. The chainer is the instrument item 2
   needs and it now has a measured rejoin rate, so item 2's result can be
   read against a known instrument rather than an assumed one.
4. `np.polyfit` emits RankWarning on near-collinear short segments in
   `stroke_curves.poly_rms`. It affects only the exploratory piecewise fit's
   smallest pieces, where `line_rms` decides the class anyway, but it is
   noted rather than silenced.

## 2026-10-10 Diagnostic D1: is a fill's ink recoverable as curves at all?

User's question: is it hopeless that a solid fill lands in the pile of lines
the curve model cannot explain? Registered before measuring.

### First, correcting something I was about to claim

The chained run shows 74.8% explained outside the fill mask against 62.5%
inside, which looks like chaining separating a fill's spine from a drawn
stroke where curvature could not. **It is not.** Matched by chain length the
gap nearly vanishes and then reverses:

| fragments per chain | outside fill | inside fill | diff |
|---|---:|---:|---:|
| 1 | 88.0% | 83.2% | +4.9 |
| 2 | 71.5% | 65.8% | +5.7 |
| 3-4 | 46.6% | 44.8% | +1.8 |
| 5+ | 16.3% | 21.9% | **-5.6** |

The whole 12-point gap is composition: 11.2% of inside-fill length sits in
chains of 5+ fragments against 4.4% outside, and long chains are explained
less. **Curvature still cannot separate a fill's spine from a stroke**, which
is the 2026-10-10 fragment-unit finding unchanged. Recorded because I nearly
reported the opposite.

### The size of the problem

40 ako5ver2 koma panels, line art:

| | |
|---|---:|
| ink pixels | 23,868,453 |
| of which inside the fill mask | 21,100,383 (**88.4%**) |
| skeleton length, total | 1,264,628 |
| outside the fill | 611,606 (48.4%) |
| inside the fill (the medial-axis lattice) | 653,022 (51.6%) |
| fill contour length, 7,136 regions | 848,871 |
| unexplained length that is inside the fill | **60.3%** |

So a fill's spine is half of all skeleton length and three fifths of
everything the curve model fails on.

### Why this is probably not hopeless, and the thing that decides it

A fill's medial axis is **an artefact of skeletonizing a filled region**. It
is not a drawn mark: nobody drew the lattice. The drawn mark at a fill is its
**outline**. So the repair is not a better curve model, it is to stop
skeletonizing filled regions and take their contour instead -- and the
separator needed for that is width, which already exists as the fill mask
itself (a distance transform, computed from the image, not from a model).

The catch, and the reason this is a measurement and not an assertion: the
fill mask includes the 4px rim, so **the fill mask swallows the fill's own
drawn outline**. Deleting the fill therefore deletes real drawn lines. The
contour has to put them back, and that only works if a fill's outline is
about as curve-like as a drawn stroke.

### The measurement and its bar, fixed now

For each fill-region contour and each outside-fill chain >= 30px, split
recursively (each piece fitted in its own PCA frame, as in the exploratory
piecewise fit) until every piece is within RMS 1.0px, capped at 20 pieces,
and record **pieces per 100px of length**.

**対処可能 if the median pieces per 100px of fill contours is within 2x the
median of outside-fill drawn chains.** Then a fill's outline is a drawn line
of ordinary complexity and the lattice can simply be dropped.

**対処不能 (for this approach) if it is worse than 2x**, which would mean the
fill boundary is not a curve the model can carry and the fill needs its own
representation rather than being folded into lines.

Negative control **NC-d1**: the same measure on contours of a **shuffled**
region set -- each fill region's mask replaced by that of a different region
from another panel, scaled to the same area. Named error: *pieces per 100px
is just a function of how long and how blobby a closed curve is, and any
closed region of this size would score the same.* No pass bar; it is there so
the contour number can be read at all.

## 2026-10-10 Stage 2 start: the two definitions the registration left open

Fifth foundation notice (2026-10-10 20:00) directs M0/M1 to start; the GPU is
idle and the chainer is CPU-only and shares no code path, so it continues in
parallel. Before anything is built, two things the stage-2 registration named
but never defined are fixed here, in advance of any score.

Note on the notice's one correction, carried into every report from here:
IC1's "best preproc" column is `preproc@32`'s **0.0002** and is used
unchanged as a check value, but **the readable ceiling is 0.0000** -- that arm
fell to its own negative control in stage 1, so 0.0002 is a control-failed
arm's number, not a ceiling.

### The region metric, defined

    region := an 8-connected component of the solid-fill mask (the stage-1
              definition, radius 4, unchanged) with area >= MIN_REGION_PX
    match  := IoU >= MATCH_IOU between a predicted and a GT region,
              assigned greedily one-to-one in descending IoU order
    region-F1 := harmonic mean of  matched/n_pred  and  matched/n_gt

    MIN_REGION_PX = 500   primary. Chosen because the closure analysis already
                          used 500px for a 面 and reported 5-71 such regions
                          per ako5 panel, which is a plausible count of 面;
                          at 100px the count is 28-289 and dominated by pupils
                          and hatching blobs.
    MATCH_IOU = 0.5       primary.

    Sensitivity, computed every run and printed, never used to pick a verdict:
    MIN_REGION_PX in {0, 100, 500}  x  MATCH_IOU in {0.25, 0.5, 0.75}.

A panel with zero GT regions and zero predicted regions scores F1 = 1.0; with
zero GT and some predicted, 0.0. Panels are **not pooled**: region-F1 is
computed per panel and reported as a per-family mean, with the near-blank and
normal strata separate, as C5 requires.

### The window grid and the stitching rule, defined

    size 480, stride 480 - OVERLAP with OVERLAP = 64; the last row and column
    are clamped to the panel edge rather than padded, so a window always
    contains real image. A panel smaller than 480 in a dimension is padded to
    480 with background white, and the pad is removed before scoring.

    stitch(reduce="overwrite")  reproduces the input exactly -- this is what
                                IC4 is run on
    stitch(reduce="mean")       averages the overlapping predictions, then
                                thresholds at 0.5. Registered now so it is not
                                chosen after seeing which rule scores better.

### The two practices the fourth notice asked for

1. **One assert self-check per new non-trivial piece of logic**, with a known
   value: the window grid, the stitcher and the region matcher each carry one,
   run on import-time-cheap synthetic input.
2. **Known limits are noted as `shortcut: <limit>, <when it is raised>` and
   collected in one place** -- `doc/shortcuts.md`, which every stage-2 tool
   appends to rather than burying the note in a docstring.

## 2026-10-10 Stage 2 instrument checks: all PASS, and the two fitting details

### Checks, before any arm is read

| check | what it anchors to | result |
|---|---|---|
| split self-check | the registration's own counts | **PASS** 33 families, 9 held, 969/4049 panels, held families identical |
| IC3 GT as prediction | the matcher and the stitcher together | **PASS** 60/60 panels at region-F1 1.0 and pixel IoU 1.0 |
| IC4 stitching | the panel **PNG on disk**, bit-identical | **PASS** 300/300 (6 panels under 480px are padded) |
| IC5 region matcher | a hand-computed analytic value | **PASS** d=16px -> 2 matched, F1 1.0, disc IoU 0.598 vs analytic 0.596; d=30px -> 1 matched, F1 0.5, IoU 0.365 vs 0.364 |
| smoke M0 / M1 | end to end on the GPU | **PASS** forward (1,1,480,480), holdout scored through the stitcher, checkpoint written |

IC1 (re-scoring stage 1's 480 tiles through the new scorer) is the remaining
one and runs with the baseline table. Carried into every report, per the fifth
notice: **the readable preprocessor ceiling is 0.0000**; the 0.0002 used as
IC1's check value is `preproc@32`'s, an arm that fell to its own negative
control in stage 1.

The stop rule is cleared: IC4 passes, so training is permitted.

### Two fitting details the registration left open, fixed now

- **`freq_table` is fitted on every 10th train panel by sorted name** (405 of
  4049), and evaluated on the held side only. A mechanical subsample, chosen
  for cost, fixed before the table was built.
- **The negative controls are applied to an arm's already computed prediction
  mask** (flip, 50px shift, off-by-one), not recomputed from a transformed
  input, so an arm and its controls differ only by the named error. NC1
  (polarity) is the exception and must be recomputed, since the error being
  named is in the reading of the input.

## 2026-10-10 D1 result: the registered bar FAILS, and the bar compared the wrong things

40 ako5ver2 koma panels, `results/fill_contour_20261010/pieces.csv` and
`d1.txt`. Pieces needed to bring every piece within RMS 1.0px, each piece
fitted in its own PCA frame, cap 20.

| population | n | total px | median len | **median pieces / 100px** | hit cap |
|---|---:|---:|---:|---:|---:|
| drawn chains outside the fill, >=30px | 4,631 | 600,055 | 80 | **1.54** | 0.0% |
| fill-region contours | 3,583 | 952,023 | 90 | **3.44** | 3.7% |
| NC-d1 random closed curve, matched perimeter and isoperimetric ratio | 3,583 | 952,023 | 90 | **5.78** | 4.5% |

**fill contour / drawn chain = 2.24x against a registered bar of 2.00x, so
D1 as registered FAILS.** The bar is not moved.

NC-d1 reads: a fill's outline is far from shapeless -- 3.44 against 5.78 for
a random closed curve of the same perimeter and the same blobbiness.

### The defect, which is in my comparison and not in the material

**I compared closed loops against open chains and did not control for
closure.** A closed curve must turn 360 degrees in total; a short one is
forced to spend pieces on that no matter how smooth it is. The floor is a
circle, which this pipeline did not produce and cannot bend:

| circumference | radius | pieces | per 100px |
|---:|---:|---:|---:|
| 35 | 5.6 | 2 | 5.67 |
| 48 | 7.7 | 2 | 4.14 |
| **90** (the median contour) | 14.2 | 3 | **3.35** |
| 203 | 32.3 | 4 | 1.97 |
| 498 | 79.2 | 4 | 0.80 |

**The median fill contour needs 3.44 pieces per 100px; a perfect circle of
the same circumference needs 3.35. The contour is at 1.03x the analytic floor
for a closed curve of its size.**

So the honest statement is in two parts, and both belong in the report:

1. **D1 as registered failed**, and the criterion stays failed.
2. **The criterion was mis-specified by me**, and the correctly anchored
   measurement says the opposite of what the failed bar suggests: a fill's
   outline is about as smooth as a circle, which is as good as a closed curve
   of that size can be. The 2.24x is the price of closure, not of the fill.

### What this does and does not license

It does **not** license "対処可能" as a measured claim, because the 1.03x
figure was computed after seeing the per-length table and no criterion was
fixed in advance of it. What it licenses is dropping the lattice from the
list of things that need a better curve model: the medial axis is an artefact
of skeletonizing a filled region, and the drawn mark at a fill is its outline,
which is already smooth.

**Registered before the contour route is built on**: a fresh test with its
bar fixed first -- contours cut into pieces, handed to the chainer's
continuation model, and scored on held-out panels against NC-d1 with the same
piece budget. Until that runs, the fill's representation stays open.

### The 88.4% that frames all of this

Of 23,868,453 ink pixels on these 40 panels, **21,100,383 (88.4%) are inside
the fill mask**, but only 51.6% of skeleton length is -- fill is area-heavy
and lines are thin. 60.3% of all unexplained length is inside the fill. Both
readings are given because either alone misstates the size of the problem,
as the foundation's second notice requires.

## 2026-10-10 IC2 wired for real, with its bar fixed first

Stage 1's `check_fill_instrument.py` already reproduces the external
inventory's `fill_ratio` **exactly (delta 0.0000) for all 7 `shared/` pools**,
both legs. What it never checked is the pools stage 2 actually uses: the
koma-derived `trainlist:` rows. It already *gathers* them, because they
consume the same shared RNG, but it only scores `shared/`.

External anchor, produced by another track on 2026-09-11 and not by this
pipeline:
`../lineart-controlnet-sdxl-fidelity/results/pool_inventory_20260911/inventory.csv`,
written by `inventory_pair_pools.py` from a 120-tile sample at seed 20260911.
Its `fill_ratio` is defined as "the share of ink belonging to strokes wider
than 8px" -- the same definition as ours, implemented in different code.

### IC2a -- the published cell, with a bar

Extend the check to every `trainlist:` koma pool the inventory publishes:

    trainlist:clippairskoma   0.0917     trainlist:ako5ver2koma  0.1310
    trainlist:fitnesskoma     0.1618     trainlist:gakuenkoma    0.1403
    trainlist:hamlabikoma     0.1427     trainlist:houseikoma    0.1249

**Pass: every pool within 0.002 of the published value on the replayed seeded
sample**, the same tolerance stage 1 used, with the exact-EDT leg within 0.02
as sensitivity and not as a bar. If the sampling cannot be replayed, the
registered fallback is the all-tile mean with the published value inside
+/- 2 SE of a 120-tile draw.

### IC2b -- the unit carry-over, descriptive, registered as having no bar

Stage 2's corpus is koma **panels**; the published value is on 480 **tiles**
from a different list. A difference between them is a measurement of the unit
change, not a failure, so **IC2b carries no pass bar** and cannot fail IC2.
Three numbers are printed together so the unit effect is measured instead of
assumed:

    published tile value (external)
    our fill_ratio on clip_pairs_v3 koma panels
    our fill_ratio on 480 windows cut from those same panels

The third isolates the unit from the material: it is our own panels, read at
the published value's unit. Registered before any of the three was computed.

## 2026-10-10 IC2 result: IC2a PASS exactly, and IC2b found that stage 1 and stage 2 are not comparable pools

### IC2a -- PASS, 6/6, delta 0.0000

`results/fill_koma_20261010/ic2a_fill_ratio.csv`. Every koma pool the external
2026-09-11 inventory publishes, reproduced on the replayed seeded sample:

| pool | published | ours (cv2) | ours (exact EDT) | delta |
|---|---:|---:|---:|---:|
| trainlist:clippairskoma | 0.0917 | 0.0917 | 0.0917 | **0.0000** |
| trainlist:ako5ver2koma | 0.1313 | 0.1313 | 0.1313 | **0.0000** |
| trainlist:fitnesskoma | 0.1619 | 0.1619 | 0.1619 | **0.0000** |
| trainlist:gakuenkoma | 0.1396 | 0.1396 | 0.1396 | **0.0000** |
| trainlist:hamlabikoma | 0.1428 | 0.1428 | 0.1428 | **0.0000** |
| trainlist:houseikoma | 0.1249 | 0.1249 | 0.1249 | **0.0000** |

Bar was 6/6 within 0.002; both legs agree to 0.0000. Stage 1 had only ever
checked the 7 `shared/` cells, so the pools stage 2 actually uses were
unverified until now.

### IC2b -- no bar, and it found the thing that matters

`results/fill_koma_20261010/ic2b_unit_carryover.csv`.

| | fill_ratio |
|---|---:|
| published tile value (external) | 0.0917 |
| our clip_pairs_v3 koma **panels** | **0.4676** |
| our 480 **windows** cut from those same panels | **0.2510** |

A four- to fivefold gap on what the registration treated as the same
material. The diagnosis, on 60 panels and the same 60-panel sample of their
pool:

| | ink | width over ALL ink | width over `ink & ~fill` | fill_ratio |
|---|---:|---:|---:|---:|
| their 480 tiles (the anchor pool) | 0.0397 | 3.54 | **2.62** | 0.0928 |
| our koma panels, native | 0.0915 | 22.18 | **2.52** | 0.4494 |
| our 480 windows from them | 0.0923 | 15.16 | **2.07** | 0.2582 |

**It is not scale.** Width over `ink & ~fill` is 2.62 / 2.52 / 2.07 -- the same
drawing scale, which is what the registration's "stroke width identical to the
480 pools" claim was about and that claim survives. (The 1.91px quoted earlier
is a pooled median; 2.52 here is the mean of per-panel medians. Different
aggregation of the same quantity, not a conflict.)

**It is selection.** Their pool carries 0.0397 ink against our 0.0915 -- less
than half. Their `clippairskoma` tiles are the survivors of the koma
pipeline's training filter, and the 2026-07-26 note records that `ink_range`
rejected **80.9% of candidates**. What a filter keyed on ink range rejects is
exactly the fill-heavy material. **The externally profiled pool is what is
left after the fills were filtered out.**

### What this changes, and it is a correction to my own registration

The stage-2 registration says the corpus is "native scale, stroke width
1.91px (identical to the 480 pools, so the radius-4 fill definition carries
over unchanged and **stage 1's masks remain comparable**)". The first half
holds. **The second half does not**: the definition carries over, the pool
profile does not. Stage 2's corpus is 4.8x more fill-heavy than stage 1's.

Consequences, to be carried in every stage-2 report:

- **`all_black`'s identity value will be several times stage 1's.** Stage 1's
  0.0939 / 0.1201 are |G|/|P| on 480 tiles of filtered material; on this
  corpus the same identity is a much larger number. Any stage-2 number read
  against stage 1's is misread.
- **C1's 0.05 margin is not scale-free.** It was fixed against stage 1's
  numbers. It stays as registered -- bars are not moved -- but the report must
  say beside it that the base rate differs by ~5x, so clearing it here is a
  weaker statement than clearing it there would have been.
- This is **good news for the task and bad news for comparability**: stage 2
  is being run on material that actually contains the thing being predicted,
  which stage 1's pool largely did not.

### Visual, as required

`results/fill_koma_20261010/montage_held_normal_random.png` (8 held panels,
normal stratum, seed 20261010 fixed before any score was read). It shows the
user's domain claim directly: the GT fill is hair and tops, and the rough has
essentially nothing dark in those places. `rough_dark` is near-empty;
`rough_dark@budget` spends its budget on the rough's lines; `freq_table`
predicts almost nothing.

## 2026-10-10 Correction: the fill's ink WAS inside the H-stroke statistic

Asked directly whether the lines that appear as a fill's spine are inside the
failed verdict's statistic. They are -- not as the spine itself, but as
half-submerged chains, and it changes the reading completely.

### What the registered subset actually contains

The cut is `in_fill < 0.5`, the share of a chain's pixels inside the fill
mask. A chain with 49% of its pixels in a fill is inside the statistic. By
length, the line art's registered subset is only **35.7% strictly fill-free**:

| in_fill | chains | px | share of the subset | explained |
|---|---:|---:|---:|---:|
| exactly 0 | 2,373 | 186,728 | 35.7% | **94.8%** |
| 0 - 0.1 | 717 | 106,938 | 20.5% | 80.7% |
| 0.1 - 0.3 | 913 | 123,831 | 23.7% | 64.5% |
| 0.3 - 0.5 | 622 | 103,486 | 19.8% | **45.3%** |

The headline 74.8% is a mixture whose explained share falls monotonically with
fill contamination, from 94.8% to 45.3%.

### The verdict does not change

| line art, >=30px | real | NC-j1 | NC-j2 |
|---|---:|---:|---:|
| as registered, in_fill < 0.5 | 74.8% | 72.9% (-1.9) | 68.7% (-6.1) |
| zero fill pixels | 94.8% | 93.7% (-1.1) | 91.2% (-3.6) |

**H-stroke still fails** -- neither cut reaches the registered 10-point control
gap. The bar is not moved and the verdict stands.

### What I reported to the user and to the foundation was wrong

I reported that explained share falls as a stroke is assembled -- 88.0% at one
fragment, 56.8% at two or more, 40.0% at three or more -- and concluded that
**"a drawn stroke carries one equation" is not supported on finished line
art**. Restricted to chains with **zero** fill pixels:

| fragments per chain | as registered (in_fill<0.5) | zero fill pixels | px (zero-fill) |
|---|---:|---:|---:|
| 1 | 88.0% | **95.3%** | 152,242 |
| 2 | 71.5% | **92.8%** | 25,633 |
| 3-4 | 46.6% | **92.8%** | 8,526 |
| 5+ | 16.3% | 100.0% | 327 (too small to read) |

**The decline was fill contamination, not length.** On fill-free line ink the
explained share is flat at 93-95% however many fragments were chained, and on
3+ fragment chains the controls fall **16.9** (NC-j1) and **45.8** (NC-j2)
points below -- far past the 10-point gap the registered subset could not
reach.

So the user's hypothesis is **supported** on fill-free line ink, and my
"not supported" was an artefact of letting half-submerged chains into the
statistic.

### And the asymmetry I headlined to the foundation collapses

`note_track_j_rough_line_asymmetry_20261010b.md` leads with line art 40.0%
against rough 81.2% at 3+ fragments and offers it as bearing on the
rough -> line problem. On fill-free ink the same cell is **92.8% line art
against 87.9% rough**. The asymmetry was the fill mask: the line art has
fills, the rough has almost none. **It is not a property of the two domains.**
A correction has been filed.

### The confound in the other direction, checked

`in_fill == 0` selects chains far from any blacked region, which could simply
be panel borders, balloons and backgrounds -- simple lines. Checked, and it is
not that: at 3-4 fragments the fill-free population is 41.8% `line`, 26.2%
`arc`, 24.8% `split`, so **51% of it is curved**, with median straightness
0.931 against 0.910 and median curvature 0.00106 against 0.00134. What shifts
is `unexplained`, 7.2% against 58.9%.

Two real limits stay: the fill-free slice at 3-4 fragments is 8,526px against
71,971px in that cell, about a tenth of it; and it remains a selected
population.

### Why the cut cannot simply be tightened, and where this joins D1

`in_fill == 0` is not the right cut either -- it **over-excludes**. The fill
mask includes the 4px rim, so a chain tracing a blacked hair's outline is a
drawn line with high `in_fill`. Tightening the cut throws away every fill
contour, which D1 measured as being at **1.03x the analytic floor** for a
closed curve of its size -- i.e. as smooth as a drawn line can be.

So both threads arrive at the same place: **near a fill, the drawn line is the
fill's contour, and it has to be taken from the contour, not from the
skeleton.** The skeleton inside a fill is a medial axis and belongs to nothing
drawn; the skeleton on a fill's rim is a drawn line the mask has swallowed.

**Registered next, with its bar fixed before it runs**: re-run H-stroke on a
three-way split -- line ink with no fill contact, fill contours taken by
`cv2.findContours` and cut into open pieces, and the medial axis excluded
entirely -- with the same model, the same thresholds, and both controls given
the same treatment. Pass is unchanged: >= 70% explained with NC-j1 and NC-j2
both at least 10 points below, now on a population where every member is
something a person drew.
