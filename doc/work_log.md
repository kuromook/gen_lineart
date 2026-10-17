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

## 2026-10-10 H-stroke-3 pre-registration: the three-way re-run

Written before the tool existed. Every member of every scored population is
something a person drew; the medial axis is excluded by construction rather
than by a threshold.

### The three populations

  **A  line ink with no fill contact.** Skeletonize `ink & ~fill`, not `ink`.
     The medial axis then does not exist to be filtered out, and neither does
     the rim. A line running into a blacked region now ends about 4px early;
     that is where B takes over, and it is stated rather than hidden.
  **B  fill contours.** `cv2.findContours` on the fill mask (RETR_CCOMP, so
     holes count), cut into open pieces (below), each piece >= 30px.
  **C  the medial axis.** Excluded by construction. Not scored, not pooled,
     and its length is printed so the exclusion is visible.

A and B are scored **separately and never pooled** -- they are different
populations, and the project rule against pooling across pools applies.
The rough is reported beside the line art and never pooled with it.

### Where a contour is cut

A drawn contour ends where another stroke meets it. So:

    a contour point is a CUT POINT if the de-filled line skeleton
    (`skeletonize(ink & ~fill)`) has a pixel within CUT_NEAR_PX = 2.0 of it

Adjacent cut points are collapsed into one (a stroke meeting a contour touches
it over a few pixels). Between consecutive cut points the contour is one open
piece. **A contour with no cut point stays closed, cannot be fitted by a model
that needs a principal axis, and is counted and reported as
`closed_uncut` -- never dropped quietly.**

### Controls

A keeps the chainer's own controls, because A has a chaining decision:
  **NC-j1** shuffled tangents, **NC-j2** random matching, both as registered.

B has no chaining decision to shuffle, so it gets two of its own:
  **NC-t2  cut points shuffled** to random positions along the same contour,
     the same number of them. Named error: *the pieces fit merely because the
     contour was cut into that many pieces, not because the cuts are where
     strokes actually meet.*
  **NC-d1  a random closed curve** with the same perimeter and the same
     isoperimetric ratio, cut by the same rule. Named error: *any closed curve
     of this size and blobbiness would score the same.*

### Pass criterion, unchanged from H-stroke

**>= 70% of length explained as line / single parabola / one-breakpoint split
at RMS <= 1.0px, with every control for that population at least 10 points
below.** Applied separately to A and to B. Thresholds, the model, and the
30px floor are all unchanged, so the numbers stay comparable to the failed
run.

### The trap applied to myself

The last run failed because a threshold on mask overlap was read as a
population filter. So for every chain in A this run records `near_fill`, the
share of its pixels within 4px of the fill mask, and **prints that
distribution inside the kept set**, not just the rule that built it.

### Instrument check IC-t1, expected values written down first

A synthetic disc of radius 60 at the centre with two straight lines running
into it, one from the left and one from below. Expected: 2 cut points, so the
disc's contour becomes **exactly 2 open pieces**; both classed `arc` or
`split` (a half-circle is not a line); and a disc with **no** line touching it
produces **0 cut points and 1 `closed_uncut`**.

## 2026-10-10 H-stroke-3 result: B PASSES, A fails again, and the reason is now isolated

40 ako5ver2 koma panels. `results/stroke_three_way_20261010/`
(`h3_line.txt`, `h3_rough.txt`, `three_way_ako5_*.csv`). IC-t1 passes on all
three cases. Bar unchanged: >= 70% explained AND every control >= 10 points
below.

### Verdict

| population | image | real | control 1 | control 2 | verdict |
|---|---|---:|---:|---:|---|
| **A** line ink, no fill contact | line | 94.4% | NC-j1 93.5% (**-1.0**) | NC-j2 92.2% (**-2.3**) | **FAIL** |
| **A** | rough | 95.3% | NC-j1 94.4% (-0.9) | NC-j2 92.8% (-2.5) | **FAIL** |
| **B** fill contours, cut at strokes | line | **76.7%** | NC-t2 62.8% (**-13.9**) | NC-d1 49.5% (**-27.2**) | **PASS** |
| **B** | rough | 55.0% | NC-t2 47.9% (-7.1) | NC-d1 28.4% (-26.6) | **FAIL** |

**The population I expected to be the problem is the one that passes.**

### B passes, and it is the first clean pass in this thread

On the finished line art, a fill region's outline -- cut where another stroke
meets it -- is explained at 76.7% by the same model, same thresholds, same
30px floor, with the cuts-shuffled control 13.9 points below and the
random-closed-curve control 27.2 below. This is the pre-registered test the
D1 entry said was needed before the contour route could be built on
(registered at 7c008a8, before the tool existed and before any B number).

D1's 2.24x "failure" and this pass are consistent: D1 compared closed loops
against open chains and the loops paid for closure. Cut the loops where
strokes meet them and the pieces are ordinary drawn curves.

**So the fill's ink is recoverable as drawn curves.** Not by fixing the curve
model, and not from the skeleton -- from the region's contour.

The rough's B is a thin and strange population (97,274px against the line
art's 658,751) because the rough has almost no fill. Its 55.0% is reported
and is **not** read as a statement about contours in general.

### A fails again -- and this time it is not contamination

The near_fill distribution **inside the kept set**, the check the failed run
lacked:

| near_fill (within 4px of the fill mask) | line art | rough |
|---|---:|---:|
| exactly 0 | 40% | 77% |
| 0 - 0.1 | 47% | 15% |
| 0.1 - 0.5 | 12% | 8% |
| >= 0.5 | 1% | 0% |

87% / 92% of A's length is at 0.1 or below, so A is genuinely clean. The
failure is structural and is the one registered after the first run:

| line art, population A | real | NC-j1 | NC-j2 | share of length |
|---|---:|---:|---:|---:|
| all, as registered | 94.4% | -1.0 | -2.3 | 100% |
| **single fragment** | 95.1% | **+0.0** | **-0.0** | **53.1%** |
| 2+ fragments | 93.8% | -3.0 | -6.5 | 46.9% |
| 3+ fragments | 90.2% | **-9.2** | **-15.9** | 13.1% |

| rough, population A | real | NC-j1 | NC-j2 | share |
|---|---:|---:|---:|---:|
| single fragment | 96.5% | +0.3 | +0.3 | 62.2% |
| 3+ fragments | 91.2% | -8.8 | -19.4 | 10.2% |

Half the scored length is chains the chainer never joined, where the real arm
and both controls are **identical by construction** -- the +0.0 gap is that
identity showing up exactly as predicted. The treatment is applied to half
the population and measured over all of it.

On chains that were actually built, NC-j2 clears the 10-point gap on both
images (-15.9, -19.4) and NC-j1 lands just under it (-9.2, -8.8). **The
binding constraint remains the join rate, not the curve model** -- the same
conclusion the first run registered, now on an uncontaminated population.

### Other numbers the run prints

- `closed_uncut`: **94 fill contours on the line art (10,831px)** are touched
  by no stroke at all -- isolated blacked blobs. Counted and excluded from B,
  never dropped silently. The rough has 0.
- Cut-threshold sensitivity, line art: **25,371 cut points at 2px, 17,968 at
  4px (used), 12,625 at 8px**. A 1.4x swing each way, so the choice is not a
  knife edge but it is not negligible either; 4px is the fill mask's own
  reach and was set from that, not fitted.
- Medial axis excluded by construction: **653,022px (line art), 109,530px
  (rough)** -- 51.6% and 11.1% of all skeleton length.

### What this changes

1. **The fill is not a dead end.** Its outline passes a pre-registered test
   as a drawn curve. The representation for a fill is: the region, plus its
   contour cut at the strokes that meet it. The medial axis is discarded.
2. **A's criterion is still the wrong shape and I am not moving it.** An
   absolute 10-point gap measured over a population that is half untreated
   cannot be cleared by any chainer. The honest fix is not a softer bar but a
   higher join rate; the registered next step (splitting IC-j6's 34.6%
   failures into "refused by cost" and "joined to the wrong arm") is now the
   critical path and not an afterthought.
3. The user's hypothesis now has a measured answer on clean material: on line
   ink that never touches a fill, 94.4% of length is a straight line, a single
   parabola, or one breakpoint -- and on the chains long enough for the
   question to mean anything, 90.2% with both controls well below.

---

## 2026-10-10 Pre-registration: IC-j7, splitting IC-j6's 34.6% rejoin failures

Written before `tools/evaluation/diagnose_icj6.py` exists. IC-j6 scores
rejoin 0.654 / wrong 0.039 on the calibration panels; the 34.6% that fail to
rejoin are one number covering several different repairs, and the H-stroke-3
entry put splitting them on the critical path. Nothing in the chainer's
constants is changed by this entry: it is a diagnosis, and any repair it
suggests gets its own registration.

**Where it runs.** The calibration panels `[100:171]` of `line`, the same set
IC-j6 was calibrated on, seed 20261010. The measurement panels `[0:40]` stay
untouched, so nothing here can reach the H-stroke number. 309 usable trials.

### The buckets, fixed now

For a failed trial, find the merged node holding one end of each of the two
true parts of the cut fragment F. Classify by what happened to that pair at
that node:

| bucket | condition | the repair it implies |
|---|---|---|
| `gate_theta` | the true pair's `theta > THETA_MAX` (60 deg) | widen the angle gate |
| `gate_d` | the true pair's `d > D_MAX` (3.0px) | widen the offset gate |
| `gate_both` | both gates reject | both |
| `cost_break_even` | ungated, but `raw cost >= 2.0` = `2 * C_UNMATCHED`, so the matcher prefers two unmatched arms | re-weight the cost or raise `C_UNMATCHED` |
| `wrong_arm_occluder` | one of the two arms is linked to an arm of the occluder | the cost ranks a false continuation above the true one |
| `wrong_arm_other` | one of the two arms is linked to some third piece of ink | same, different culprit |
| `over_max` | the node carries more than `MAX_ARMS` = 8 arms and is skipped whole | raise the cap or decompose the node |
| `topology` | the two parts share no merged node (a part was absorbed as a stub, or the cut produced two clusters that did not merge) | `MERGE_PX`, or the fragment unit itself |
| `residual` | none of the above | **must be 0**; asserted, not reported |

The two categories the H-stroke-3 entry named are
`refused = gate_* + cost_break_even + over_max` and
`wrong = wrong_arm_*`, with `topology` a third that the registered two-way
split had no slot for. I am refining the registered split rather than forcing
a third cause into one of two boxes.

`rejoin + sum(buckets) == n` is asserted every run.

### IC-j7, the instrument check: five constructed trials, buckets written down first

Each is built by hand, not sampled, with the expected bucket recorded here
before the tool runs:

1. straight F, occluder at 70 deg, parts collinear -> **rejoined** (not a failure)
2. F bent 80 deg at the cut (`theta` 80 > 60) -> **`gate_theta`**
3. F's two parts offset sideways by 6px (`d` 6 > 3), tangents parallel -> **`gate_d`**
4. F bent 55 deg with a 0.5px offset: `55/30 + 0.5/1.5 = 2.17 >= 2.0`, both
   gates pass -> **`cost_break_even`**
5. F straight, with a second straight stroke entering the node as a better
   continuation of one part than the other part is -> **`wrong_arm_other`**

C1: all five land in the written bucket, or nothing is read.

### Counterfactual confirmation

A bucket label is a claim about *cause*, so each is tested by removing exactly
that cause at that node and re-running the node's matching:

- `gate_*` -> `THETA_MAX = D_MAX = inf` for that node only
- `cost_break_even` -> `C_UNMATCHED = 1e9` (pairing always beats leaving arms free)
- `wrong_arm_*` -> delete the occluder's arms from the node
- `topology`, `over_max` -> no intervention; reported as unflippable by construction

**flip** = the two true parts end in the same chain afterwards.

### Negative controls

- **NC-j7b, shuffled attribution.** Each failed trial is given a bucket drawn
  at random from the observed bucket distribution and receives *that*
  intervention. Named error: the bucket is being read off something that
  merely correlates with a trial's difficulty rather than the blocking cause,
  in which case any intervention flips about as often as the right one.
- **NC-j7c, intervention on trials that already rejoined.** Named error: the
  intervention is just "join more things", which would also damage trials that
  were never broken. Reported as the change in the `wrong` rate. No bar: this
  is the price of the repair, and it is what the follow-up registration will
  have to pay.
- **NC-j7a is deliberately not run.** Applying bucket X's intervention to
  bucket Y is a construction-zero in both directions -- relaxing a gate cannot
  change a pair that was never gated, and raising `C_UNMATCHED` cannot make a
  gated pair a candidate. A control whose answer is fixed by construction is
  an identity, not evidence, and is named as one here rather than printed as a
  pass.

### Pass criteria, fixed now and not to be loosened

- **C1** IC-j7: 5/5 constructed trials in the written bucket. Fail -> nothing read.
- **C2** `rejoin + sum(buckets) == n` and `residual == 0`.
- **C3** the split is **actionable** only if, for a bucket holding at least
  10% of the failures, the matched intervention flips **>= 60%** of that
  bucket **and** NC-j7b is **at least 20 points below** the matched flip rate
  pooled over buckets. If the gap is under 20 points the entry reports "the
  split is measured but not actionable" and no constant is changed from it.
- **C4** no constant moves in this entry whatever the result. A repair is a
  separate registration, scored on IC-j6's own `rejoin - wrong` on the
  calibration set, with the predicted direction of the H-stroke change
  written down before H-stroke is re-run.

### Visual

`results/stroke_chain_20261010/icj7_buckets.png`: one row per bucket, three
sampled failures each, showing the cut site with F in one colour, the
occluder in another, and the arms at the node marked. Seed 20261010.

shortcut: the diagnosis looks only at the node where the two parts meet, so a
failure caused two nodes away is attributed to `topology`; raised if `topology`
turns out to be a large bucket.

### IC-j7 amendments, all made before any real panel was read

The six constructed trials were built and run first; `--check` is a separate
entry point from `--root` precisely so this order is forced. Each amendment
and its reason is in the tool's docstring; in brief:

- **A1** all five counterfactual levels are printed, but the *matched* test per
  bucket stays the minimal one that targets its named cause (`gates`,
  `force`, `drop_rival`). Reason, arithmetic rather than empirical:
  `THETA_MAX / THETA0 = 60 / 30 = 2.0` is exactly the break-even, so a gated
  angle is already past it and `gates` alone can essentially never flip a
  `gate_theta` failure. Keeping the matched tests minimal is what keeps the
  buckets distinguishable and NC-j7b a real control.
- **A2** `wrong_arm_*`'s counterfactual is `drop_rival` (delete whatever arm
  the true arm was actually linked to), not "delete the occluder": IC-j7 case
  5's rival is an ordinary stroke and dropping the occluder changes nothing.
- **A3** the fixture's occluder moved from 70 deg to -70 deg. At 70 deg it sits
  10-20 deg from the second part of F on the bend cases, merges with it, and
  the node stops existing. The fixture was broken, not the classifier; no
  bucket threshold moved.
- **A4** case 4 is a plain 45 deg bend (theta 45.0, d 2.06, cost 2.87) instead
  of 55 deg with a 0.5px offset (2.17), which sat inside skeletonisation's own
  few degrees of play around the 2.0 break-even. The three inequalities that
  make it a `cost_break_even` by definition are asserted.
- **A5** a fifth level `rival_force` is reported. Measured reason: in case 5,
  removing the rival leaves the true pair at cost 2.90, still past break-even,
  so it still does not rejoin. **For that failure no single-cause repair
  exists.**
- **A6** a sixth trial, case 1b, was added as a **finding**: the straight
  crossing case 1 rejoins becomes `topology` at -55 deg because the skeleton
  puts two junction clusters there with a 1px fragment between them, and
  `fragments()` assigns both of that fragment's ends to the dominant cluster,
  so `merge_nodes` cannot union them. Written up in doc/shortcuts.md. No
  constant changed (C4).

**IC-j7: PASS, 6/6**, every bucket as written down, with the flip pattern the
amendments predicted: `gates` alone flips nothing gated, `force` flips the
ungated-but-expensive case, `both` flips all the refusals, and only
`rival_force` flips the wrong-arm case.

### IC-j7 first run, and two corrections before it was read as more than a diagnosis

Panels `[100:171]`, 182 usable trials, rejoin 0.632. **Result on record:**

| bucket | n | of failures |
|---|---:|---:|
| `topology` | 52 | **77.6%** |
| `gate_d` | 5 | 7.5% |
| `cost_break_even` | 5 | 7.5% |
| `wrong_arm_occluder` | 4 | 6.0% |
| `gate_both` | 1 | 1.5% |

C2 PASS (115 + 67 = 182, residual 0). C3 said **NOT ACTIONABLE**.

The registered two-way split -- "refused by the cost break-even" versus
"joined to the wrong arm" -- covers **22.4%** of the failures between them.
The thing I registered as a residual catch-all is three quarters of the
problem.

Two corrections, **A7**, neither of which touches a pass threshold:

- **(a)** `topology` was registered as "unflippable by construction". Wrong,
  and it is 77.6% of the failures. Its named cause -- A6's two clusters
  bridged by a fragment whose ends were both assigned to the dominant one --
  is removable and testable like any other. A sixth level `merge_touch`
  rebuilds the whole image's node map by unioning every cluster a fragment of
  at most `MERGE_PX` *touches*, using the neighbour sets `fragments()`
  discards, and re-runs every node. It flips IC-j7 case 1b (where the
  mechanism is known by construction) and flips none of cases 2, 4, 5, so it
  is specific. **Nothing is adopted; `stroke_chain.py` is untouched (C4).**
- **(b)** NC-j7b printed **0/0**, because the shuffled bucket was drawn from a
  distribution that is 78% `topology` and `topology` had no counterfactual, so
  nearly every draw produced "no test". **The `+46.7 point` gap that run
  printed is not evidence of anything** and is recorded here as the starved
  reading it was. The draw now comes only from buckets that have a
  counterfactual, the sample size prints beside the rate, and anything under
  20 samples is labelled STARVED in the output itself.

Also recorded from that run, and clean: **NC-j7c = 0/108.** No
already-rejoined trial is broken by removing the gates and forcing a pair at
its node. But this is **node-local**: a counterfactual at one node cannot see
damage done elsewhere, and `merge_touch` changes every node. The global price
is IC-j6's own `wrong` rate and belongs to the repair's registration, not
here.

The second run uses panels `[40:171]`, which is what the IC-j6 headline
(rejoin 0.654 / wrong 0.039) was measured on, so the split is comparable to
the number it is splitting. `[0:40]` stays untouched.

## 2026-10-11 IC-j7 result on the IC-j6 panel set

`[40:171]`, 1560 sites -> **311 usable trials**, rejoin **0.643** (the IC-j6
headline was 0.654 on 309 trials of the same panels and constants; the two
trials' difference comes from site ordering, and 0.643 is the figure the split
below is a split *of*). C2 PASS: 200 rejoined + 111 buckets = 311, residual 0.

| bucket | n | of failures | matched counterfactual | flip |
|---|---:|---:|---|---:|
| **`topology`** | 90 | **81.1%** | `merge_touch` | **37.8%** |
| `gate_d` | 7 | 6.3% | `gates` | 0% |
| `wrong_arm_occluder` | 7 | 6.3% | `drop_rival` | 57% |
| `cost_break_even` | 4 | 3.6% | `force` | 100% (**IDENTITY**, see below) |
| `gate_both` | 2 | 1.8% | `gates` | 0% |
| `wrong_arm_other` | 1 | 0.9% | `drop_rival` | 0% |

All other levels, for every bucket: `gates` 0% everywhere; `both` 86% / 50% /
100% / 57% / 100%; `merge_touch` **0% on every non-topology bucket**, so the
counterfactual is specific and is not simply joining more things.

### What the split says

1. **The registered question was the wrong question.** "Refused by the cost
   break-even" and "joined to the wrong arm" together are **11.8%** of the
   failures. Four fifths of them never get as far as a cost: the two halves of
   one cut stroke sit at nodes that no matching can reach across.
2. **The merge defect is real but partial.** `merge_touch` -- union every
   cluster a fragment of at most `MERGE_PX` *touches*, instead of the two its
   ends were assigned to -- flips **37.8%** of the topology failures, which is
   **30.6% of all failures** and would move rejoin from 0.643 to about 0.75 if
   it carried no price. The remaining **50% of all failures** are topology
   failures that this rule does not fix, and their cause is not yet named.
3. `gates` alone flips nothing anywhere, exactly as A1's arithmetic predicted
   (`THETA_MAX / THETA0 = 2.0` is the break-even, so a gated angle is already
   refused on cost). The angle gate is not a separate mechanism from the cost
   at these constants.
4. `force` flipping 100% of `cost_break_even` is **an identity**, labelled as
   one: those pairs are ungated, and forcing a pair at a node that has nothing
   else to offer leaves only one choice.

### Verdicts, with their thresholds

- **C1** IC-j7 6/6 constructed trials in the bucket written down -> **PASS**.
- **C2** exhaustive, residual 0 -> **PASS**.
- **C3** (a bucket holding >= 10% of failures must flip >= 60% under its own
  counterfactual, and NC-j7b must sit >= 20 points below the pooled 37.8%)
  -> **NOT ACTIONABLE**, on both legs:
  - `topology`, the only bucket over 10%, flips **37.8%**, under 60.
  - NC-j7b **30/96 = 31.2%, gap +6.6 points**, under 20.

**The second leg was unreachable by construction and I am recording that
rather than moving it.** With one bucket holding 81.1% of the failures, a
bucket drawn at random from the observed distribution *is the correct label*
81% of the time. From the printed numbers: a shuffled draw picks
`merge_touch` with p = 0.811, and `merge_touch` flips 37.8% of topology
trials and 0% of everything else, so the shuffled rate cannot fall much below
0.811 x 0.811 x 0.378 = 0.249, bounding the achievable gap at about **13
points** against a 20-point bar. This is the same defect as population A's
criterion in H-stroke-3: an absolute gap measured over a population one
category dominates. **Neither bar moves.** What the registration should have
required is a gap conditional on the bucket, and that is for the next entry
to register in advance, not for this one to adopt after seeing the result.

- **C4** no constant was changed. `stroke_chain.py` is byte-identical to the
  calibrated version; `merge_touch` lives only in `diagnose_icj6.py` as a
  counterfactual.

### Price

**NC-j7c = 0/391.** No already-rejoined trial is broken by removing the gates,
forcing a pair, or the touch merge. **Node-local only**: a counterfactual
applied at one node cannot see damage elsewhere, and `merge_touch` changes
every node in the image. The global price is IC-j6's own `wrong` rate (0.039
today) and belongs to the repair's own registration.

Visual: `results/stroke_chain_20261010/icj7_buckets.png` (one row per bucket,
three sampled failures each; F orange, occluder blue, the winning rival
magenta, the two true arm tips circled). Per-trial rows:
`results/stroke_chain_20261010/icj7_failures.csv`. Run log:
`results/stroke_chain_20261010/icj7_line_full.txt`.

### The next registration, named but not yet written

Split `topology` again. The 38% that `merge_touch` flips have a named and
measured cause. The 62% that it does not are the critical path now, and the
candidates to register against are: a true part absorbed as a stub under
`MERGE_PX`; the occluder's own junction swallowing a part whole; and more than
two clusters in one crossing. Pass criteria and controls to be fixed before
the tool is written, and the conditional-gap form of C3 to be registered
there.

## 2026-10-11 Pre-registration: IC-j8, splitting IC-j7's `topology` (81.1% of the rejoin failures)

Written before `diagnose_topology.py` exists. Nothing in this entry is
measured; the only numbers below are the ones already on record and the bars
derived from them.

The previous entry's split put 90 of 111 rejoin failures in one bucket,
`topology`: no merged node holds an arm of each half of the cut stroke, so no
matching of any kind can join them. `merge_touch` flips 37.8% of it and has a
named mechanism. The other 62% -- **about 50% of all IC-j6 rejoin failures**
-- has no named cause, and that is the critical path.

### The lesson from the last entry, applied in advance

Three defects in my own IC-j7 registration are being paid for here, not
repeated:

1. **No bucket is declared "unflippable by construction" without a
   falsifier.** `topology` was so declared and it was 81% of the data. Every
   sub-bucket below gets a counterfactual, and the one whose claim *is*
   structural gets `all_levels` (every repair at once) specifically so the
   claim can fail.
2. **Every sub-bucket is defined structurally, never by its outcome.** If
   `t_touch_split` were defined as "merge_touch fixes it", its flip rate would
   be 100% by construction. It is instead defined by the *signature* of the
   mis-assignment (a short fragment touching two clusters). Its flip rate is
   then a real measurement, and a trial that is merged and still refused on
   cost is a new finding rather than a bookkeeping error.
3. **C3 is registered in conditional form.** IC-j7's shuffled-label control
   was unreachable because one bucket held 81% of the population, which
   bounded the achievable gap at ~13 points against a 20-point bar. NC-j8b is
   therefore a **within-bucket** contrast: for each sub-bucket, its own
   matched repair against a repair drawn from the *other* sub-buckets, both
   measured on the same trials. Dominance cannot reach that comparison.

### The sub-buckets, fixed now, in this evaluation order

A trial is in `topology` iff `shared_node()` returns None: no merged node
carries an arm of f0 and an arm of f1. The cause is decided from `nodes`,
`absorbed`, `nmap` and the touch sets -- all already computed -- in this
order, and the order is part of the registration because T1 and T3 can
co-occur:

| # | bucket | structural definition | matched repair |
|---|---|---|---|
| T1 | `t_part_absorbed` | f0 or f1 is in `absorbed`: a true half of the stroke is at most `MERGE_PX` long and was swallowed into the node, so it has no arms at all | `keep_absorbed` |
| T2 | `t_no_node` | f0 or f1 touches no junction cluster at either end: the occluder left a free tip instead of a crossing | `free_end` |
| T3 | `t_touch_split` | both halves have nodes, the nodes are distinct, **and** some fragment of length <= `MERGE_PX` touches both clusters -- the signature of the `fragments()` mis-assignment (A6) | `merge_touch` |
| T4 | `t_bridge_long` | distinct nodes, no such short bridge, but a path exists between the two clusters in the fragment graph: the topology is connected and only `MERGE_PX` refuses it | `merge_px_inf` |
| T5 | `t_disconnected` | no path between the two clusters in the fragment graph | `all_levels`, as a falsifier |
| -- | `t_residual` | anything the five do not cover. **Must be 0.** | none |

### The repairs, and which of them are new mechanisms

- `keep_absorbed` -- union the clusters as now, but keep the absorbed fragment
  chainable, with both its arms at the merged node, instead of deleting it.
  **A defect repair:** the bridge's ink is currently dropped from every chain.
- `merge_touch` -- as built in IC-j7. **A defect repair.**
- `merge_px_inf` -- absorb a bridging fragment regardless of length. **A
  constant change, not a defect repair**, and it is labelled as one wherever
  it is printed: a pass here does not license adopting it.
- `free_end` -- match two free tips (node -1) to each other when they are
  within `D_MAX` and `THETA_MAX`, via a synthetic node holding both. **A new
  mechanism, not a defect repair**, and labelled as one. If T2 is large, the
  honest finding is that the chainer's node-only design, not a bug, is the
  limit.
- `all_levels` -- all of the above at once. Diagnostic everywhere; the
  registered falsifier for T5.

`stroke_chain.py` is not modified (C4 carries over). `merge_px_inf` and
`free_end` live in the diagnostic only.

### IC-j8, the instrument check: six constructed trials, buckets written first

Each is a crop drawn by hand, not a real panel, with the mechanism known by
construction. Expected sub-bucket and expected flip are written here before
the tool runs.

| case | construction | expected bucket | expected flip |
|---|---|---|---|
| A | IC-j7 case 1b verbatim: straight crossing, occluder at -55 deg | `t_touch_split` | `merge_touch` yes |
| B | straight stroke whose second half is 5px long (under `MERGE_PX`) between two clusters | `t_part_absorbed` | `keep_absorbed` yes |
| C | occluder laid along the stroke's end so it covers the tip without crossing it | `t_no_node` | `free_end` yes, `merge_touch` no |
| D | occluder at -20 deg, shallow, so the two crossings sit more than `MERGE_PX` apart along it | `t_bridge_long` | `merge_px_inf` yes, `merge_touch` no |
| E | occluder wide enough to erase a 10px span: the halves are separate skeleton components | `t_disconnected` | nothing flips, **including `all_levels`** |
| F | IC-j7 case 1 verbatim, occluder at -70 deg | not `topology` at all (rejoins); the sub-classifier must not be asked | n/a |

The bucket decision function is pure (`sub_bucket(...)`, taking the same
already-computed structures and no image), so every case above is also an
assert cell, and one assert carries a hand-computed value as required.

### The consistency check that anchors this to the previous run (lesson 12)

Already measured, by different code, before these sub-buckets existed: 34 of
the 90 `topology` trials flip under `merge_touch`. The touch union can only
help when a short fragment touches two clusters, so:

**Every trial that `merge_touch` flips must land in T3.** A flip landing in
T4 or T5 means the sub-bucketing is wrong, and the run asserts it rather than
printing it. This is the leg of the check that this tool did not produce.

### Negative controls, one named error each

- **NC-j8b, the within-bucket shuffle.** For each sub-bucket, apply a repair
  drawn uniformly from the *other* sub-buckets' matched repairs, on the same
  trials, same count. Named error: the sub-buckets are labels without
  content, and any repair helps any bucket equally. Sample size prints beside
  every rate; under 20 samples the tool itself prints **STARVED**, and a
  starved control makes its criterion NOT ACTIONABLE, never a pass.
- **NC-j8c, the local price.** All five repairs applied to the 200
  already-rejoined trials. Named error: `keep_absorbed` / `free_end` /
  `merge_px_inf` join things that currently chain correctly. Node-local, and
  the caveat is reprinted with the number.
- **NC-j8d, the global price.** The one NC-j7c could not see. A *copy*,
  `stroke_chain_touch.py`, carrying the touch-merge repair and nothing else,
  scored by `calibrate_stroke_chain.py` on the same panels `[40:171]` for
  both of IC-j6's numbers. Named error: two genuinely distinct crossings
  within `MERGE_PX` fuse into one node, so `wrong` rises while `rejoin` also
  rises, and a recall-only reading would call that a win.
- **No construction-zero is registered as a control.** An uncut trial has one
  part, so f0 == f1 and the trial is degenerate; NC-j7a was excluded for the
  same reason and is not reintroduced under a new name.

### Pass criteria, fixed now and not to be loosened

- **C1** 6/6 constructed trials in the sub-bucket written above, with the
  expected flips -> else the classifier is not trusted and no real number is
  read.
- **C2** exhaustive: T1..T5 sum to the `topology` count and `t_residual` is
  **0**.
- **C3 (conditional form)** for every sub-bucket holding >= 10% of
  `topology`: its own repair flips **>= 50%** of it, **and** each other
  bucket's repair flips **<= 20%** of it. The 50 is below IC-j7's 60 by
  deliberate choice and the reason is arithmetic, written before the fact: a
  merged node is necessary but not sufficient, since the pair must still beat
  the 2.0 break-even, and IC-j7 measured `force`-only flips at 100% on
  `cost_break_even` -- so a structural repair can be correct and still lose
  trials to cost. 50% is the point at which the structural cause is the
  majority cause for that bucket.
- **C3b** NC-j8b sits **>= 20 points** below the matched rate, within each
  bucket of >= 20 trials. Buckets under 20 are STARVED and NOT ACTIONABLE.
- **C4** `stroke_chain.py` byte-identical to the calibrated version; its
  SHA-256 is printed by the run.
- **C5** the anchor assert above holds.

### The adoption bars for the touch merge, fixed here and now

NC-j8d produces the numbers that decide whether the repair goes into
`stroke_chain.py` in the *next* registration. The bars are fixed in this
entry, before the measurement, so that they cannot be chosen to fit it.
Derivation from numbers already on record: 34 of 311 trials flip, so rejoin
has **+10.9 points** available if the repair were free; today's values on
these panels are rejoin 0.643 (0.654 at n=309), wrong 0.039, score
rejoin - wrong 0.615.

- rejoin **>= 0.70** (half of the available +10.9 points survives)
- wrong **<= 0.059** (today 0.039, an allowance of +2.0 points absolute)
- score **>= 0.675** (today 0.615, so +0.06)

**All three must hold.** If rejoin clears 0.70 but wrong exceeds 0.059, the
repair is **not adopted** and the headline of that entry is that the touch
merge buys recall with precision -- not a footnote.

### Splits, and what stays untouched

Panels `[40:171]`, `SEED = 20261010`, the same 311 trials the split above
splits. `[0:40]` remains untouched and unread. No constant is swept.

Visual, required before any number is quoted:
`results/stroke_chain_20261010/icj8_topology.png`, one row per sub-bucket,
three sampled trials each, with the two cluster labels drawn so the reader
can see the split node. Per-trial rows: `icj8_topology.csv`. Run log:
`icj8_line.txt`.

*shortcut: the sub-bucket is decided from the trial's own crossing only, as
in IC-j7; a failure whose real cause is a node elsewhere in the panel is
attributed here to the local structure. Raised when the chainer is scored
end to end on whole panels rather than per cut site.*

### IC-j8 amendments B1-B5, all made from code and arithmetic before the tool was written and before any real panel was read

**B1 `t_touch_split` is decided by the touch *closure*, not by a single
bridge.** Registered wording: "some fragment of length <= `MERGE_PX` touches
both clusters". That misses a crossing whose skeleton yields three clusters
chained by two short bridges, which is the same defect one step longer. The
test is now: under the touch union (`node_view_touch`, the IC-j7 level), do
the two halves share a node. It is still structural -- the union is computed
from the touch sets alone, before any matching runs -- and it makes C5's
anchor exact, since `merge_touch` can only flip a trial whose closure shares
a node.

**B2 `t_part_absorbed` is a construction zero and holds no criterion.**
Arithmetic, written out: `parts_of` requires `PART_MIN_PX = 8` ordered
pixels, `order_path` walks pixels and does not resample, so consecutive steps
are at least 1px and a true part has `path_len >= 7.0`, while `MERGE_PX =
6.0`. **Neither half of the cut stroke can ever be absorbed.** A bucket with
no possible members is not a bucket, for the same reason NC-j7a was not a
control. It stays in the enumeration as an **asserted zero** -- if it ever
fires, this arithmetic is wrong and the run says so -- and `keep_absorbed` is
struck with it.

**B3 `t_no_node` and `t_disconnected` are expected near-zero, and are kept
without a criterion.** The occluder is **added ink** (`ink | occ_mask` in
`make_trial`), not an erasure, so each half necessarily touches the
occluder's ink and a junction forms. No number is predicted; the expectation
is recorded so that a large count is read as a surprise rather than as
routine.

**B4 the registered T4 was nearly a tautology, and splitting it is the
substantive change.** `roots()` is transitive, so two halves chained through
an *intermediate* fragment already count as a rejoin. The occluder crosses
F by construction, so a path between the two halves almost always exists and
"a path exists" would have swallowed the bucket. What a `topology` failure
with a path actually means is that the relay was refused at a node. T4 is
therefore two buckets, and the distinction is one IC-j6's own scoring forces:

- **`t_relay_other`** -- a path exists that uses no occluder fragment.
  Matched repair **`relay_force`**: gates off and pairing forced at *every*
  node, which is strictly stronger than forcing only the path's nodes, so a
  0% here is conclusive and a high rate is an upper bound, and it is
  labelled as an upper bound wherever it is printed.
- **`t_relay_occluder_only`** -- every path between the halves uses an
  occluder fragment. **Rejoining through it is scored `wrong` by IC-j6's own
  definition**, so for these trials "rejoin" and "clean" cannot both be had
  by relaying, and the only repair is the direct one. No matched repair;
  `all_levels` is its registered falsifier.

Consequently **every flip is printed twice**: `flip` (the two halves end in
one chain, the registered criterion, unchanged) and `flip_clean` (that, and
no occluder fragment in the chain). Any divergence between the two goes in
the result's headline.

**B5 for the constructed trials the criterion is the structural label; the
flip column is determinate only where the case's mechanism makes it so.**
Cases A and C have a determinate expected flip and it is part of C1. For the
relay and disconnected fixtures the matching's behaviour at a hand-drawn node
is not determined by the mechanism under test, so their flips are printed as
diagnostics with their measured reason, as A5 did for `rival_force`. No
threshold moves; C1 is still 6/6 on the labels.

**`merge_px_inf` stays a diagnostic and is never a matched repair**, since
absorbing a bridge of any length is a constant change rather than a defect
repair, and `all_levels` deliberately excludes it so that the falsifier is
the strongest combination of *repairs*, not of constants. `free_end`'s radius
is `FREE_R = 8.0` px, registered here, and it too is labelled a new
mechanism.

Final bucket list, in evaluation order: `t_part_absorbed` (asserted zero) ->
`t_no_node` (`free_end`) -> `t_touch_split` (`merge_touch`) ->
`t_relay_other` (`relay_force`) -> `t_relay_occluder_only` (falsifier only)
-> `t_disconnected` (falsifier only) -> `t_residual`. The decision tree is
exhaustive by construction, so **`t_residual` is a vacuous zero and is
labelled as one**; the non-vacuous half of C2 is that the six buckets sum to
the `topology` count.

### IC-j8 amendment B6 and the instrument check: PASS

**B6 `relay_only`, a sixth level, added because IC-j8 case D measured
something the registration had assumed away: forcing a pair does not force
the *right* pair.** Case D is two junctions with a bent relay of non-occluder
ink between them, mechanism known by construction. `relay_force` -- gates off
and pairing forced at every node, which B4 called an upper bound -- **does
not** rejoin it: at a three-arm node the forced pair goes to the cheaper
rival, not to the relay. So `relay_force` is an upper bound on *that* repair
and not on "a relay chain could be formed". `relay_only` additionally drops
every rival arm at the nodes along the shortest non-occluder path, and it
flips case D. It is a diagnostic, never a matched repair, for the same reason
`drop_rival` was: dropping the rivals needs knowledge the chainer does not
have. No threshold moves; `t_relay_other`'s matched repair is still
`relay_force`, and the two columns are printed side by side.

Also measured on case A and recorded: `relay_force` leaves the trial unjoined
**and dirty** -- the forced pair links a true half to the occluder. Forcing is
not free, and the `flip_clean` column of B4 is what makes that visible.

**IC-j8: PASS.** 5 constructed trials in the sub-bucket written down in the
registration, plus the B2 arithmetic asserted:

| case | construction | sub-bucket | criterion flip |
|---|---|---|---|
| A | straight crossing at -55 deg (IC-j7 case 1b) | `t_touch_split` | `merge_touch` Y |
| C | 6px gap, occluder touching nothing | `t_no_node` | `free_end` Y |
| D | two junctions, bent relay of other ink | `t_relay_other` | diagnostic (B5): `relay_force` n, `relay_only` Y |
| E | two crossings in separate components | `t_disconnected` | nothing flips, `all_levels` included |
| F | the same crossing at -70 deg (IC-j7 case 1) | not `topology` (rejoins) | n/a |

Case E's first fixture was unusable, not misclassified: the crossing stroke
was drawn through the middle of each half, which split it into two parts and
gave 4 F parts instead of 2. The crossing now sits at each half's end. The
fixture was broken, not the classifier, and no label or threshold moved.

`stroke_chain.py` sha256 `e9e49267c339f885`, unmodified (C4).

## 2026-10-12 IC-j8 result: the 46% is the crossing's own middle ridge, and `MERGE_PX` was never swept

Panels `[40:171]`, the same 311 trials, rejoin 0.643, 111 failures, 90
`topology` (81.1%). **IC-j8 PASS** (5 constructed trials in the sub-bucket
written in the registration, plus the B2 arithmetic asserted). **C2 PASS**
(sub-buckets sum 90 = topology 90; `t_residual` 0 is VACUOUS and
`t_part_absorbed` 0 is B2's asserted construction zero). **C5 anchor PASS**
(0 violations: every `merge_touch` flip landed in `t_touch_split`, which was
measured by different code four days earlier).

| sub-bucket | n | of topology | of all failures | matched repair | flip / flip_clean |
|---|---:|---:|---:|---|---|
| `t_touch_split` | 37 | 41.1% | 33.3% | `merge_touch` | **92% / 92%** |
| `t_relay_other` | 51 | **56.7%** | **46.0%** | `relay_force` | 10% / 10% |
| `t_relay_occluder_only` | 1 | 1.1% | 0.9% | none, by registration | 100% / **0%** |
| `t_disconnected` | 1 | 1.1% | 0.9% | none | 0% / 0% |
| `t_no_node` | 0 | -- | -- | -- | B3's expectation held |
| `t_part_absorbed` | 0 | -- | -- | -- | B2 construction zero, asserted |

### Verdicts, with their thresholds

- **C3 conditional, `t_touch_split`: PASS.** Own repair 91.9% (bar 50%),
  every other bucket's repair 0.0% (bar 20%), NC-j8b within-bucket shuffle
  **0/37**, gap **+91.9 points** (bar 20). This is the first time a control of
  this family has cleared its bar in this thread, and the reason is the form,
  not the data: the contrast is inside the bucket, so the 81% dominance that
  made IC-j7's C3 unreachable cannot reach it.
- **C3 conditional, `t_relay_other`: NOT ACTIONABLE**, on the own-rate leg
  only -- 9.8% against a 50% bar. Its specificity leg is clean (0.0%).
- **C4** `stroke_chain.py` sha256 `e9e49267c339f885`, unmodified.

### Where the relay is -- the question the bucket's definition could not answer

`t_relay_other` says only that *some* non-occluder path exists, and two very
different things satisfy that: the crossing's own middle ridge (the right
join) or a detour through the rest of the drawing (which would satisfy
IC-j6's success test by connectivity, not by the join we want). The
registration did not distinguish them. **Measured, as a distribution:**

| | min | q25 | median | q75 | max |
|---|---:|---:|---:|---:|---:|
| fragments on the path | 1.0 | 1.0 | **1.0** | 1.0 | 1.0 |
| their length (px) | 6.2 | 7.7 | 8.1 | 9.2 | 11.5 |
| farthest stray from the cut (px) | 3.4 | 3.9 | 4.5 | 5.1 | 6.8 |
| gap between the facing tips (px) | 11.2 | 13.0 | 14.8 | 17.1 | 19.9 |

**51/51 are a single fragment, and 51/51 lie within 20px of the cut.** The
detour reading is refuted: every one of them is the crossing's own middle
ridge -- the piece where F's ink and the occluder's ink fused, which
`parts_of` classifies as neither part (it is not 80% inside either mask), so
it is invisible to the trial's ground truth and sits between the two halves.
Chaining f0 -> ridge -> f1 is the correct join and it is clean.

### The finding this turns into: `MERGE_PX` was never swept

The ridge is **6.2 to 11.5px** long. `MERGE_PX = 6.0`. These 51 failures are
fragments that missed the absorption threshold by between 0.2 and 5.5px.

And the calibration grid (972 combinations, `GRID` in
`calibrate_stroke_chain.py`) swept `TAN_SKIP`, `TAN_WIN`, `KAPPA_REF`,
`THETA_MAX`, `D_MAX`, `W_KAPPA`. **`MERGE_PX` and `C_UNMATCHED` are not in
it.** The two constants that now look load-bearing -- the absorption
threshold and the break-even that IC-j7 showed makes the angle gate
redundant -- have never been varied, and the calibrated values are the ones
they were written with.

This is not a licence to raise it. `merge_px_inf` (absorb a bridge of any
length) flips 45% of `t_relay_other` but **breaks 111 of the 200 trials that
already rejoin**, because it absorbs real strokes. An unbounded loosening is
refused by measurement. A *bounded* sweep is the next registration.

### NC-j8c, the price, corrected

| level | breaks | newly dirty |
|---|---:|---:|
| `merge_touch` | 0/200 | 0/200 |
| `free_end` | 0/200 | 0/200 |
| `relay_force` | 0/200 | 0/200 |
| `relay_only` | 0/200 | 0/200 |
| `merge_px_inf` | **111/200 = 55.5%** | 2/200 |
| `all_levels` | 0/200 | 1/200 |

The first run printed `relay_only` breaking 126/200 = 63%. **That was my
artifact, not a price:** on a trial whose halves already share a node there is
no relay path, the function returned "no path", and the control read it as a
breakage. Corrected to "drop the rivals at the shared node and force a pair",
which is the defined intervention there. Recorded because the 63% was printed
once and is not evidence of anything.

### NC-j8d, the global price of the touch merge: NOT ADOPTED

Same 311 trials (asserted: same fragments, same F parts, same occluder
parts), all 11 constants identical in both files (asserted).
`stroke_chain_touch.py` sha256 `091539ea88cfe012`.

| arm | n | rejoin | wrong | score |
|---|---:|---:|---:|---:|
| `stroke_chain` (calibrated) | 311 | 0.643 | 0.061 | 0.582 |
| **+ touch merge (candidate)** | 311 | **0.752** | 0.061 | **0.691** |
| `force`, gates off (**IDENTITY**) | 311 | 0.714 | 0.193 | 0.521 |

Trial by trial, not only by count: **rejoined gained 34, LOST 0; wrong newly
0, no longer wrong 0.** The repair is strictly additive on this trial set.

- rejoin 0.752 vs bar >= 0.700 -> PASS
- wrong 0.061 vs bar <= 0.059 -> **FAIL**
- score 0.691 vs bar >= 0.675 -> PASS
- all three required -> **NOT ADOPTED**

**The bar that failed is one the baseline itself cannot clear, and that is my
error, recorded rather than repaired.** `wrong <= 0.059` was derived as
"0.039 + 2.0 points", and the 0.039 came from `icj6_grid` at **n = 309**, a
different trial set. On these 311 trials the baseline's `wrong` is **0.061**.
The candidate's effect on `wrong` is **zero by set, not merely by count**,
and it still fails, because the comparison point was imported from another
measurement. **This is the third instance of one defect class:**

1. H-stroke-3 population A -- an absolute gap over a population one category
   dominates;
2. IC-j7 C3's second leg -- unreachable at 81% dominance, bound ~13 points
   against a 20-point bar;
3. here -- a bar derived from a number measured on a different trial set.

**The bar does not move and the verdict stands at NOT ADOPTED.** The
measurement already exists; what the next entry registers is the bar in the
form it should have had -- a delta against the *same-trial* baseline
(`wrong - wrong_base <= +0.010`) -- and the adoption decision belongs to that
entry, not to this one.

Also on record from that run: the candidate beats the trivial solution on
both axes (rejoin 0.752 > 0.714, `wrong` 0.061 against 0.193, a third), and
IC-j7's node-local prediction of "+10.9 points if free" came out at exactly
+10.9 with **LOST 0**, so a node-local counterfactual was a sound predictor
of a global change here.

Visual: `results/stroke_chain_20261010/icj8_topology.png`. Per-trial:
`icj8_topology.csv`. Logs: `icj8_line_v2.txt`, `icj8_ncd_v2.txt`.

### The next registration, named and not yet written

1. **`MERGE_PX` as a swept axis**, bounded (6, 8, 10, 12, 16), crossed with
   the touch merge on/off, both of IC-j6's numbers plus the absorbed-px total,
   because absorbed ink joins no stroke and that damage is invisible to
   `rejoin`. `C_UNMATCHED` joins the sweep for the same reason: it was never
   varied either.
2. **The `wrong` bar in delta form**, as above, with the touch-merge adoption
   decided under it.
3. **The relay repair is not a `merge_nodes` fix.** `match_node` already
   enumerates every pairing at the node exhaustively up to `MAX_ARMS`, so it
   is optimal *there*; the ridge loses to the occluder's arm on a cost that is
   node-local while the evidence (f0 -> ridge -> f1 is straight, f0 -> occluder
   is a corner) is chain-local. Registering that needs a two-node cost, and it
   is a design change, not a defect repair.

## 2026-10-16 Stage 2 M0, read in the registered order: every non-learned arm falls to NC3, and none beats "don't paint"

`results/fill_koma_20261010/per_panel_held.csv` (969 held panels, 13,566
rows) completed 2026-10-11 and **had not been read until today**. Read now
through `report_fill_koma.py`, which prints the negative controls before the
arms (C4) and keeps the strata apart (C5). Run log:
`results/fill_koma_20261010/report_held.txt`.

### Normal stratum, 956 panels -- the controls, first

| control | region-F1 | pixel IoU | pred area | named error |
|---|---:|---:|---:|---|
| `nc1_polarity` | 0.0035 | 0.0987 | 0.9467 | the rough read white-on-black |
| `nc3_hflip:rough_dark` | 0.0063 | 0.0063 | 0.0069 | mirrored left-right |
| `nc3_hflip:rough_dark@budget` | 0.0104 | 0.0311 | 0.0394 | mirrored left-right |
| `nc3_hflip:freq_table` | 0.0084 | 0.0021 | 0.0000 | mirrored left-right |
| `nc5_shift50:*` | 0.0063 / 0.0104 / 0.0084 | | | rolled 50px, sensitivity only |

### Then the arms

| arm | region-F1 | pixel IoU | pred area | GT area | pred ratio | GT ratio | verdict, with its threshold |
|---|---:|---:|---:|---:|---:|---:|---|
| `blank` | 0.0084 | 0.0021 | 0.0000 | 0.1040 | 0.0406 | 0.5193 | **IDENTITY**: 0 unless the GT is empty |
| `all_black` | 0.0037 | 0.1040 | 1.0000 | 0.1040 | 0.0406 | 0.5193 | **IDENTITY**: pixel IoU = \|G\|/\|P\| |
| `rough_dark` | 0.0065 | 0.0139 | 0.0069 | 0.1040 | 0.0406 | 0.5193 | **falls to NC3** (gap +0.000 < 0.05) |
| `rough_dark@budget` | 0.0110 | 0.0451 | 0.0394 | 0.1040 | 0.0406 | 0.5193 | **falls to NC3** (gap +0.001 < 0.05) |
| `freq_table` | 0.0084 | 0.0021 | **0.0000** | 0.1040 | 0.0406 | 0.5193 | **falls to NC3** (gap +0.000 < 0.05) |

- **C1 ("beats blank" := region-F1 - blank >= 0.05): all three arms FAIL.**
  `rough_dark` is 0.0065 against blank's 0.0084, i.e. **-0.0019: worse than
  painting nothing.** `rough_dark@budget` is +0.0026 and `freq_table` +0.0000,
  both far under the 0.05 margin.
- **`freq_table` is an identity and must be labelled as one.** Its predicted
  area is **0.0000**: no (y bin, area bin) cell ever reached 0.5, so the arm
  predicts nothing and *is* `blank`. Its exact tie with blank at 0.0084 is
  that identity, not a result. Lesson 14's required no-training control
  therefore returned "no cell is more often filled than not", which is
  itself the informative part.
- **The arm that mattered has answered: roughs do not mark the fill.** The GT
  is **10.4% of panel area and 52.0% of the ink** (both readings, together),
  while the rough's own dark ink is **0.69% of the area**. There is a great
  deal of fill to predict and the rough does not carry it. The stop rule's
  release condition -- "if `rough_dark` alone satisfies C1 and C3, the GPU
  slot is released" -- is **not met**, so the slot is not released.

### NC2 is void: a control with no samples is not a control

**`nc2_offbyone` produced 0 rows of 13,566.** The scorer skips the pair when
`pred.shape != gt.shape` ("panels differ in size; skip, do not pad" -- my own
registered choice), and no two consecutive held panels share a shape. So
"panel i's prediction against panel i+1's GT" is **unmeasured**, and nothing
in this entry rests on it. This is the second control of this family to come
back starved after NC-j7b's 0/0, and the same rule applies: it is reported as
void, not as a pass. Repair for the next registration: pair each panel with
the nearest *same-shape* held panel by sorted name, or crop both to the common
top-left overlap -- and whichever is chosen, print the sample count beside the
rate.

### Near-blank stratum, 13 panels, kept apart (C5)

`blank` 0.6923, `rough_dark@budget` 0.6923, `freq_table` 0.6923,
`rough_dark` 0.3846, `all_black` 0.0000. The three at 0.6923 all predict
nothing here, so this is the same identity again: most near-blank panels have
no fill and painting nothing is right 9 of 13 times. The hflip controls score
identically for the same reason. **No verdict is read from this stratum**, and
it is never averaged with the 956.

### What this changes

1. **Training is warranted and the registered gate is now passed in the only
   direction it could be: nothing non-learned predicts fill regions.** The
   number to beat is `blank`'s **0.0084**, with the instrument's ceiling at
   **1.0** (IC3, GT as prediction, 60/60 panels).
2. **But the whole non-learned table lives between 0.0037 and 0.0110**, and
   the mirrored control sits inside that range. Before a four-day slot is
   spent, one pre-flight is required and is registered here: **IC6, can
   region-F1 at `MIN_REGION_PX = 500` / `MATCH_IOU = 0.5` see a partial win?**
   Feed the GT with a known fraction of its regions deleted (keep 25%, 50%,
   75%) and with each region eroded by a known radius, expected values
   written down first (keeping half the regions and matching all of them is
   precision 1.0, recall 0.5, region-F1 0.667). If a half-correct prediction
   does not land near 0.667, the metric cannot grade the thing being trained
   and the trainer waits. **No GPU time is spent before IC6 is read.**
3. The GPU slot of 2026-10-12..10-15 (Mon-Thu) **passed unused**. The next
   standard slot is **2026-10-19..10-22**. The operational cause is recorded
   in the entry below, not excused.

### Why this sat for five days, recorded as a process failure

H-stroke-3 put splitting IC-j6's failures on the critical path, so taking up
the chainer was a registered choice, not drift. Reading this CSV was not:
`report_fill_koma.py` costs seconds, it is the gate that decides whether the
GPU slot is needed at all, and it existed since 2026-10-10. It should have
been run before the chainer work, and the four-day slot was lost to that
ordering. Recorded here so the ordering rule is explicit: **a cheap gate that
can release or claim a scheduled slot is read before any expensive branch,
regardless of what the critical path says.**

## 2026-10-16 Pre-registration: IC6, can region-F1 grade a partially correct prediction?

Written before `check_region_f1_partial.py` exists. The stage-2 entry above
registered this as a pre-flight and the condition stands: **no GPU time is
spent before IC6 is read.** The reason it is needed: the entire non-learned
table sits between 0.0037 and 0.0110 with the mirrored control inside that
band, so nothing yet shows that region-F1 at `MIN_REGION_PX = 500` /
`MATCH_IOU = 0.5` can put a half-correct prediction anywhere between
`blank`'s 0.0084 and IC3's ceiling of 1.0.

`region_metrics.py` (sha256 `8d9fd437ec7641e0`) and `fill_mask.py` (sha256
`d914087a44e2c64e`) are used unmodified and their hashes are printed by the
run. Built on the real held-side GT fill masks, not synthetic discs: IC5
already did discs, and what is in question here is the real region-size
distribution.

### The census the expectations are built on, measured first

This is a **descriptive census with no pass criterion**. It fixes the x-axis
of the erosion sweep and the per-panel arithmetic, and nothing else; it grades
no arm and no hypothesis. Measured on the held side, `fill_mask(ink_of(line))`
exactly as `score_fill_koma.py` builds it:

- **956 normal panels, 13 near-blank** (the stage-2 strata, unchanged).
- **22,602 regions >= 500px** in the normal stratum. Per panel: mean 23.64,
  median 17, max 187. **948 panels have >= 1 region; 8 have none.**
- Region area percentiles, with the equivalent disc radius
  `R = sqrt(A/pi)` and the analytic half-match radius `0.2929R` beside each:

| percentile | area | R | 0.2929R |
|---|---:|---:|---:|
| p0 | 500px | 12.62 | 3.70 |
| p25 | 796px | 15.92 | 4.66 |
| **p50** | **1500px** | **21.85** | **6.40** |
| p75 | 4058px | 35.94 | 10.53 |
| p90 | 17187px | 73.96 | 21.67 |
| p100 | 3948154px | 1121.04 | 328.47 |

- **GT fill area living in sub-500px components: 8,398,055 / 417,412,042 =
  2.0%.** The `>= 500px` filter therefore grades 98.0% of the fill *area*,
  and this is the bound on what any trained model can score through this
  metric. It is small, which is the answer to "does the filter grade only
  large regions": by area it does not.
- **The near-blank stratum cannot support this check and no number will be
  printed for it.** Its 13 panels hold region counts
  `[0,0,1,1,0,0,0,0,0,3,3,0,0]`: nine have no regions at all, so there is
  nothing to retain or erode, and the four that do cannot carry a retention
  fraction. It is reported as unsupported, not as a value.

### Leg 1, retention -- and it is an IDENTITY, stated before it is run

Of the GT regions surviving the `>= 500px` filter, keep a fraction k and use
the kept components, pixel-exact, as the prediction.

- Selection rule, fixed mechanically: the region ids sorted ascending, then
  shuffled by `random.Random(f"IC6:{panel_name}")`, then the first `n_keep`
  taken. Seeded per panel, so it depends on nothing global and on no ordering
  of the panel list.
- `n_keep = int(round(k * n))`, Python's round-half-to-even. Note in advance:
  a panel with n = 1 at k = 0.50 keeps **zero** regions, because
  `round(0.5) == 0`. 33 panels keep zero at k = 0.25 and 19 at k = 0.50;
  nothing is special-cased for them and the per-panel expectation already
  accounts for it.
- The 8 panels with n = 0 score region-F1 **1.0** by `match`'s "no GT and no
  prediction" rule. **That is an identity**, it has nothing to do with partial
  credit, and those panels are excluded from the retention statistics and
  reported separately by count.

**This leg is an identity and must be labelled as one wherever it is
printed.** A pixel-exact subset of GT components gives, provably without
running: `n_pred = n_keep` (every kept component was already >= 500px), each
kept component overlaps only its own GT component (components are disjoint)
and at IoU 1.0, so `matched = n_keep`, `precision = 1.0`,
`recall = n_keep / n`, and

    region-F1 = 2 * n_keep / (n + n_keep)

So this leg does **not** discover anything about the metric; it checks that
the implementation -- the `>= 500px` filter, `regions`, the greedy matcher,
`score_panel`'s plumbing -- actually realises that identity. That is what an
instrument check is for, and it would catch a filter that drops a kept
region, a matcher that fails to pair a region with its own copy, or a shape
bug. Expected values, computed by hand from the census before the run:

| k | nominal 2k/(1+k) | per-panel mean over the 948 panels with n >= 1 | panels keeping zero |
|---|---:|---:|---:|
| 0.25 | 0.400 | **0.3885** | 33 |
| 0.50 | 0.667 | **0.6552** | 19 |
| 0.75 | 0.857 | **0.8602** | 0 |

The per-panel figures differ from the nominal ones **only** by the rounding of
`k * n` at small n; that is arithmetic, not a discrepancy, and it is written
here so it cannot be read as one later.

### Leg 2, erosion -- the leg that actually answers the question

Erode every GT region by a disc of radius e. This prediction is *not* a subset
of components, so nothing is pinned at 1.0 and the result is a real
measurement.

Analytic model, written first: a disc of radius R eroded by e has
`IoU = ((R-e)/R)^2`, so it survives `IoU >= 0.5` while
`e <= R(1 - 1/sqrt(2)) = 0.2929R`. A region therefore stops matching when its
area falls below `pi*(e/0.2929)^2`. With `n_pred = n` and `matched = m`,
precision and recall are both `m/n`, so **predicted region-F1 = the fraction
of regions still matching.** Radii chosen to straddle the measured median's
0.2929R = 6.40px and the quartiles' 4.66 and 10.53:

| e | area threshold | predicted still matching | predicted region-F1 |
|---:|---:|---:|---:|
| 1 | 37px | 100.0% | 1.000 |
| 2 | 146px | 100.0% | 1.000 |
| 4 | 586px | 90.8% | 0.908 |
| 6 | 1318px | 54.5% | 0.545 |
| 8 | 2344px | 36.3% | 0.363 |
| 12 | 5273px | 20.9% | 0.209 |

**Predicted half-match radius: e = 6.40px**, which is `0.2929 *` the median R
by construction.

**Expected deviations, with their direction, written before the run.** Real
fill regions are not discs -- hair strands and clothing folds are elongated --
and an elongated region loses its match at a far smaller e than its
equivalent-disc radius implies. Two further effects are not in the model at
all: an eroded region can **fall below 500px** and vanish from the prediction
(`n_pred` drops, precision rises, recall falls) and it can **split** into
several components (`n_pred` rises, precision falls). So the measurement is
expected to sit **below** the table above, and by how much is the finding.
Vanish and split counts are printed per radius. **The deviation is reported
against the prediction; agreement is not asserted.**

### Pass criteria, fixed now and not to be loosened

- **C1** at each retention level, the measured mean region-F1 is within
  **0.02** of the per-panel expectation above, over **>= 200** normal-stratum
  panels. Otherwise region-F1 is not grading what it is assumed to grade and
  the trainer waits.
- **C1b**, additionally and more strictly, since the expectation is an
  identity: **0 panels** may deviate from `2*n_keep/(n+n_keep)` at all. C1
  stays as written; C1b is reported beside it and a C1b failure names the
  panels.
- **C2** the three retention levels are strictly ordered and separated by
  **>= 0.15** (the expectations are 0.267 and 0.190 apart). This is what "can
  see a partial win" means operationally. For the record, fixed here:
  `blank` scores **0.0084** and the mirrored control `nc3_hflip:rough_dark`
  **0.0063**, so all three retention levels sit **more than 0.37 above the
  entire 0.0037-0.0110 band**. If C2 passes, that band is not the metric's
  floor being insensitive -- it is the arms genuinely predicting almost no
  correct region.
- **C3** the erosion match rate is **monotonically non-increasing in e**, and
  the radius at which half the regions stop matching is reported beside the
  predicted **6.40px**. No bar is set on the *distance* between measured and
  predicted, because the disc model is known to be wrong for real shapes;
  the bar is on monotonicity, which a working metric must have.
- **One assert cell with hand-computed values** runs at import: a 40x40 square
  (1600px, above the filter) kept and dropped, giving region-F1 exactly 2/3
  for 1 of 2 kept; and a disc of R = 20 eroded by e = 6 giving
  `IoU = (14/20)^2 = 0.49 < 0.5` so it must stop matching, while e = 5 gives
  `(15/20)^2 = 0.5625 >= 0.5` so it must survive.

### Visual, and what is not tested

Montage `results/fill_koma_20261010/ic6_degradation.png`: one row per
retention level and per erosion radius, GT beside the degraded prediction, so
that the degradation is visibly what it claims to be.

**Not tested here, and named so it is not mistaken for covered:** over-
painting. Erosion only probes a prediction that is too small; a dilated
prediction survives `IoU >= 0.5` while `d <= 0.414R`, and that leg is not in
this registration. Displacement is already probed by NC3 and NC5 in the
stage-2 table. Both are for a later entry.

### IC6 amendments E1-E2, made from the assert cell before any real panel was read

**E1 the assert cell's disc case at e = 6 was a knife-edge and is replaced by
e = 8.** The cell failed on its first run, which is what it is for. Measured
reason: `dist > e` *is* the correct morphological erosion by a disc of radius
e, but cv2's `DIST_L2` maskSize-3 transform is an approximation --
`fill_mask.py` documents this on its own legs -- and the **effective erosion
measures e - 0.25px**, calibrated at R in {20, 40, 80} for e in 1..12
(e = 1 -> 0.91, e = 2 -> 1.83, e = 4 -> 3.76, e = 6 -> 5.72, e = 8 -> 7.84,
e = 12 -> 12.00). Measured IoU therefore sits 0.01-0.02 **above**
`((R-e)/R)^2`, and at R = 20, e = 6 that moves 0.4901 to **0.5099**, across
the matching threshold. The assert now uses e = 5 (analytic 0.5625, must
match) and e = 8 (analytic 0.3600, must not), and the calibration itself is
asserted as a **bounded** claim -- effective erosion in [e-0.5, e] at R = 40
-- rather than as a point value. **No pass criterion moves.**

**E2 the erosion table prints a calibrated column beside the registered
prediction.** Boundary `e - 0.25 <= 0.2929R` instead of `e <= 0.2929R`. The
**registered column stays the comparison of record**; C3 is on monotonicity,
so no bar moves. The column exists so that a known 0.25px offset is not
quietly absorbed into "the disc model overshoots".

## 2026-10-16 IC6 result: region-F1 CAN grade a half-correct prediction, and it is harsh on boundary accuracy

`region_metrics.py` sha256 `8d9fd437ec7641e0`, `fill_mask.py` sha256
`d914087a44e2c64e`, both unmodified. Run log
`results/fill_koma_20261010/ic6_run.txt`.

**956 normal panels, 13 near-blank skipped, 8 normal panels with 0 regions
excluded** (their region-F1 of 1.0 is the "no GT and no prediction" identity
and has nothing to do with partial credit). **948 panels used, 22,602
regions, median area 1500px.**

### Leg 1, retention -- the identity, realised exactly

| k | measured | registered expectation | nominal 2k/(1+k) | \|diff\| | C1b max deviation |
|---|---:|---:|---:|---:|---:|
| 0.25 | **0.3885** | 0.3885 | 0.400 | **0.0000** | 5.6e-17 |
| 0.50 | **0.6552** | 0.6552 | 0.667 | **0.0000** | 1.1e-16 |
| 0.75 | **0.8602** | 0.8602 | 0.857 | **0.0000** | 1.1e-16 |

- **C1 PASS** (within 0.02 of the registered per-panel expectation, over
  >= 200 panels: 948 used, diff 0.0000 at every level).
- **C1b PASS** (every panel exactly on `2*n_keep/(n+n_keep)`; the largest
  deviation anywhere is machine epsilon).
- **C2 PASS**: gaps **0.267** and **0.205**, both >= 0.15.

**What is and is not an identity here, stated plainly.** The retention values
are arithmetic: a pixel-exact subset of GT components gives
`2*n_keep/(n+n_keep)` provably, and C2's *separation* follows from that
arithmetic too. Neither is a discovery. What is **not** an identity is C1b --
that the implementation realises the identity on all 948 real panels, with
the `>= 500px` filter, the greedy matcher and `score_panel`'s plumbing in the
path -- and that is now established at machine epsilon. And C2's informative
half is not the separation but the **distance to the measured floor**:
`blank` **0.0084** and `nc3_hflip:rough_dark` **0.0063** sit **0.380 and
0.382 below the lowest retention level.** So the stage-2 band of
0.0037-0.0110 is **not** the metric being blind in that range; it is the arms
genuinely predicting almost no correct region.

### Leg 2, erosion -- the leg that measures, and the disc model fails in the registered direction

| e | measured F1 | predicted (registered) | calibrated (E2) | match rate | pred/gt regions | diff |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9091 | 1.000 | 1.000 | 0.856 | 0.895 | -0.091 |
| 2 | 0.8223 | 1.000 | 1.000 | 0.719 | 0.778 | -0.178 |
| 4 | 0.3821 | 0.908 | 0.983 | 0.255 | 0.525 | **-0.526** |
| 6 | 0.2546 | 0.545 | 0.577 | 0.147 | 0.376 | -0.291 |
| 8 | 0.1986 | 0.363 | 0.380 | 0.105 | 0.313 | -0.165 |
| 12 | 0.1438 | 0.209 | 0.216 | 0.068 | 0.253 | -0.066 |

- **C3 PASS**: the match rate is monotonically non-increasing in e.
- **Half the regions stop matching at e <= 4px**, against the registered
  prediction of **6.40px**. (The median `0.2929R` over the regions actually
  used is 6.40px, identical to the registration, so the two columns are
  comparing the same region set.)
- **`pred/gt regions` is below 1 at every radius and falls monotonically,
  0.895 -> 0.253.** Of the two unmodelled effects named in the registration,
  **vanishing happened and splitting did not**: eroded regions drop below the
  `>= 500px` filter rather than breaking into pieces. At e = 1 -- a single
  pixel -- **10.5% of regions have already vanished and 14.4% no longer
  match.**

The disc model overshoots by up to **0.526** in F1, far beyond E2's 0.25px
calibration, and the reason is visible in the montage: the regions that die
first are the **long thin sweeps** -- hair, folds -- whose equivalent-disc
radius badly overstates their width. This was registered as the expected
direction and its size is now measured.

### Verdict

**region-F1 at `MIN_REGION_PX = 500` / `MATCH_IOU = 0.5` CAN grade a
half-correct prediction, measured on 948 normal-stratum panels: a prediction
holding half the regions scores 0.655 against `blank`'s 0.0084.** The
instrument is not the reason the stage-2 table is flat.

**But it is harsh on boundary accuracy, and the trainer must be told so:** a
prediction whose regions are all in the right place but uniformly 4px thin
scores **0.382**, having lost 74.5% of its matches, and 1px thin already
costs 14.4%. `score_panel` emits pixel IoU beside region-F1 and **both must
be read**, because region-F1 alone will under-report a model that finds the
right regions with slightly tight boundaries. This is a property of the
metric at these thresholds, not a defect, and the sensitivity grid
`{0,100,500} x {0.25,0.5,0.75}` already in every row is where a looser
reading lives if one is ever wanted -- it is never used to pick a verdict.

### The near-blank stratum, reported as unsupported rather than as a number

Its 13 panels hold region counts `[0,0,1,1,0,0,0,0,0,3,3,0,0]`: nine have no
regions at all and the four that do cannot carry a retention fraction.
**No IC6 number is printed for it**, and none should be quoted.

### What this does not cover

**Over-painting is untested.** Erosion probes only a prediction that is too
small; a dilated prediction survives `IoU >= 0.5` while `d <= 0.414R`, and
given how badly the disc model failed for erosion, that boundary should be
measured and not assumed. Displacement is already probed by NC3 and NC5 in
the stage-2 table. Both belong to a later entry.

Visual: `results/fill_koma_20261010/ic6_degradation.png` (one row per level,
GT beside the degraded prediction). Per-panel:
`results/fill_koma_20261010/ic6_per_panel.csv`, 948 rows.

### Consequence for the GPU slot

IC6's pre-flight condition from the stage-2 entry is **met**: region-F1 can
grade a partial win, so the metric can grade a trained model. Together with
that entry's finding -- every non-learned arm falls to NC3 and `rough_dark`
is worse than painting nothing -- **the 2026-10-19..10-22 slot should be
claimed.** The number to beat remains `blank`'s **0.0084**, the ceiling is
IC3's **1.0**, and region-F1 and pixel IoU are read together for the reason
above.

## 2026-10-16 Pre-registration: IC-j9, the touch merge decided on the never-read holdout `[0:40]`

Written before anything is run on `[0:40]`. The foundation routed back a
reading correction (their `doc/CURRENT.md`, Known Tool Traps, 2026-10-16),
**as a correction and not as an adoption request**:

> A control the reference arm itself fails cannot discriminate. The leg was
> meant to ask "does this repair cost precision" and in fact asked "does it
> *improve* precision", which a zero-cost repair can never pass. **Write a
> cost leg as a difference from the reference arm**
> (`wrong_candidate - wrong_reference <= 0`), not as an absolute imported
> from another population.

I accept the prescribed form, which is stricter than the `+0.010` I had
proposed myself. **The IC-j8 verdict is not edited: it stands at NOT ADOPTED
under the bar that was registered then.** This is a new registration with a
new bar, and the adoption decision belongs to it.

### The procedural problem, and what is done about it

**The measurement already exists.** On `[40:171]` the candidate scored
rejoin 0.752 / wrong 0.061 / score 0.691 against the reference's 0.643 /
0.061 / 0.582, trial by trial 34 gained, 0 lost, 0 newly wrong. Registering a
delta bar now and then "reading" that number would be fitting the bar to the
result, whoever proposed the form.

So the bar is evaluated on **panels `[0:40]`, which have never been read in
this thread** -- reserved since the IC-j6 registration for exactly this. From
here on:

- `[40:171]` (311 trials) is the **discovery set** and is labelled as such
  wherever it appears. No bar is decided on it again.
- `[0:40]` is the **holdout**, used **once**. Whatever it says is the verdict;
  it is not re-read under a different bar, and if it refuses the repair the
  repair is refused.

### The bars, in the prescribed form, fixed now

`SEED = 20261010` unchanged, both arms on the same trials with trial identity
asserted, all 11 constants asserted identical between the two files.

1. **Cost leg (the foundation's form).**
   `wrong_candidate - wrong_reference <= 0`.
   At this n the delta is a bar on counts, so the discrete form is registered
   and is what is read: **`wrong_count_candidate <= wrong_count_reference`
   AND `newly_wrong = 0` by set, not by count.** Stating it discretely is
   stricter than the continuous form, not looser: `[40:171]` had ~2.37 usable
   trials per panel, so `[0:40]` is expected to give roughly 95 trials, where
   one trial is ~1.05 points of `wrong` and a continuous bar at 0 would be
   indistinguishable from a bar at +0.01.
2. **Recall leg.** `rejoin_candidate - rejoin_reference > 0`.
3. **Figure-of-merit leg.** `score_candidate - score_reference >= +0.05`
   (score = rejoin - wrong, IC-j6's own). The discovery set showed +0.109;
   half of it is the bar, the same halving rule used for IC-j8's rejoin bar.
4. **All three required.** Any one failing means NOT ADOPTED, and that is the
   headline of the entry, not a footnote.

### Pre-registered reasons to refuse to read it at all

- **Power.** If `[0:40]` yields **fewer than 60 usable trials**, the holdout
  is underpowered for a count-level cost leg and the entry says
  "UNDERPOWERED, not actionable" instead of a verdict. The repair then stays
  unadopted and waits for a larger reserved set.
- **Comparability.** If the reference arm's rejoin on `[0:40]` falls outside
  **0.643 +/- 0.10**, the two panel ranges are not the same kind of material
  and the holdout cannot arbitrate a 10-point effect measured on the other
  range. That is reported as a comparability failure, not as a verdict.
  (Both thresholds are written here, before the run, precisely so that a
  disappointing number cannot be reclassified as one of these afterwards.)

### What is NOT claimed even if all three legs pass

Adoption would mean `fragments()` keeps the full touch set and `merge_nodes`
unions every cluster a short fragment touches, in `stroke_chain.py`. It would
**not** mean:

- that the 46% `t_relay_other` bucket is addressed -- it is not, and the ridge
  sits 0.2 to 5.5px above `MERGE_PX`, whose sweep is a separate registration;
- that `MERGE_PX` or `C_UNMATCHED` are justified -- neither has ever been
  swept (972-combination grid, 6 axes, neither in it);
- anything about deliberate gaps, which this construction cannot contain
  (0 of 90 free-end cases) and which Track K now owns on real ink.

Run log: `results/stroke_chain_20261010/icj9_holdout.txt`. Tool:
`score_touch_merge.py`, extended with the delta-form bars and a
`--discovery` flag that prints the discovery-set row beside the holdout row,
both labelled.

## 2026-10-17 IC-j9 result: **ADOPT** -- the touch merge passes all three bars on the holdout `[0:40]`

Run log: `results/stroke_chain_20261010/icj9_holdout.txt` (finished 2026-10-16
23:03; read 2026-10-17 14:03, which is a delay of 15 hours on a number that
was already on disk -- see the process note at the end). Registration
`b29b4c4`, scorer `7b5273d`. The holdout is used **once**; this entry is that
one reading.

### Both refusal preconditions, checked before any leg was read

- **Power.** 97 usable trials from 40 panels (2.43 per panel, against the
  2.37 per panel the registration predicted from the discovery set, which is
  why ~95 was written down). `97 >= 60` -> not underpowered.
- **Comparability.** Reference rejoin on `[0:40]` is **0.619**, inside the
  registered `0.643 +/- 0.10` band `0.543..0.743` -> comparable.

Both were fixed in advance precisely so a disappointing number could not be
reclassified afterwards as one of them. Neither fired.

### The three arms, same 97 trials asserted (same fragments, same F parts, same occluder parts)

| arm | n | rejoin | wrong | score |
|---|---:|---:|---:|---:|
| `stroke_chain` (calibrated, sha256 `e9e49267c339f885`) | 97 | 0.619 | 0.072 | 0.546 |
| **+ touch merge** (sha256 `091539ea88cfe012`) | 97 | **0.711** | 0.072 | **0.639** |
| `force`, gates off (**IDENTITY**) | 97 | 0.711 | 0.196 | 0.515 |

The third row is the trivial solution and is labelled an identity: pairing
everything reaches the same rejoin and pays 0.196 of `wrong` for it, which is
exactly what the cost leg refuses. The candidate reaches that rejoin for
0.072, the same `wrong` as the reference.

**Trial by trial, which is what decides it and not the counts:**
rejoined base 60 -> candidate 69, **gained 9, LOST 0**;
wrong base 7, candidate 7, **newly wrong 0, no longer wrong 0**.
An identical count is not an identical set; the set is identical too. The
repair is strictly additive here, as it was on the discovery set.

### The three legs

| leg | registered bar | measured | verdict |
|---|---|---:|---|
| cost | `wrong_count_cand <= wrong_count_ref` AND `newly_wrong = 0` by set | 7 vs 7, newly wrong 0 | **PASS** |
| recall | `rejoin_cand - rejoin_ref > 0` | **+0.093** | **PASS** |
| merit | `score_cand - score_ref >= +0.05` | **+0.093** | **PASS** |

All three required, all three pass -> **ADOPT**.

### The trap this holdout demonstrates a second time

IC-j8's failed bar was the absolute `wrong <= 0.059`, imported from
`icj6_grid` at n=309. On this holdout **the reference arm scores 0.072**, so
that absolute bar would have failed the reference itself by 1.3 points -- the
foundation's "a control the reference arm itself fails cannot discriminate",
now measured on a second, independent trial set. The difference form
(`cand - ref <= 0`) reads 0.000 and discriminates. **IC-j8's NOT ADOPTED
verdict is still not edited**; it stands under the bar registered then, and
this adoption belongs to IC-j9's bar.

### Discovery set beside the holdout, both labelled

| set | n | ref rejoin | cand rejoin | delta | ref score | cand score | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[40:171]` discovery (no bar decided on it again) | 311 | 0.643 | 0.752 | +0.109 | 0.582 | 0.691 | +0.109 |
| `[0:40]` **holdout** (the verdict) | 97 | 0.619 | 0.711 | +0.093 | 0.546 | 0.639 | +0.093 |

The holdout effect is 1.6 points smaller than the discovery effect and in the
same direction. The verdict is the holdout's.

### What is NOT claimed, carried forward from the registration unchanged

- The **46%** `t_relay_other` bucket is **not** addressed. The touch merge
  repairs `t_touch_split` (41% of topology failures); the ridge cases sit
  0.2-5.5px above `MERGE_PX` and need a two-node cost, a design change under
  a separate registration.
- `MERGE_PX = 6.0` and `C_UNMATCHED = 1.0` remain **unjustified**: neither was
  in the 972-combination grid. The bounded sweep is still unwritten.
- Nothing about **deliberate gaps**. This construction cannot contain them
  (0 of 90 free-end cases, and a deliberately broken stroke is unjoinable at
  **0% by architecture**, `doc/shortcuts.md`). Track K owns that on real ink.
- `wrong` counts only contamination by the **artificial** occluder.
  Intentional joins are unmeasured; precision here is a proxy.

### Process note, recorded because it is the second instance of one habit

The run finished at 23:03 and was read at 14:03. The same shape as the
`per_panel_held.csv` failure (completed 10-11 01:01, read 10-16): **a result
already on disk is not a result until it is read.** The rule written then --
*a cheap gate that can release or claim a scheduled slot is read before any
expensive branch* -- is widened: **a finished run's log is read in the same
session it finishes in, before any new registration is drafted.**

### 2026-10-17 The adoption carried into the code, and what it moves

`stroke_chain.py` now carries the repair. sha256 **`e9e49267c339f885` ->
`53b8406cfbb4f4d1`**. The pre-adoption file is recoverable at commit
`1a82ea5` and a copy was kept while the swap was verified.

Verification, before anything was re-measured:

- **Assert (byte identity).** The adopted file's body, with the module
  docstring stripped by `ast`, is **byte-identical to the candidate that was
  measured** (22,783 bytes). The adoption is not a re-implementation of the
  candidate; it *is* the candidate, with the original docstring restored and
  an adoption paragraph appended.
- **Assert (constants).** All 10 swept/registered constants identical across
  the swap: `TAN_SKIP 2, TAN_WIN 20, KAPPA_REF 200.0, THETA_MAX 60.0,
  D_MAX 3.0, W_KAPPA 0.0, MERGE_PX 6.0, C_UNMATCHED 1.0, THETA0 30.0,
  MAX_ARMS 8`. **`SPUR_PX` was named and found absent in both** -- J has no
  spur constant because J does not prune. The cross-track comparability note
  that follows (Track F prunes with `spur_max = 6` and reads a cached
  skeleton, J calls `skeletonize(ink)` in place, so fragment, node and arm
  counts are **not comparable between tracks for the same image**) was
  recorded with Track K and in `../lineart/inbox/`, and **not** in this log.
  Correcting the claim rather than leaving it: it was not "already recorded"
  here, and writing it here is one of the owed entries listed below.
- **Installation check, on the discovery set, not the holdout.** With the
  repair installed the two arms of `score_touch_merge.py` are the same code
  path, and on panels `[40:60]` (36 trials) they read **gained 0, LOST 0,
  newly wrong 0** -- which is what an identical code path must read.

That last reading is also a hole, so it was closed rather than noted: a
self-comparison reads exactly like a repair that does nothing. **A
`PRE_ADOPT_SHA` guard was added to `score_touch_merge.py`: if the reference
arm is not `e9e49267c339f885`, the tool prints `NO VERDICT` and returns
before any bar is read**, naming the commit to check out to reproduce IC-j8
or IC-j9. Verified by running it with `--holdout` and watching it refuse.

**Figures of record that this moves.** IC-j6's `rejoin` is now the
post-adoption number wherever the current tool is run:

| set | pre-adoption (figure of record until 2026-10-17) | post-adoption |
|---|---:|---:|
| `[40:171]` discovery | 0.643 | **0.752** |
| `[0:40]` holdout | 0.619 | **0.711** |

`wrong` is unchanged on both sets (0.061 / 0.072). Every entry before today
citing 0.643 or sha `e9e49267c339f885` is a pre-adoption reading and stays as
written.

**What the remaining failure profile becomes -- ARITHMETIC, not a new
measurement.** On the discovery set: 111 failures, 34 newly rejoined, so 77
remain. All 34 gains land in `t_touch_split` (34 of its 37 is the 92% flip
rate IC-j8 measured, and C5's anchor found 0 `merge_touch` flips outside that
bucket). Then `t_relay_other`'s 51 cases are **66.2% of what is left**, up
from 46.0% of all failures. This is division on measured parts and is
labelled as such; the post-adoption buckets have not been re-measured, and
the next registration that needs them must measure them.

**`t_relay_other` is now the whole remaining problem and it is unaddressed.**
It needs a two-node cost (the ridge loses on a node-local cost while the
evidence is chain-local), which is a design change under its own
registration, and the bounded `MERGE_PX` / `C_UNMATCHED` sweep -- the ridge
sits 0.2-5.5px above `MERGE_PX = 6.0` -- is still unwritten.

## 2026-10-17 Owed entries: the cross-track instrument agreement, and two comparability notes

These were agreed with Track K on 2026-10-16 and recorded there and in
`../lineart/inbox/`, but not here. Written now, with the numbers recomputed
from the CSV rather than carried in prose.

### (c) IC-k1: one instrument, two tracks, 0.99 -- a lesson-12 anchor

Track K measured koma-wide GT solid-fill area at `r = 4` independently, on
their own code path, and reports a median of **0.0646**. Recomputed from
`results/fill_koma_20261010/per_panel_held.csv` on my side (969 panels, one
row per panel, `gt_area_frac`):

| stratum | n | area_frac mean | **median** | q25 | q75 | ink-normalised `fill_ratio` mean / median |
|---|---:|---:|---:|---:|---:|---:|
| all panels | 969 | 0.1027 | **0.0655** | 0.0238 | 0.1455 | 0.5143 / 0.5622 |
| normal | 956 | 0.1040 | **0.0665** | 0.0246 | 0.1476 | 0.5193 / 0.5666 |
| near-blank | 13 | reported separately, no verdict read | | | | |

**0.0646 / 0.0655 = 0.986.** Two tracks, two code paths, the same definition
of the target: agreement to 1.4%. This is the leg lesson 12 asks for -- one
anchor to something this pipeline did not produce -- and it is the first time
a Track J number has been reproduced from outside Track J.

**What it does NOT establish, stated because it is the easy misreading:**
it confirms that *the same constant gives the same value*. It says nothing
about whether **`FILL_HALF_WIDTH_PX = 4.0` is the right half-width.** Both
tracks used 4.0. The sensitivity analysis of that constant is registered
separately below and is not a re-tune.

**The skew caveat, which the agreement itself nearly hid.** My stage-2
headline is the **mean 0.1040**; the quantity K matched is the **median
0.0665**. The distribution is strongly right-skewed (mean / median = 1.56,
q25 0.0246 vs q75 0.1476, a 6x spread across the middle half). **Medians are
compared to medians and means to means**, and any cross-track figure must say
which it is. A mean-to-median comparison here would have read as a 61%
disagreement and would have been wrong.

### (c2) J does not prune; Track F does. Fragment, node and arm counts are not comparable between tracks

- **Track J**: `stroke_chain.fragments()` calls `skeletonize(ink)` in place
  and **keeps short pieces**. There is no spur constant -- `SPUR_PX` was
  named and found absent when the chainer was swapped (entry above). The only
  absorption is `merge_nodes` at `MERGE_PX = 6.0`, and the absorbed length is
  returned and reported as `absorbed_px` rather than vanishing.
- **Track F**: prunes with `spur_max = 6` and reads a **cached** skeleton
  (`panel_skeleton_20260917`).

So for **the same image** the two tracks do not agree on how many fragments,
nodes or arms there are. Any table that places a J fragment count beside an F
or K one is comparing two different decompositions. Track K recorded this and
will not place K2's bucket counts beside J's.

**The second-order consequence, which Track K got right and I got wrong.**
I warned them that `GAP_MAX = 24px` would admit my 51 ridge cases (facing
tips 11.2-19.9px apart *with ink between*). That was wrong: those 51 have
nodes at both ends, so they never enter their free-end box at all. **Their
mechanism is the real one**: `spur_max = 6` pruning removes free ends from
the *skeleton* while **the ink stays in the image**, so pruned sites produce
free-end pairs facing each other across ink. They added the condition "no ink
on the segment between the tips" (sampled 1px along, collar
`max(2.0, local ink width / 2)` at each end) and will **print the
distribution of ink-pixel counts** with a stop rule if the 0 and non-0 peaks
do not separate.

### (c3) K2's numbers cannot be compared to IC-j6's rejoin -- the mirror of K's own note

Track K registered "K2 の数は IC-j6 の rejoin と比較できない". The same holds
in this direction and is recorded here so neither side has to remember it:

- IC-j6's `rejoin` is measured on **artificial damage**: a straight occluder
  is *added* across a real fragment, so the ground truth is known by
  construction. K2 works on **real ink** with no known ground truth per site.
- IC-j6 therefore cannot contain the defect K2 is built for. 0 of 90
  topology failures were free-end cases, and **a deliberately broken stroke
  is unjoinable at 0% by architecture** (`build_links` only collects arms
  with node id >= 0), not "sometimes missed". This is the foundation's fifth
  control trap -- an instrument built from artificial damage cannot contain
  damage the material never shows.
- The decompositions differ as well, per (c2).

Division of labour as agreed, unchanged: **deliberate gaps on real ink ->
Track K**; **gap closing (item 2) -> Track J**; **chainer and the bounded
`MERGE_PX` / `C_UNMATCHED` sweep -> Track J**. Track K changed K4's variable
from J's `MERGE_PX = 6.0` to Track F's bridge-collapse threshold
`max(3.0, 1.5 x median(local ink width))`, because Track F produced the lines
the author actually marked.

## 2026-10-17 Pre-registration: IC-j10, the sensitivity of the conclusions to `FILL_HALF_WIDTH_PX`

Written before the tool exists. Approved order (user, 2026-10-17): M1 takes
the 10-19..10-22 slot, this and the bounded `MERGE_PX` sweep run on CPU
beside it. **One change to that order, with its reason:** this one runs
**first, today**, because `FILL_HALF_WIDTH_PX` defines the target M1 would be
trained on, and knowing on Monday how much the stage-2 conclusions depend on
it costs a Saturday of idle CPU. Running it after M1 would mean training
against a target whose sensitivity is unknown.

### Why

`FILL_HALF_WIDTH_PX = 4.0` is one of three constants that were written down
and never swept (`MERGE_PX`, `C_UNMATCHED`, this one), and it is the worst of
the three: **it defines what is being predicted.** `fill_mask` is a
morphological opening by a disc of this radius, so `r = 4` means "ink wider
than 8px is solid fill". Nothing has ever tested that.

Track K's independent 0.986 agreement (entry above) does **not** help here.
Both tracks used 4.0; the agreement shows the same constant gives the same
value.

### What this is, and what it is explicitly not

**It is a sensitivity analysis of the conclusions already drawn.** It asks
whether the stage-2 verdicts survive a change of `r`.

**It is not a re-tune and cannot select `r`.** There is no quantity to
maximise over `r` -- a larger `r` makes the target smaller and every arm's
job different, so "the `r` with the best score" is a meaningless object. If
the conclusions turn out to depend on `r`, that dependence is the finding,
and `r` then has to be decided by something outside this analysis (what the
author's own fills actually are -- a labelled judgement), never by picking
the `r` that reads best. Written here so it cannot be done later.

### Grid and sample

`r` in **{0, 2, 3, 4, 6, 8}**. `r = 0` is the instrument check, not a
sensitivity point. Sample: the **956 normal** held panels; the **13
near-blank** panels are reported separately with no verdict read (IC6 already
established that stratum cannot support a region metric -- nine have zero
regions). If a radius costs more than 40 minutes the sample is subsampled
**mechanically, every k-th panel by sorted name, with k and n printed**, and
k is chosen from the timing probe before any score is read.

### Instrument check, with a known value

**IC.** At `r = 0` the opening is the identity, so `fill_mask(ink, 0)` must
equal `ink` **exactly**, pixel for pixel, on every panel. Expected value:
`0` differing pixels, `area_frac(r=0) == ink_ratio` to 1e-12. Asserted, not
eyeballed. If this fails the tool is wrong and nothing else is read.

### C1 nesting, on two implementations -- the registered expectation differs between them

Opening by a nested family of discs gives nested results, so for every panel

    fill(r=8) subset fill(r=6) subset fill(r=4) subset fill(r=3) subset fill(r=2)

- **Exact EDT (`scipy`): 0 violating pixels expected**, asserted.
- **cv2 maskSize-3 (what every published Track J fill number used):
  violations are expected to be non-zero** and are *printed, not asserted*.
  E1/E2 of IC6 already measured that this approximation shifts an effective
  radius by about 0.25px, so nesting can break in the rim. **The cv2
  violation count is the number this leg exists to produce**: it is the price
  of the approximation, on the mask that defines the target.

Both are reported. If cv2's violating pixels exceed **1%** of `fill(r=2)`
area on the median panel, the published fill numbers depend on the
approximation more than on the constant, and that is the headline instead.

### The decisive prediction, with its numbers, before the run

Stage 2 M0 (2026-10-16) concluded, at `r = 4`: every non-learned arm falls to
NC3, and `rough_dark` beats `blank` by **-0.0019** -- worse than painting
nothing. The mechanism recorded then: the rough's own dark ink is **0.69% of
panel area** while GT fill is **10.40%** (mean, normal stratum), a 15x
deficit.

**Prediction: the verdict does not move at any `r` in {2,3,4,6,8}.** A
smaller `r` enlarges both the GT and `rough_dark`'s prediction, so the
deficit is roughly scale-free; no radius can close 15x.

Fixed now, and these are the words the entry will use with their thresholds:

- **"robust to r"** := at every `r` in {2,3,4,6,8}, both
  `region_F1(rough_dark) - region_F1(blank) < 0.05`
  and `region_F1(rough_dark) - region_F1(nc3_hflip:rough_dark) < 0.05`.
- **"r-dependent"** := at some `r`, either gap is **>= 0.05**. Then the
  stage-2 verdict was an artefact of `r = 4`, it is **withdrawn in the
  headline**, and M1's target definition is reopened before the slot is used.

### Second pre-registered expected value, from an independent earlier sample

An `r`-census on 60 held panels (recorded 2026-10-14) measured that **0.628
of the `r = 4` core area survives at `r = 8`** and 0.429 at `r = 12`.
Repeating the `r = 8` leg on 956 panels: **predicted 0.63, band +/- 0.10.**
Outside that band, the 60-panel census did not generalise and every number
derived from it is reread.

### What is reported, in the required form

Per radius, per stratum: `gt_area_frac` **mean and median** (the skew caveat
of entry (c): mean/median = 1.56, so both are printed and labelled),
`gt_fill_ratio` (ink-normalised) beside it as C5 requires, region count at
`MIN_REGION_PX = 500`, and the three arms `blank` (IDENTITY),
`rough_dark`, `nc3_hflip:rough_dark`.

**Only those three arms.** The full 14-arm table costs ~2.7h per radius (a
timing probe measured >10s per panel), and the verdict at stake is carried by
`rough_dark` against `blank` and NC3. The other arms are named as not
re-measured: `all_black` (an identity), `rough_dark@budget`, `freq_table`
(which at `r = 4` predicted **nothing** -- predArea 0.0000, i.e. it *is*
`blank`, an identity), NC1, NC2 (void, 0 rows of 13,566), NC5.

Tool: `tools/evaluation/sens_fill_radius.py`, new, with the IC and C1 asserts
in it. Log: `results/fill_koma_20261010/icj10_radius.txt`, per-panel CSV
beside it. Montage of the same panel at every `r` to
`results/fill_koma_20261010/icj10_radius.png` -- numbers cite a visual
artefact under `results/`, never `/tmp`.

## 2026-10-17 Pre-registration: IC-j11, the bounded `MERGE_PX` / `C_UNMATCHED` sweep

Written before the tool exists. The second CPU item of the approved order.

### Why, and why "bounded" is in the name

`MERGE_PX = 6.0` and `C_UNMATCHED = 1.0` were written down by eyeball and
**were never in the 972-combination grid** (6 axes: `TAN_SKIP`, `TAN_WIN`,
`KAPPA_REF`, `THETA_MAX`, `D_MAX`, `W_KAPPA`). IC-j8 then measured that
`MERGE_PX` is the rate limiter on the remaining failures: the relay ridge
sits at **6.2 / 7.7 / 8.1 / 9.2 / 11.5 px** (min, q25, median, q75, max over
51 cases), i.e. **0.2 to 5.5px above the threshold**, and all 51 are a single
fragment within 20px of the cut.

**Unbounded loosening is already refused by measurement.** NC-j8c:
`merge_px_inf` flips 45% of `t_relay_other` and **breaks 111 of 200
already-rejoining trials (55.5%)**. So the sweep exists to find whether there
is a *bounded* value that gains without breaking, and the quantity that
decides it is **`broken`**, not `rejoin`.

### Grid

- `MERGE_PX` in **{6, 8, 10, 12, 16}** (6 is the current value; 16 is past the
  ridge maximum of 11.5)
- `C_UNMATCHED` in **{0.5, 1.0, 2.0}** (1.0 is current; `THETA0 = 30` costs
  1.0, so at 0.5 the angle gate alone can exceed the unmatched price and at
  2.0 it cannot -- the two sides of the break-even that made `THETA_MAX = 60`
  exactly redundant)
- touch merge **on / off** (on is now the adopted chainer; off is the
  pre-adoption path, still reachable as `merge_nodes(..., touch=None)`)

30 cells. Trials are built **once** -- the cut, its ground truth and the
skeleton of the cut image do not depend on any swept constant, which is the
same property IC-j6's grid relied on.

### Which panels, and the one thing a sweep must not do

- **Exploration: `[40:171]`**, the labelled discovery set, 311 trials.
  **No cell is adopted from this set.** A sweep is a tuning operation and
  tuning on the set that produced the hypothesis is how the 0.2582 happened.
- **Reserved verification: `[171:271]`, 100 panels, never read by anything.**
  Registered here, unread. At 2.37 usable trials per panel it should give
  **~237 trials**. It is read **once**, only if a cell is proposable, and
  whatever it says is the verdict.
- Panels `[0:40]` (the IC-j9 holdout) and `[100:171]` (the original
  calibration set, which lies *inside* the discovery range -- recorded here
  because it is a real overlap in the existing numbers) are **not** reused.

### Instrument check, with known values

The two cells that are the current chainer must reproduce numbers measured by
different code on different days, on `[40:171]`:

- `MERGE_PX = 6, C_UNMATCHED = 1.0, touch ON` -> **rejoin 0.752, wrong 0.061**
- `MERGE_PX = 6, C_UNMATCHED = 1.0, touch OFF` -> **rejoin 0.643, wrong 0.061**

Asserted to +/- 0.005 (the trial set is identical, so exact equality is
expected and the tolerance exists only for float formatting). **If either
fails the sweep is not read at all** -- it would mean the grid is not
scoring the chainer that is installed.

### Expected shapes, written down first so a surprise is visible

1. `rejoin` **rises** with `MERGE_PX` and **flattens by 12**, because the
   ridge maximum is 11.5px.
2. `wrong` **rises monotonically** with `MERGE_PX`.
3. `absorbed_px` (the total ink absorbed into nodes, which the chainer
   already returns rather than letting it vanish) **rises monotonically**.
   Not asserted: absorption depends on node ids that cascade as merges
   happen, so a non-monotone cell is possible and **violations are printed
   and counted** rather than assumed away.
4. At `MERGE_PX = 12`: `rejoin >= baseline + 0.05`. The 51 ridge cases are
   16.4% of the 311 trials, so a threshold past the ridge should recover a
   large part of them; **failing this floor means the ridge is not actually
   blocked by `MERGE_PX`** and the two-node-cost design becomes the only
   route.
5. `broken > 0` at some `MERGE_PX` in the grid, because `merge_px_inf` broke
   55.5%. If **no** cell breaks anything, the grid is too narrow and that is
   said rather than read as a free win.

### Bars, fixed now

**`broken` is the decisive quantity.** For a cell to be **proposable**
(not adopted -- proposable):

- `broken_count = 0` **by set**: no trial that rejoins under the installed
  chainer stops rejoining;
- `newly_wrong = 0` **by set**;
- `score - score_baseline >= +0.05` on `[40:171]`.

**Adoption requires the reserved `[171:271]` read once**, under the same
three legs in difference form (the foundation's prescription), with the same
two refusal preconditions as IC-j9: **< 60 usable trials -> UNDERPOWERED**;
**reference rejoin outside its `[40:171]` value +/- 0.10 -> NOT COMPARABLE**.

If **no cell is proposable**, that is the result, it is the headline, and
`MERGE_PX = 6` stands as the value that survived a sweep rather than the
value nobody tested.

### What is not claimed either way

Nothing about deliberate gaps (Track K owns them on real ink; this
construction cannot contain them). Nothing about intentional joins -- `wrong`
counts only contamination by the artificial occluder. And a proposable cell
would not address the relay ridge *as a design problem*: the ridge loses on
a node-local cost while the evidence is chain-local, and `match_node` is
already optimal within a node.

Tool: `tools/evaluation/sweep_merge_px.py`, new. Log:
`results/stroke_chain_20261010/icj11_sweep.txt`, CSV beside it, montage of
the cells that change a trial's outcome to
`results/stroke_chain_20261010/icj11_cells.png`.

### 2026-10-17 Two amendments, both made before any result of their own run was read

**F2 (IC-j10, sampling).** The registration said: if a radius costs more than
40 minutes, subsample mechanically every k-th panel by sorted name. Measured
cost on the full run: **15.6 s/panel, 969 panels, ~4.2 h, i.e. ~50 min per
radius -- over the registered 40.** No subsample was taken and the run is
reading all 969 panels. Reason, recorded rather than left implicit: that rule
was written to stop a CPU job eating the GPU slot, and this one lands on an
idle Saturday with nothing waiting on it. **Using more panels than registered
is a strengthening, not a loosening**, and the decisive D1 bars are untouched.
If it had been the other direction -- fewer panels than registered -- it would
have needed k and n printed, which the tool does print.

**G1 (IC-j11, a blind spot in the registered expectation).** A plumbing test
on an **unregistered** 4-panel fixture `[300:304]` (not any registered set; no
number read from it) showed the touch switch moving `absorbed_frags`
(12->13, 3->4, 7->8) and the count of distinct merged nodes (5->4, 9->8)
while `absorbed_px` stayed **identical to one decimal**. Cause, found in the
code: the fragments the adoption additionally absorbs are **single-pixel
fragments, whose `path_len` is 0.0 by construction** (`path_len` returns 0.0
for a one-point path). Registered expected shape 3 was written on
`absorbed_px` alone and is blind to them, so **`absorbed_frags` is swept and
printed beside it**. No bar changes.

G1 is worth more than its size: `absorbed_px` was introduced so that absorbed
ink would be *reported rather than vanish*, and it cannot see a zero-length
absorption at all. The same blindness applies to any earlier reading of
`absorbed_px`.

## 2026-10-17 IC-j11 result: **no cell is proposable** -- `MERGE_PX = 6` survives a sweep, and the bar that rejected the rest is the one worth looking at

Log: `results/stroke_chain_20261010/icj11_sweep.txt`, every cell in
`icj11_sweep.csv`. Registration `c9074ad`, tool `ccdda59`. Discovery set
`[40:171]`, **311 trials**, 30 cells. The reserved set `[171:271]` is
**still unread.**

### IC, both legs

| cell | rejoin | expected | wrong | expected | |
|---|---:|---:|---:|---:|---|
| `MERGE_PX 6, C_UNMATCHED 1.0`, touch **on** | 0.752 | 0.752 | 0.061 | 0.061 | PASS |
| the same, touch **off** | 0.643 | 0.643 | 0.061 | 0.061 | PASS |

Exact to three decimals against values measured by different code on
different days. The grid is scoring the chainer that is installed.

### The registered verdict

**0 proposable cells** (bar: `broken = 0` by set AND `newly_wrong = 0` by set
AND `score - 0.691 >= +0.05`). Under the bar registered before the run,
`MERGE_PX = 6.0` **stands as the value that survived a sweep** rather than
the value nobody tested. That is the headline.

### Which bar did the rejecting -- and the reading that follows

Two cells fail on **one clause only**, and it is not `broken` and not
`score`:

| cell (touch on) | rejoin | wrong | score | vs baseline | broken | newly_wrong |
|---|---:|---:|---:|---:|---:|---:|
| baseline `MERGE_PX 6, C_UNMATCHED 1.0` | 0.752 | 0.061 | 0.691 | -- | 0 | 0 |
| `MERGE_PX 8, C_UNMATCHED 1.0` | 0.830 | **0.048** | 0.781 | **+0.090** | **0** | **1** |
| `MERGE_PX 10, C_UNMATCHED 1.0` | 0.868 | **0.035** | 0.833 | **+0.142** | **0** | **3** |

Both **lower the `wrong` rate** (0.061 -> 0.048 -> 0.035, i.e. roughly 19
contaminated trials -> 15 -> 11) while creating 1 and 3 new ones. They break
**nothing**. They fail because I registered `newly_wrong = 0` **by set**.

That bar is not the foundation's trap -- the reference passes it trivially,
by definition -- but it is a **different one-sided reading error, and it is
mine**: *a set-level zero on creations cannot see a net improvement.* It
refuses any **trade**, and only an exactly additive repair can clear it.
IC-j9's repair happened to be exactly additive (gained 9, lost 0, newly wrong
0), so the conjunction cost nothing there and I carried it over without
noticing that it is a different kind of bar.

**The verdict is not changed by noticing this.** The bar was registered, it
rejected these cells, and the entry says so. What follows is a *new*
registration, below, decided on the **reserved** set.

### Expected shapes: one confirmed, one refuted, one confirmed at scale

1. **Confirmed.** `rejoin` rises with `MERGE_PX` and flattens by 12: 0.752,
   0.830, 0.868, 0.878, 0.868 (touch on, `C_UNMATCHED 1.0`) -- the peak is at
   12 and 16 is slightly worse, which is what a ridge maximum of 11.5px
   predicts.
4. **Met.** `rejoin` at `MERGE_PX = 12` is **+0.125** against the registered
   floor of +0.05. **So the ridge IS reachable by the threshold** -- the
   relay failures are not structurally out of reach, they are priced out.
5. **Confirmed.** 960 broken trials over the grid, so the grid was wide
   enough to contain the damage; this is not a free win anywhere.
2. **REFUTED, and this is the surprise.** `wrong` was registered to rise
   monotonically with `MERGE_PX`. It **falls**: 0.061, 0.048, 0.035, 0.035,
   0.035. `wrong monotone False` in all six rows, in the opposite direction
   to the one registered.
   **Mechanism, labelled a hypothesis and not measured here:** absorbing more
   fragments into nodes removes *the occluder's own middle ridge* as a rival
   arm, and `wrong` is defined as chaining an F part to a piece of the
   occluder. Loosening the threshold eats the thing that `wrong` counts. If
   that is right, the falling `wrong` is partly an artefact of the artificial
   construction rather than a real precision gain -- which is a reason to
   read the reserved set, not a reason to assume either way.
3. **`absorbed_px` monotone everywhere; and G1 confirmed at scale.** At
   `MERGE_PX = 6` the touch switch moves absorbed fragments **1662 -> 2373
   (+711)** while `absorbed_px` stays at **4578.6 in both arms, identical to
   one decimal**. 711 fragments absorbed, 0 pixels of ink: they are all
   single-pixel fragments with `path_len = 0.0`. Any earlier reading of
   `absorbed_px` is blind to them.

### Two readings that fall out for free

- **`C_UNMATCHED = 0.5` is refuted as a direction.** It breaks 42-46 trials
  (touch on) and 68-71 (touch off) while driving `wrong` to 0.006: at half
  the break-even price the chainer stops pairing at all. The 30 degree angle
  gate costing 1.0 against a break-even of 2.0 was already recorded; this
  measures what happens on the other side of it.
- **An internal identity.** Touch **off** at `MERGE_PX = 6, C_UNMATCHED 1.0`
  reads `broken 34` against the installed baseline -- and IC-j9 measured the
  adoption gaining exactly **34** trials on this set. The same 34, counted
  from the other end, by different code.

## 2026-10-17 Pre-registration: IC-j12, the two cells IC-j11's one-sided clause rejected, read once on the reserved set

Written before the reserved set `[171:271]` is touched. **IC-j11's verdict is
not edited**: it stands at "no cell proposable" under the bar registered
then. This is a new registration with a new bar, the same separation IC-j9
kept from IC-j8.

### The conflict of interest, stated before the bar

I am about to drop the clause that rejected these cells, after seeing that it
rejected them. The difference form I am replacing it with
(`wrong_count_cand <= wrong_count_ref`) is the foundation's prescription and
**predates this result** -- it was routed back on 2026-10-16 and is already
in use in IC-j9 -- so the *form* is not chosen to fit the data. But the
*choice to apply it here* is made after seeing the data, and nothing about
the ordering hides that. The protection is the same one IC-j9 used and it is
the only one that works: **the bar is read on a set that has never been
read, once, and whatever it says is the verdict.**

### The cell, selected mechanically

Rule, applied to the discovery table without judgement: **the highest
`score` among cells with `broken = 0`.**

- `MERGE_PX 10, C_UNMATCHED 1.0`, touch on -- score 0.833, broken 0. **This
  is the cell.**
- `MERGE_PX 12, C_UNMATCHED 1.0` scores higher (0.842) and is **excluded by
  the rule**, because it breaks 3 trials. Recorded so the exclusion is
  visible rather than silent.
- `MERGE_PX 8` (score 0.781, broken 0) is the runner-up and is **not** read;
  one cell, one read of the reserved set.

### Set and refusals, fixed now

`[171:271]`, 100 panels, never read by anything. Expected ~237 trials.

- **< 60 usable trials -> UNDERPOWERED**, not a verdict.
- **reference rejoin outside 0.752 +/- 0.10 -> NOT COMPARABLE**, not a
  verdict.

### The diagnostic that is read BEFORE the legs, and can refuse them

IC-j11 refuted the registered expectation that `wrong` rises with
`MERGE_PX`: it falls (0.061 -> 0.035). The hypothesis recorded there is that
absorbing more fragments **eats the occluder's own ridge**, and `wrong` is
*defined* as chaining an F part to a piece of the occluder -- so the fall may
be a loss of **opportunity**, not a gain in precision. That has to be
measured before the cost leg means anything.

**Diagnostic:** the share of trials in which at least one occluder part is
still **unabsorbed**, for the reference and for the cell.

- **If `survive_cell / survive_reference < 0.80`, `wrong` is NOT COMPARABLE
  between the arms, the cost leg cannot be read, and the cell stays
  unadopted.** That is a refusal, not a verdict, and it is printed in those
  words.
- 0.80 is not a free parameter: the measured fall in `wrong` is 43%
  (0.061 -> 0.035), so a 20% loss of opportunity would already explain a
  large part of it.
- `wrong` is printed **raw and opportunity-normalised together**
  (`wrong / survival`), the same rule the fill numbers follow for area rate
  and ink-normalised value.

### The three legs

1. **Cost (difference form).** `wrong_count_cand <= wrong_count_ref` **AND
   `broken = 0` by set.** The IC-j11 clause `newly_wrong = 0 by set` is
   **dropped**, and this is a weakening. Reason: a set-level zero on
   creations refuses any trade and can only be cleared by an exactly
   additive repair, while `broken = 0` is the clause that protects what the
   chainer already gets right. `newly_wrong` is still **printed**.
2. **Recall.** `rejoin_cand - rejoin_ref > 0`.
3. **Merit.** `score_cand - score_ref >= +0.05`.

All three required. Any failure is the headline.

### What adoption would and would not mean

It would set `MERGE_PX = 10.0` in `stroke_chain.py`. It would **not** mean the
relay ridge is solved as a design problem -- `match_node` is already optimal
within a node and the ridge loses on a node-local cost while the evidence is
chain-local. It would not touch `C_UNMATCHED = 1.0`, which this sweep leaves
at its registered value (0.5 refuted as a direction, 2.0 never clears a cost
leg). And it says nothing about deliberate gaps.

Log: `results/stroke_chain_20261010/icj12_verify.txt`.

### 2026-10-17 Amendment H1: NC2's first repair is refuted by measurement before it was ever read as a control

The repair registered on 2026-10-16 for NC2's void run was "pair with the
nearest same-shape panel by sorted name, or crop to a common overlap, and
print sample counts". The first option was implemented and then **measured
before being used**:

| held panels | distinct shapes | panels alone in their shape | panels in a pairable group |
|---:|---:|---:|---:|
| 969 | **920** | **894** | 75 (**7.7%**) |

Group sizes: 894 shapes of size 1, 3 of size 2, 23 of size 3. Panel area runs
from 418,383 to 31,191,649 px, median 3,215,382 -- these are cropped manga
panels, so a shared pixel shape is close to an accident.

**The same-shape repair would have produced 49 NC2 rows per arm instead of
0**: void -> useless. It is not used, and the measurement is recorded here so
the next person does not try it again.

**So the registration's second option is taken: crop to the common overlap.**
Consecutive panels in sorted name order are paired (the original intent -- a
neighbouring panel of the same work is the most similar wrong answer
available) and both are **centre**-cropped to `(min h, min w)`. Centre and
not top-left, because a top-left crop would systematically keep one corner of
every panel and manga panels carry content throughout.

The objection that made cropping the second choice -- an arm's NC2 row would
no longer be comparable to the arm's own row -- is answered by **scoring the
arm on the same crop too**, emitted as `arm@nc2crop`. NC2 is read against
that row and never against the full-panel row.

Guard fixed before the run: **if the overlap is under 25% of the smaller
panel's area the pair is skipped and counted**, because a tall-by-wide pair
can overlap in a sliver that is not a panel. The overlap distribution and the
skip count are printed.

**Verified on the first 40 held panels, against the 2026-10-11 CSV: 560
non-NC2 rows compared, 0 differ.** The pairing change moves NC2 and nothing
else.

This matters beyond NC2's own row: **C1 requires the best arm to beat its
NC2, and C4 excludes any arm that fails NC2.** With NC2 void, neither could
be read, so M1's verdict was blocked on it. It is unblocked now.

## 2026-10-18 IC-j12 result: **NOT ADOPTED** -- the weakening bought nothing, and the quantity I selected the cell on is the one that failed to transfer

Reserved set `[171:271]`, **used once**, read against the bar registered in
commit `4cbc6de` before the set was touched. Log:
`results/stroke_chain_20261010/icj12_verify.txt`.

100 panels -> 1186 candidate sites -> **379 usable trials** (registration
expected ~237; more, not fewer).

### Preconditions, read first

- **379 >= 60** usable trials -> not underpowered.
- reference rejoin **0.710** against the discovery set's 0.752, inside
  `+/- 0.10` -> **comparable**. Both legs may be read.

### The diagnostic that could have refused the legs did not fire -- and it refutes IC-j11's mechanism hypothesis

| arm | trials with an occluder part still unabsorbed | ratio to reference |
|---|---:|---:|
| reference (`MERGE_PX 6`, touch on) | 379 / 379 | -- |
| cell (`MERGE_PX 10`, touch on) | 379 / 379 | **1.000** |

Refusal bar was `< 0.80`. The ratio is **1.000**: at `MERGE_PX = 10` the
occluder's ridge is still present as a chainable arm in **every single
trial**, so `wrong` is measuring the same opportunity in both arms.

`wrong`, raw and opportunity-normalised together, as registered:
**reference 0.061 / 0.061, cell 0.042 / 0.042** -- identical, because
survival is 1.000 in both arms.

**This refutes the mechanism hypothesis IC-j11 recorded** (labelled there as
unmeasured): the fall in `wrong` as `MERGE_PX` loosens is **not** a loss of
opportunity. The occluder ridge is not eaten. The fall is a real precision
gain, and the IC-j11 entry's "shape 2 refuted, mechanism unknown" now has a
mechanism it is **not**: absorption.

### The three legs

| leg | bar | measured | verdict |
|---|---|---|---|
| cost | `wrong_count_cand <= wrong_count_ref` **AND `broken = 0` by set** | wrong **16 vs 23**, **broken 2** | **FAIL** |
| recall | `rejoin_cand - rejoin_ref > 0` | **+0.113** (0.710 -> 0.823) | PASS |
| merit | `score_cand - score_ref >= +0.05` | **+0.132** (0.649 -> 0.781) | PASS |

All three required. **NOT ADOPTED.** `newly_wrong` printed as registered: 4.
`gained` 45.

The reserved set is used once and is **not re-read under another bar.**

### The two readings that matter, and the second one is about me

**1. `broken` did not transfer.** This cell was selected by one mechanical
rule -- *the highest `score` among discovery cells with `broken = 0`* -- and
on the discovery set `[40:171]` it broke **0** of 311 trials. On the reserved
set it breaks **2** of 379. The quantity the selection rule was built on is
the quantity that failed to replicate. That is the substantive result, and it
is the argument for a reserved set stated as a measurement rather than as a
principle: the cost half of a 30-cell table is the half a discovery set
cannot be trusted on, because selecting the minimum of a noisy count is
exactly how a zero is manufactured.

**2. The weakening bought nothing, and that is the cleanest answer available
to my own conflict of interest.** I dropped `newly_wrong = 0 by set` after
seeing it reject two cells, recorded the conflict, and relied on the reserved
set as the only protection. The cell then failed on the clause I **kept**.
So the weakening did not reach into the verdict: had `newly_wrong = 0` still
been in the bar, the verdict would be the same four-letter word. The
protection held.

### The clause worth naming, with no third registration attached to it

`broken = 0 by set` is zero-tolerance on a regression count, and it has now
rejected a cell that is better on **every aggregate** (`rejoin` +0.113,
`wrong` -7 trials, `score` +0.132) at a cost of 2 trials in 379. That is the
same family as the one-sided bar IC-j11 found. **Twice now a zero-tolerance
clause of mine has bitten after the fact.**

What I am **not** doing is registering IC-j13 to re-read this cell under a
tolerance. Two things stop it: the reserved set is used, and the pattern
itself -- a bar weakened once per failure is not a bar. Recorded instead as a
**standing design question, to be fixed before any new set exists**: should a
cost leg on a count admit a bounded regression (e.g. `broken <= 1% of
trials`), and if so the number goes into the registration **before** the
material does. `doc/shortcuts.md` carries it.

### What stands

**`MERGE_PX = 6.0` has now survived a bounded sweep (IC-j11, 30 cells, 0
proposable) and a reserved-set verification of its best challenger (IC-j12,
cost leg failed).** It is the installed value and no evidence yet displaces
it. `C_UNMATCHED = 1.0` likewise: 0.5 was refuted as a direction and 2.0
never cleared a cost leg.

What is **not** settled is the relay ridge itself. IC-j11 measured that
`MERGE_PX = 12` reaches it (+0.125 rejoin), and IC-j12 now measures that
loosening does not cost opportunity -- the ridge is **priced out at a
node-local cost, not out of reach**. That is an argument for the relay repair
as a two-node cost, which is registered-but-unwritten and remains so.

### Comparability note, recorded because the word "reserved" is doing work here

`calibrate_stroke_chain.load_crops` indexes the **raw 5018-file `line/`
directory** of
`dataset/regions_clip_pairs_v3_koma_panels_20260826`, not `koma_split`'s
`held` 969. So `[0:40]`, `[40:171]` and `[171:271]` are reserved **within
Track J's chainer work only** -- they are not disjoint from the koma
`train` split that the fill tracker's `freq_table` arm is fitted on. For a
non-learned instrument that is irrelevant, and it is written down so the word
is not read as more than it is. 4747 panel indices remain unread.
