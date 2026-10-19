# Lineart Generator

ラフ画像(手書きスケッチ)から線画を生成する深層学習プロジェクト。

ただし**現在の中心課題はモデルの学習そのものではありません**。「線画とは何か」を
人が読める形で測り、その表現から生成する、という段階にいます。理由は下の
「いまの状態」のとおりです。

更新: 2026-10-19。**正本は `doc/CURRENT.md`** です。この README は入口で、
数と判定の現住所は常に台帳側にあります。

---

## いまの状態

- **学習したどのモデルも、前処理器(`LineartDetector`)単体に勝てていません。**
  だから評価のベースラインは他のモデルではなく、**常に前処理器の素の出力**です。
  どんな数字を出すときも、条件画像自身の score を隣に置きます。
- **ピクセル一致で近づける路線は 2026-10-04 に棄却されました。** 人の判定で、
  ラフの墨から完璧に削除したオラクル(このプロジェクト自身の「完璧な削除」の定義)が
  **線画として 0/15 使えない**、GT は 14/15。つまり
  **線画はラフの墨の部分集合ではありません。** 削除は前段の掃除としては残りますが、
  成果物への経路ではありません。
- **現在の路線: ストロークを語、コマを文として扱う。** 表現を先に作り、そこから生成します。
- **作業は track(テーマごとのブランチ)で並行**して進みます。このブランチ
  `foundation` は共通基盤で、スクリプト・データセットの経路・プロジェクト文書の置き場所です。
  実験そのものは track 側で走ります。

---

## 読む順番

| 文書 | 役割 |
|---|---|
| `doc/CURRENT.md` | **正本の台帳。** 現在の状態・各 track の到達点・既知の罠・次の一手 |
| `doc/glossary.md` | 用語、確立した事実、既知の罠(対照が生まれつき無意味になる道) |
| `doc/worktree_policy.md` | 運用規則。track の作り方、誰が push するか、ブランチの素性 |
| `doc/work_log.md` | 時系列の実験ログ |
| `results/CURRENT.md` | 結果物の入口 |
| `doc/preprocess/EXTRACTION_RULES.md` | 生データからの抽出・保存・監査ルール |

`inbox/` と `outbox/` は往復書簡専用で git 管理外です(プロジェクト文書は `doc/` に置きます)。

---

## トラック一覧

大テーマごとに track(ブランチ / worktree)を立てて並行して進めます。
**状態と到達点の正本は `doc/CURRENT.md` の Track Ledger** で、ここは索引です。
各 track は自分の `doc/initial_notice.md`(引き継ぎ)と `doc/work_log.md`(時系列)を持ちます。

### 語の路線(現在の前線)

ストロークを語、コマを文として扱う路線。2026-09-18 のユーザーの再定義で開きました。

| | track | ブランチ | 問い | 状態 |
|---|---|---|---|---|
| **F** | stroke-grammar | `stroke-grammar` | ストロークを語として符号化できるか。語の間に文法はあるか | **測定終了・封印**(読める状態で保存、変更しない) |
| **G** | panel-generation | `panel-generation` | 概念を与えて、語を選び配置できるか | 稼働中(別クローン)。H の単位待ち |
| **H** | face-words | `face-words` | 顔の部位語(目・鼻・口・眉・耳)を人の戦略ラベルで固定し、機械規則で再現できるか | **稼働中・この路線の前線** |
| **K** | stroke-topology | `stroke-topology` | 1px 細線化とリンクが作る接続の誤り。**語の単位そのもの**が正しいか | 静止中(新しい作者の印を待つ) |

- **F の結果は明確な否定で、この路線の形を決めました** —— **コマの中に語の文法は無い**。
  自己回帰 6.94 bits/word 対 unigram 7.35、bigram は 7.40 で unigram より悪い、
  12 の shot type で 0.036 bits、WD14 40 タグで 0 —— 5つの測定が一致。
  **ただし語自体は実在します**(FSQ 500 語の符号帳、92% 使用)。配置には構造がある
- **K の到達点**: 作者の「解体すべき線」の印は**畳まれた橋**を通るトークンに集中する
  (PAF 0.4855、取り分けで支持、絵柄に固有でない)。**しかしその規則では修理が届かない**
  (精度 3.8% 対 必要 50%)。寄与割合は議論の資格、作れるかは精度が決める

### ベタ(面)の路線

| | track | ブランチ | 問い | 状態 |
|---|---|---|---|---|
| **J** | solid-fill | `solid-fill` | ラフからベタを予測できるか。**線ではなく面**の問題 | 稼働中 |

前処理器の `fill_ratio` が 0.0%(GT は 24.5%)で、**エッジ検出器は原理上面を塗れない**。
「学習したモデルは前処理器に勝てない」が構成上当てはまらない唯一の残差です。

### 選択の路線(保留)

| | track | ブランチ | 問い | 状態 |
|---|---|---|---|---|
| **C** | stroke-selection | `stroke-selection` | 前処理器が引きすぎた線のどれを消すか | 保留(2026-09-17 以降) |

画素単位の分類器は、床を揃えると**ベースライン超え +0.0258**。ただし消すべき線の 37% を
消してしまう(`c_survival` 0.6283)。**2026-10-04 に路線自体は棄却**(人の判定で 0/15)
されましたが、前段の掃除としては残っています。

### 保留 / 終了

| | track | ブランチ | 結果 |
|---|---|---|---|
| **E** | aesthetic-judge | `aesthetic-judge` | 保留。**壊れて読めないものを美しいとは言えない** —— 美の軸ではなく「壊れ/可読」の軸だった、という知見が残った |
| **D** | pair-signal | `pair-signal` | 終了。位置の合ったペアでも**条件画像に無い GT の線は描かない**(lesson 8)。生成でのピクセル一致路線を閉じた |
| **I** | image-prompt | `image-prompt` | 自ら終了(2026-10-04)。IP-Adapter は空間的に局在しないので、削除にも未発見の線にも効かない |
| **A/B** | controlnet-sd15-refine / controlnet-sdxl-fidelity | 同名 | 終了。**diffusion は前処理器に勝てない** |
| — | cleanup-refiner | `cleanup-refiner` | 休止(2026-08-04、CNN+GAN 期) |
| — | controlnet-realpairs | `controlnet-realpairs` | 終了。クロスハッチへのハルシネーションの究明は目的達成 |

### この基盤(`foundation`)の役目

**記録と回送**です。実験は track 側で走ります。基盤は台帳・用語集・運用規則を持ち、
通達を出し、帰無値と帯(合格条件)の確認を引き受けます。

---

## 確立した事実(抜粋)

全文と出典は `doc/glossary.md`。ここは「これを知らずに実験を設計すると必ず戻ってくる」もの:

- **前処理器は GT の 1.4 倍の墨を置く。** 削除だけのオラクルで f1 **0.7425**、
  同じ 192 タイルで前処理器自身は 0.3231。削除の天井はここ
- **ベタ(塗り)には前処理器が原理上触れない**(`fill_ratio` 0.0% 対 GT 24.5%)。
  エッジ検出器は面を塗れないので、**ここだけは構造的に負けません**
- **「何も描かない」でも f1 0.1941 出ます**(`holdout_lineart_family`)。
  だから `bsds_f1` は単独では使えません
- **線の方程式は破片単位でしか成立しません。** 描かれた線は放物線1本ではなく、短い鎖
- **交差点で線をつなぐとき曲率は役に立ちません。** 効くのは接線の一致と横ずれ
- **線画は面をほとんど囲んでいません**(1コマあたり約 1.2 領域)。
  だから「面」= ベタの連結成分
- **つながりの失敗の 81.1% は、対にする段階に届く前に落ちています**
- **作者の「ここは解体すべき線」という印は、畳まれた橋を通るトークンに集中します**
  (PAF 0.4855、10作品を外した取り分けで支持、絵柄に固有ではない)。
  **しかしその規則では直せません** —— 曝露の精度が約 4% で、費用の棒が要求する 50% から 13 倍遠い

leakage のあった時期の高い f1(旧 `shape1` / `std15` 系)は**採用基準に使いません**。

---

## 進め方の規則

このリポジトリの性格はここに出ます。track を開く人はまずこれに従ってください。

- **事前登録。結果を見た後で棒(合格条件)を動かさない。** 曝露や母集団を
  棒を越えるために広げるのも、事後に棒を動かすのと同じ
- **器具確認(IC)と陰性対照(NC)を本番の前に。** 器具が自分自身を測っていないこと
- **判定保留は正直な結論。** 撤回した数は記録から消さず、撤回と併記する
- **数は出典(通達 / commit)と一緒に。判定語は棒と一緒に** —— 棒なしの「良い」は記録にならない
- **プールを跨いで平均しない。** 行ごとに印字する

---

## ブランチ

- **`foundation`** — 共通基盤。全 track はここから生やします
  (`git worktree add ../lineart-<topic> -b <topic> foundation`)。
  2026-10-18 に `diffusion` から改名(閉じた方向の名前だったため)
- **`paper-scout`** — cross-domain paper scout(他分野の論文から転用可能な着想を探す週次ジョブ)の専用レーン。
  常時起動マシンの cron が `candidates*.md` だけを push します。
  2026-10-19 に `main` から改名(trunk と誤読される名前だったため)。`main` ブランチはもうありません
- **`<topic>`** — track ごとのブランチ。一覧と物理位置(worktree か別クローンか)は
  `doc/worktree_policy.md`。**ブランチの push は track 自身がやります**

---

## 環境

- Ubuntu 22.04 / RTX 3060 (12GB) / CUDA 12.8 / Python 3.10

```bash
pip install torch torchvision pillow opencv-python
```

長時間の GPU ジョブは**月〜木の連続バッチが標準枠**です(`doc/worktree_policy.md` の
Schedule Rules)。所要時間を理由に手法を却下・縮小しません。

---

## データセット構成

```
dataset/
  pairs_480/   # 480x480 ペア(主学習データ)
    train/rough/  train/line/
    test/rough/   test/line/
    *.txt         # 学習・評価用ファイルリスト
  pairs_256/   # 256x256 生成ペア
  pairs/       # 旧 256x256 ペア
  raw/         # 元データ
  psd_line_koma_panels_20260918/   # コマ単位の抽出(語の路線で使う)
```

新しい raw dataset は、同一座標抽出を行う前に alignment 診断と QC を通してください
(`doc/preprocess/EXTRACTION_RULES.md`)。

---

## 学習

```bash
python scripts/train.py \
  --file-list dataset/pairs_480/<train_list>.txt \
  --checkpoint-dir checkpoints/<experiment_name> \
  --autocontrast
```

| オプション | デフォルト | 説明 |
|---|---|---|
| `--file-list` | `dataset/pairs_480/valid_train.txt` | 学習ファイルリスト |
| `--checkpoint-dir` | `checkpoints/base` | チェックポイント保存先 |
| `--resume` | なし | 途中再開するチェックポイントパス |
| `--autocontrast` / `--no-autocontrast` | True | rough に autocontrast を適用 |

チェックポイントは 10 epoch ごとと best loss 更新時に保存されます。

## 推論

```bash
python scripts/inference.py \
  --checkpoint checkpoints/<experiment_name>/best.pth \
  --input path/to/rough.jpg \
  --output results/output.png \
  --autocontrast
```

---

## ファイル構成

| ファイル | 役割 |
|---|---|
| `lineart/unetgenerator.py` | UNetGenerator(ResBlock, DilatedConvBlock) |
| `lineart/losses.py` | edge / tolerant F1 / ink loss |
| `scripts/train.py` | 学習スクリプト |
| `scripts/inference.py` | 推論スクリプト |
| `tools/evaluation/stroke_decomposition.py` | 低い `neither_ink` と高い `b_recall` の両立を測る(単一指標の代わり) |
| `tools/evaluation/audit_pair_dataset_integrity.py` | train/eval leakage と欠損の監査 |
| `tools/evaluation/evaluate_fixed_outputs.py` | 固定サンプル評価 |
| `tools/pair_extraction/` | raw pair 抽出・alignment・clean list 生成 |
| `tools/preprocess/` | 前処理(コマ分割・タイル化など) |
| `experiments/` | 実験ランナー |
