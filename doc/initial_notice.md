# Track O(`krita-cleanup`)—— 引き継ぎ

**開設**: 2026-11-04(ユーザーの指示)/ **元**: `foundation` @ `847b8fb`
**この文書は短く保つ。履歴は `doc/work_log.md` に溜める。**

## 1. この track の目的

> **Krita の Fast Sketch Cleanup(非生成系、OSS)を調べる。** 下絵を線画に変える処理で、
> **この企画が初期にやっていた下絵→線画の直接変換に近い。**

**なぜ要るか**: この企画の直接回帰の系統は、**どのモデルも前処理器(`LineartDetector`)
単体に勝てずに止まった**(台帳 `Lessons` / メモリ「Preprocessor Baseline」)。
Fast Sketch Cleanup は**別のチームが、別のデータで、同じ種類の問題を解いたもの**。
**その結果が、線画という表現について何を言っているか**を取り出す(第1条)。
**「良い道具かどうか」を決める track ではない。**

**言わないこと**: 「Krita の方が良い / 悪い」。棒は2本(前処理器の素の出力が下限、
本当の棒は人の判定)。**固定の棒に勝っても、目的に近づいたとは言えない。**

## 2. 既に分かっていること(**基盤が 2026-11-04 に公開資料だけを読んだ。未確認**)

出所: <https://krita.org/en/posts/2024/fast_sketch_background/> /
<https://krita.org/en/posts/2024/fast_sketch_plugin/>

- **構造**: 畳み込みと逆畳み込みだけ(全結合なし)、最後に sigmoid。
  **Simo-Serra et al. 2016(Learning to Simplify)の記述に倣っている。** 変種が複数
  (`tinyTinier` / `tooSmallConv` / `typicalDeep` / `tinyNarrowerShallow` / `original` ほか)
- **損失**: `mse`、または `blackWhite`(0.5 付近の画素で小さくなる mse の変形)。最適化は adadelta
- **データ**: **寄付された作品の対(下絵 → 線画)だけ**。増強は回転・拡縮・鏡映・反転
- **推論**: OpenVINO(CPU / GPU / NPU)。入力 256 のタイルに切り、余白付きで処理して繋ぐ
- **学習コード**: <https://invent.kde.org/tymond/fast-line-art>(資料の時点で `33869b6`)
- **データセット**: <https://files.kde.org/krita/extras/FastSketchCleanupPluginKritaDataset.zip>
  (**ライセンスを注意して読め**と資料にある。中身は未確認)
- **プラグイン**: <https://download.kde.org/stable/krita/FastSketchPlugin-1.0.2/FastSketchPlugin1.0.2.zip>
- **Simo-Serra のモデルは Krita のライセンスと合わず使わなかった**と書いてある

**この節は基盤の要約で、一次資料ではない。数や名前を引用する前に、一次資料で確かめる。**

## 3. 手順(**段ごとに区切る。前の段の結論が出るまで次へ進まない**)

**段1 —— 読む(画素に触らない)**
- 学習コード・プラグイン・データセットの**ライセンスを3つ別々に**記録する(コード / 重み / データ)
- 構造・損失・前処理・タイルの切り方・**対の位置合わせをどうしているか**
- **何で評価したか** —— 人の判定か、数か。数なら、その数の床は何か
- 出力: `doc/survey_krita_fsc_<日付>.md`。**分かったこと / 推定 / 不明を分けて書く**

**段2 —— 配られた重みを手元の対に当てる**(**走る前に登録する**。`moat-register`)
- **器具**: `tools/evaluation/stroke_decomposition.py`(**条件画像と GT の対**から A/B/C を作る。
  この題材は対があるので使える)。**前処理器の素の出力と、条件画像自身の score を隣に置く**
- **帰無値を棒の前に出す**(規律9)。**落としたタイルは理由別に数えて印字する**
- **世代ごとに行を分ける**(`ako6` と `ako7` を1行にしない)。既知セルは**世代ごとに1つ**
- **変種ごとに行を分ける**。**一番良い変種だけを見出しにしない**(読んでいない腕を見出しに入れない)

**段3 —— 手元の対で学習し直す**(**ユーザーの判断が要る。段2の後で基盤経由で聞く**)

## 条例

**上位法**: 規律は `../lineart/doc/CURRENT.md § Working Discipline`、運用は
`../lineart/CLAUDE.md`。**この木の `CLAUDE.md` は branch 時点の複製なので読まない**(T92)。
**画素と重みの扱い(許可表・`data/` 配下・重みの承認)も上位法。写さない。**

1. **外から来たものは、原稿と混ぜない。** 置き場を分け、名前に `krita_` を付ける
   (`data/krita_*` / `results/krita_*`)。**寄付データセットの画像は他人の作品である** ——
   原稿と同じく**追跡しない・外に出さない**。ライセンスを読むまで手元に展開しない
2. **配られた重み(OpenVINO の `.xml` + `.bin`、PyTorch の `.pth` 等)は追跡しない。**
   `.bin` は重みの拡張子表にある。**自分で学習した重みの公開はユーザーの承認**
3. **落としてきたコードは、空の専用ディレクトリに置き、`python -I` で読む**
   (同名モジュールの横取りを避ける)。**自分の器具はその外に置く**
4. **公開資料の主張は、確かめるまで「Krita の資料では」と主語を付けて書く。**
   この track の測定と混ぜない
5. **原稿の情報の列は自分で作らない** —— 要る列は `census` に頼む(結果は census に返させる)
6. **絵が何を描いているかは作者が決める**(規律6)。出力を論じるときは**原寸の切り出し**

## 4. 実行環境で効くこと

- **Python は `/home/sh1/deepl/lineart/venv/bin/python`。この worktree に `venv/` は無い**
- GPU は1枚。**2026-11-04 時点で 7.5GB 使用中**(`stable-diffusion-webui-forge` の常駐と思われる)。
  OpenVINO の推論は CPU で足りるはず。**GPU を使う前に `nvidia-smi`**
- **長時間ジョブは月〜木の連続バッチ。所要時間を理由に手法を却下・縮小しない**
- **対の在り処**: `/home/sh1/deepl/lineart/dataset/pairs_480/`(`.gitignore` 配下)

## 5. 往復書簡

- `inbox/` `outbox/` は往復書簡専用で `.gitignore` 配下。**プロジェクト文書は必ず `doc/` に置く**
- 基盤へ: `cp outbox/<file> ../lineart/inbox/`
- **push はこの track 自身がやる**: `git push -u origin krita-cleanup`。基盤は `foundation` だけ

## 6. 次の手

**段1 は 2026-11-04 に済んだ**(`doc/survey_krita_fsc_20261104.md`。基盤へ通達済み)。**次は段2 の登録**
(`moat-register`)。申し送りは調査書 §6。
