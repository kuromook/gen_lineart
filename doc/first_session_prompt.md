# Track G 最初のセッションに渡すプロンプト(案)

以下をそのまま(または軽く調整して)新しいセッションの最初のメッセージとして渡してください。

---

このリポジトリ `/home/sh1/deepl/lineart-panel-generation`(branch `panel-generation`)で
Track G(コマ生成)の作業を引き継いでください。前身 Track F(線の文法)は測定フェーズ
完了で封存済みです。

まず必ず読むもの:
1. `doc/initial_notice.md` — 目的・5P/5C の操作定義・設計の柱・禁止事項・運用ルール
2. `doc/work_log.md` — 時系列(まだ設立記録だけ)
3. Track F の測定結果の詳細が必要なら `../lineart-stroke-grammar/doc/work_log.md`
   (2026-09-20/21 の項。ただし Track F リポジトリは**読み専用**として扱うこと)

環境:
- Python: `/home/sh1/deepl/lineart/venv/bin/python`(torch 2.7.1+cu118、RTX 3060 12GB)
- 語彙・コーパス等の資産は Track F の results/ をパス参照する
  (initial_notice の資産表に全パスあり。`results/grammar_corpus_20260920/corpus.npz`、
  `results/panel_composition_20260921/`(型ラベル・タグ多値・type_names.json)など)
- Track F のコードを import しない。必要ならファイルごとこのリポジトリにコピーし、
  冒頭に「Track F tools/stroke/XXX.py 由来」と明記する

最初のタスク(設計は initial_notice「最初の一手」のとおり):
条件(ショット型12個+WD14タグ)から「単語集合+配置」を出す生成モデルを、
**セットモデル(一括)**と**2段(選択→配置)**の2案でスモーク規模に実装・比較する。

実行の前に必ず:
1. **work_log に事前登録を書く**(結果を見る前に)。基準案: 配置の bits が
   Track F の実測(条件なし 7.398 / 型条件 7.312、unigram 7.348)を上回ること、
   生成サンプルの目視モンタージュで「コマとして読める」こと。
   基準はこの案のままでよいが、変えるなら結果を見る前に。
2. スモークテスト(小さい limit)で道具が動くことを確認してから本番。
3. バックグラウンド実行は setsid、GPU 長時間ジョブは月〜木のバッチ枠のルールに従う。

鉄則(initial_notice「運用ルール」より。逸脱しない):
- 新しい数値が出たら必ず「これは道具を測っていないか」を疑う
- 数値には必ず results/ 配下の目視モンタージュを併記
- 学習損失で進捗を判断しない。基準の事後緩和禁止
- 待機ループや pkill -f で自分を巻き込まない

単語の名づけ(Track F の naming UI、port 8471、Track F 側で動作中)は並行して
使ってよいが、命名は解釈用であり定量判定に使わない。

---
