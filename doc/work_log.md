# Track H work log

## 【2026-09-25】track 設立

Track G(ブランチ `panel-generation`)の GT 配置分析・単語の信頼性の見極めと、ユーザー所見
(w342・w343・w344 は「閉じた目」のバリエーションか/顔パーツを先に確定する/配置は単語間の相対位置で
決まり、関係の種類で意味が変わる)を受けて起案・承認された。
起案書: `outbox/track_h_face_words_proposal_20260924.md`。確定事項・資産・運用ルールは `doc/initial_notice.md`。

- ブランチ `face-words` を `panel-generation`(751eea0)から分岐。Track G の文書は `doc/track_g/` に参照用として移した
  (G の最新は `panel-generation` ブランチ側が正)
- 実験の事前登録は必ずこのファイルに**結果を見る前に**書く

### 訂正(同日): フォルダを分けた
- 当初 `lineart-panel-generation` フォルダのままブランチだけ切り替えたが、track ごとにフォルダを分ける
  運用から外れていた(ユーザー指摘)。git worktree で **`/home/sh1/deepl/lineart-face-words`** を作り、
  ここを Track H のフォルダとした。`lineart-panel-generation` は `panel-generation`(Track G)に戻した
- Track G の結果(`results/` は git 管理外)は `../lineart-panel-generation/results/` をパス参照する

