# lineart-face-words — Track H(ブランチ `face-words`)

顔パーツ単語の確定: 目・鼻・口・眉・耳について、機械の語(`wNNN`)と人の戦略ラベルを対応づけ、
機械的に再現できる規則を確定する。確定したラベルは Track G(配置)へ渡す。
設計・現状・運用ルールは `doc/initial_notice.md`、時系列は `doc/work_log.md`。
このフォルダは `lineart-panel-generation` リポジトリの git worktree。Track G(コマ生成・配置)は
`../lineart-panel-generation`(ブランチ `panel-generation`、読み専用として参照。このブランチでは `doc/track_g/` に参照用の写し)。
前身(測定フェーズ): `../lineart-stroke-grammar` (Track F)
