"""Candidate word list for the 2026-09-05 exhaustive caption-word ablation
sweep (inbox/initial_notice.md: after hypotheses #1-5 all rejected,
including the "manga panel" removal (hypothesis #5) which unexpectedly
made orientation_entropy *worse*, the user asked to inventory every other
word that appears in data/captions.csv's tags column and check each one's
individual contribution to the cross-hatch bias, not just "manga"/"comic").

All Danbooru-style tags from data/captions.csv appearing in >= 0.5% of the
8467 training rows (42+ rows), sorted by frequency descending. Computed by
results/word_ablation_20260905/tally_tags.py (see that script's output in
inbox/initial_notice.md for the full frequency table).
"""

CANDIDATE_TAGS = [
    "monochrome", "greyscale", "solo", "white_background", "1girl", "comic",
    "simple_background", "close-up", "no_humans", "open_mouth", "smile",
    "long_hair", "silent_comic", "closed_eyes", "1boy", "male_focus",
    "breasts", "short_hair", "head_out_of_frame", "blush", "sketch",
    "cable", "bangs", "sweatdrop", "portrait", "glasses", "looking_at_viewer",
    "sweat", "text_focus", "closed_mouth", "collarbone", "nude",
    "multiple_girls", "hat", "large_breasts", "pokemon_(creature)", "2girls",
    "chibi", "letterboxed", "silhouette", "shirt", "admiral_(kancolle)",
    "yuri", ":3", "signature", "musical_note", "upper_body", "black_border",
    "english_text", "weapon", "nipples", "cleavage", "from_side", "bow",
    "sleeping", "bikini", "lineart", "|_|", "fujiwara_no_mokou", "swimsuit",
    "hair_between_eyes", "blood", "facial_hair",
]
