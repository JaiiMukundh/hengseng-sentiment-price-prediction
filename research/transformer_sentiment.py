"""Score social posts with FinBERT (English) and KR-FinBERT (Korean); lexicon for Chinese/other."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd

from sentpred.sentiment import score_text

KOREAN = re.compile(r"[가-힣]")
CJK = re.compile(r"[\u4e00-\u9fff]")


def score_posts(posts: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    from transformers import pipeline

    posts = posts.reset_index(drop=True).copy()
    texts = posts["text"].fillna("").astype(str).tolist()
    scores = [0.0] * len(texts)

    korean_idx = [i for i, t in enumerate(texts) if KOREAN.search(t)]
    english_idx = [i for i, t in enumerate(texts) if not KOREAN.search(t) and not CJK.search(t)]
    other_idx = [i for i, t in enumerate(texts) if i not in set(korean_idx) | set(english_idx)]

    print(f"posts={len(texts)} korean={len(korean_idx)} english={len(english_idx)} other={len(other_idx)}")

    for name, model, indices in (
        ("KR-FinBERT", "snunlp/KR-FinBERT-SC", korean_idx),
        ("FinBERT", "ProsusAI/finbert", english_idx),
    ):
        if not indices:
            continue
        print(f"loading {name}...")
        clf = pipeline("text-classification", model=model, tokenizer=model,
                       truncation=True, max_length=256, device=-1, top_k=None)
        start = time.time()
        for pos in range(0, len(indices), batch_size):
            batch_idx = indices[pos:pos + batch_size]
            batch = [texts[i][:1000] for i in batch_idx]
            outputs = clf(batch)
            for i, labels in zip(batch_idx, outputs):
                probs = {d["label"].lower(): d["score"] for d in labels}
                scores[i] = float(probs.get("positive", 0.0) - probs.get("negative", 0.0))
            if pos % (batch_size * 10) == 0:
                print(f"  {name}: {pos + len(batch_idx)}/{len(indices)} ({time.time()-start:.1f}s)")

    for i in other_idx:
        scores[i] = score_text(texts[i])[0]

    posts["sent_score"] = scores
    return posts


if __name__ == "__main__":
    posts = pd.read_csv("data/processed/social_posts.csv")
    out = score_posts(posts)
    out.to_csv("data/processed/social_posts_transformer.csv", index=False)
    print("saved", len(out))
    print(out["sent_score"].describe().round(3).to_string())
