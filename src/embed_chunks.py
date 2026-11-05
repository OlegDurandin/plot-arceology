"""Chunk texts + sentence embeddings for the BERTopic / embedding-arc experiments (H6).

Chunks are rebuilt with exactly the same chunking as prep.py (subtitles: 2-4 min by screen
time) and prep_text.py (prose: 150-300 words by sentences), so (eid, i) match the LDA data.
Sources (one shared multilingual embedding space):
  TV_RU, TV_EN : the 20k held-out episodes used by ksweep.py (sample(20000, random_state=0))
  TALES, FICBOOK: all texts;  PROZA: the 20k-story sample used by project.py
Model: intfloat/multilingual-e5-small (512-token window, 384-d), "passage: " prefix; chunks longer than
220 words are embedded as several windows and mean-pooled (word-count weights).
Usage:  python src/embed_chunks.py [SRC ...]
Output: work/EMB/<SRC>.npz  (emb float16 [N,384], eid, i, n, words) and <SRC>_texts.jsonl.gz
"""
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from paths import SERIALS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "work", "EMB")
MODEL = os.environ.get("EMB_MODEL", "intfloat/multilingual-e5-small")
BATCH = int(os.environ.get("EMB_BATCH", 64))
WIN = 220                          # words per embedding window (~400-480 e5 tokens for Russian)
N_TV = 20000


def tv_chunks(lang):
    from prep import chunk_count, parse_srt
    ep = pd.read_csv(os.path.join(ROOT, "work", lang, "episodes.csv"))
    ep = ep.sample(N_TV, random_state=0)
    for r in ep.itertuples():
        path = os.path.join(SERIALS, r.series, f"Season{r.season}",
                            f"{r.series}_s{r.season}_e{r.episode}_{lang}.srt")
        try:
            blocks = parse_srt(path)
        except OSError:
            continue
        dur = max(b[1] for b in blocks)
        n = chunk_count(dur)
        if n != r.n_chunks:
            continue
        parts = [[] for _ in range(n)]
        for mid, _, lines in blocks:
            k = min(n - 1, max(0, int(mid / dur * n)))
            parts[k].extend(lines)
        for i, p in enumerate(parts):
            yield r.eid, i, n, " ".join(p)


def prose_chunks(src):
    import prep_text
    if src == "PROZA":
        keep = set(pd.read_csv(os.path.join(ROOT, "work", "PROJ_PROZA", "episodes.csv")).eid)
    else:
        keep = set(pd.read_csv(os.path.join(ROOT, "work", src, "episodes.csv")).eid)
    df = prep_text.load(src)
    for r in df[df.eid.isin(keep)].to_dict("records"):
        sp = prep_text.split_chunks(r["text"], r["para_sep"])
        if sp is None:
            continue
        n, parts = sp
        for i, part in enumerate(parts):
            yield r["eid"], i, n, " ".join(s[0] for s in part)


def main(srcs):
    from sentence_transformers import SentenceTransformer
    os.makedirs(OUT, exist_ok=True)
    model = SentenceTransformer(MODEL, device="mps")
    model.max_seq_length = 512
    for src in srcs:
        if os.path.exists(os.path.join(OUT, f"{src}.npz")):
            print(src, "exists, skip", flush=True)
            continue
        t0 = time.time()
        gen = tv_chunks(src[3:]) if src.startswith("TV_") else prose_chunks(src)
        meta, texts = [], []
        with gzip.open(os.path.join(OUT, f"{src}_texts.jsonl.gz"), "wt", encoding="utf-8") as f:
            for eid, i, n, t in gen:
                meta.append((eid, i, n, len(t.split())))
                texts.append(t)
                f.write(json.dumps({"eid": eid, "i": i, "n": n, "text": t}, ensure_ascii=False) + "\n")
        print(f"{src}: {len(texts)} chunks extracted ({time.time() - t0:.0f}s)", flush=True)
        # long chunks (subtitles: median ~540 tokens) are split into windows of <= WIN words so that
        # nothing is truncated at 512 tokens; window embeddings are averaged with word-count weights
        win_text, win_owner, win_w = [], [], []
        for j, t in enumerate(texts):
            w = t.split()
            k = max(1, -(-len(w) // WIN))
            for q in range(k):
                part = w[q * len(w) // k:(q + 1) * len(w) // k]
                win_text.append(" ".join(part)); win_owner.append(j); win_w.append(max(1, len(part)))
        win_owner, win_w = np.array(win_owner), np.array(win_w, np.float32)
        order = np.argsort([len(t) for t in win_text])      # length-sorted batches: less padding
        dim = model.get_sentence_embedding_dimension()
        acc = np.zeros((len(texts), dim), np.float32)
        step = 20000
        for s in range(0, len(order), step):
            idx = order[s:s + step]
            e = model.encode(["passage: " + win_text[j] for j in idx], batch_size=BATCH,
                             normalize_embeddings=True, show_progress_bar=False)
            np.add.at(acc, win_owner[idx], e * win_w[idx, None])
            print(f"  {src}: embedded {min(s + step, len(order))}/{len(order)} windows "
                  f"({time.time() - t0:.0f}s)", flush=True)
        acc /= np.linalg.norm(acc, axis=1, keepdims=True) + 1e-12
        emb = acc.astype(np.float16)
        np.savez_compressed(os.path.join(OUT, f"{src}.npz"), emb=emb,
                            eid=np.array([m[0] for m in meta]), i=np.array([m[1] for m in meta], np.int16),
                            n=np.array([m[2] for m in meta], np.int16),
                            words=np.array([m[3] for m in meta], np.int32))
        print(f"{src}: done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["TALES", "FICBOOK", "TV_RU", "TV_EN", "PROZA"])
