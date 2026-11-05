"""Prose pipeline, stage 2b: ONE topic model on a balanced mix of
Russian TV subtitles + fairy tales + proza.ru stories (REPORT.md, section 7.3, step 2).

Vocabulary is rebuilt on the union:
  * person names (pymorphy3 tags) and stop words dropped;
  * a capitalised word (>= 80 % capitalised mid-sentence) is dropped as a proper noun only if it is
    rare in EVERY source (TV: < 30 % of series; tales: < 3 % of tales; proza: < 1 % of stories),
    so "Россия", "Москва" survive while character names do not;
  * "too common" (> 50 % of chunks) only if so in at least 2 of 3 sources ("думать" survives);
  * rare: kept if chunk-df >= min(50, max(10, 0.2 % of chunks)) in any source.
Training: 100k TV chunks + 100k proza chunks + all tale chunks x3 + all ficbook chunks x2 (>= 30 tokens).
Inference: TV 20k random episodes, all tales, all ficbook works, the 20k-story proza sample.
Usage:  python src/joint.py
Output: work/JOINT_{TV,TALES,PROZA}/{topics.csv, episodes.csv, doc_topics.npz}
"""
import gzip
import json
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import tomotopy as tp

from project import infer_all, select_texts
from stopwords import STOP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = lambda *p: os.path.join(ROOT, "work", *p)
K, SEED, ITERS = 127, 42, 500
WORKERS = os.cpu_count()
N_TRAIN = {"TV": 100_000, "PROZA": 100_000}
TALE_WEIGHT = 3
FIC_WEIGHT = 2                    # ficbook.net fanfics (~33k chunks)
N_TV_EPISODES = 20_000


def raw_files(src):
    d = W("RU" if src == "TV" else src, "raw")
    return [os.path.join(d, f) for f in sorted(os.listdir(d))]


def iter_raw(src):
    for fn in raw_files(src):
        with gzip.open(fn, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def build_vocab():
    tv = pickle.load(open(W("RU", "vocab_stats.pkl"), "rb"))
    tv = dict(cap=tv[0], chunk_df=tv[1], unit_df=tv[2], tok=tv[3], n_chunks=tv[4],
              n_units=tv[5], prop_thr=0.30)
    stats = {"TV": tv}
    for src, thr in [("TALES", 0.03), ("PROZA", 0.01), ("FICBOOK", 0.03)]:
        if not os.path.exists(W(src, "vocab_stats.pkl")):
            continue
        s = pickle.load(open(W(src, "vocab_stats.pkl"), "rb"))
        stats[src] = dict(cap=s["cap"], chunk_df=s["chunk_df"], unit_df=s["text_df"],
                          tok=s["tok_total"], n_chunks=s["n_chunks"], n_units=s["n_texts"],
                          prop_thr=thr)
    words = set().union(*[set(s["tok"]) for s in stats.values()])
    stop = STOP["RU"]
    keep, reasons = set(), {}
    for w in words:
        if w.startswith("\x00") or w in stop or len(w) < 2:
            continue
        if not any(s["chunk_df"].get(w, 0) >= min(50, max(10, 0.002 * s["n_chunks"]))
                   for s in stats.values()):
            reasons["rare"] = reasons.get("rare", 0) + 1; continue
        a = sum(s["cap"].get(w, (0, 0))[0] for s in stats.values())
        b = sum(s["cap"].get(w, (0, 0))[1] for s in stats.values())
        if b >= 5 and a / b >= 0.8 and all(s["unit_df"].get(w, 0) < s["prop_thr"] * s["n_units"]
                                           for s in stats.values()):
            reasons["proper-noun"] = reasons.get("proper-noun", 0) + 1; continue
        if sum(s["chunk_df"].get(w, 0) > 0.5 * s["n_chunks"] for s in stats.values()) >= 2:
            reasons["too-common"] = reasons.get("too-common", 0) + 1; continue
        keep.add(w)
    print(f"joint vocab: {len(keep)} kept; dropped {reasons}", flush=True)
    return keep


def filt(toks, vocab):
    return [w for w in toks if w in vocab]


def main():
    vocab = build_vocab()
    rng = random.Random(SEED)
    mdl = tp.LDAModel(k=K, alpha=0.1, eta=0.01, seed=SEED)
    mdl.optim_interval, mdl.burn_in = 20, 100
    n_added = {}
    for src in ["TV", "PROZA"]:
        sample, seen = [], 0
        for d in iter_raw(src):
            t = filt(d["t"], vocab)
            if len(t) < 30:
                continue
            seen += 1
            if len(sample) < N_TRAIN[src]:
                sample.append(t)
            else:
                j = rng.randrange(seen)
                if j < N_TRAIN[src]:
                    sample[j] = t
        for t in sample:
            mdl.add_doc(t)
        n_added[src] = len(sample)
    for src, weight in [("TALES", TALE_WEIGHT), ("FICBOOK", FIC_WEIGHT)]:
        if not os.path.isdir(W(src, "raw")):
            continue
        n_added[src] = 0
        for d in iter_raw(src):
            t = filt(d["t"], vocab)
            if len(t) >= 30:
                for _ in range(weight):
                    mdl.add_doc(t)
                n_added[src] += weight
    print("training docs:", n_added, flush=True)
    t0 = time.time()
    for i in range(0, ITERS, 20):
        mdl.train(20, workers=WORKERS)
        print(f"  iter {i + 20}: ll/word={mdl.ll_per_word:.4f} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(W("JOINT"), exist_ok=True)
    mdl.save(W("JOINT", "lda.bin"))
    rows = []
    for k in range(K):
        words = [w for w, _ in mdl.get_topic_words(k, top_n=10)]
        rows.append(dict(topic=k, label=" ".join(words[:7]), top10=" ".join(words),
                         alpha=float(mdl.alpha[k])))
    topics = pd.DataFrame(rows)
    topics.to_csv(W("JOINT", "topics.csv"), index=False)

    model_vocab = set(mdl.used_vocabs)
    for src in ["TALES", "FICBOOK", "PROZA", "TV"]:
        if src != "TV" and not os.path.isdir(W(src, "raw")):
            continue
        out = W(f"JOINT_{src}")
        os.makedirs(out, exist_ok=True)
        if src == "TV":
            ep = pd.read_csv(W("RU", "episodes.csv")).sample(N_TV_EPISODES, random_state=SEED)
        else:
            ep = select_texts(pd.read_csv(W(src, "episodes.csv")), 20000)
        ep.to_csv(os.path.join(out, "episodes.csv"), index=False)
        topics.to_csv(os.path.join(out, "topics.csv"), index=False)
        keep = set(ep.eid)
        n, tot, cov = infer_all(mdl, (d for d in iter_raw(src) if d["eid"] in keep),
                                model_vocab, os.path.join(out, "doc_topics.npz"), K)
        c = dict(texts=len(ep), chunks=n, content_tokens=tot, in_vocab=cov,
                 coverage=round(cov / tot, 4), model="JOINT")
        json.dump(c, open(os.path.join(out, "coverage.json"), "w"), indent=1)
        print(src, c, flush=True)


if __name__ == "__main__":
    main()
