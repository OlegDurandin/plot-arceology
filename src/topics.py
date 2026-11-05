"""Stage 2: 127-topic LDA (collapsed Gibbs sampling, like Mallet) on chunk documents.

Usage:  python src/topics.py EN|RU [n_train_chunks] [iterations]
Output: work/<lang>/topics.csv        topic id, top words
        work/<lang>/doc_topics.npz    per-chunk expected token counts per topic + chunk meta
"""
import gzip
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import tomotopy as tp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = 127
INFER_ITERS = int(os.environ.get("INFER_ITERS", 40))
SEED = 42
WORKERS = os.cpu_count() or 16


def read_chunks(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main(lang, n_train=300_000, iters=500):
    work = os.path.join(ROOT, "work", lang.upper())
    src = os.path.join(work, "chunks.jsonl.gz")

    # ---- training sample (reservoir) -------------------------------------------
    rng = random.Random(SEED)
    sample, seen = [], 0
    for d in read_chunks(src):
        if len(d["t"]) < 30:
            continue
        seen += 1
        if len(sample) < n_train:
            sample.append(d["t"])
        else:
            j = rng.randrange(seen)
            if j < n_train:
                sample[j] = d["t"]
    print(f"training on {len(sample)} of {seen} chunks", flush=True)

    mdl = tp.LDAModel(k=K, alpha=0.1, eta=0.01, seed=SEED)
    mdl.optim_interval = 20      # hyper-parameter (asymmetric alpha) optimisation, as Mallet
    mdl.burn_in = 100
    for t in sample:
        mdl.add_doc(t)
    del sample
    t0 = time.time()
    for i in range(0, iters, 20):
        mdl.train(20, workers=WORKERS)
        print(f"  iter {i + 20}: ll/word={mdl.ll_per_word:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    mdl.save(os.path.join(work, "lda.bin"))

    rows = []
    for k in range(K):
        words = [w for w, _ in mdl.get_topic_words(k, top_n=10)]
        rows.append(dict(topic=k, label=" ".join(words[:7]), top10=" ".join(words),
                         alpha=float(mdl.alpha[k])))
    pd.DataFrame(rows).to_csv(os.path.join(work, "topics.csv"), index=False)

    # ---- inference for every chunk -------------------------------------------------
    eids, idx, ns, lens, counts = [], [], [], [], []
    batch_docs, batch_meta = [], []

    def flush():
        if not batch_docs:
            return
        docs = [mdl.make_doc(t) for t in batch_docs]
        res, _ = mdl.infer(docs, iterations=INFER_ITERS, workers=WORKERS)
        for (e, i, n, L), dist in zip(batch_meta, res):
            eids.append(e); idx.append(i); ns.append(n); lens.append(L)
            counts.append((np.asarray(dist, dtype=np.float32) * L) if L else np.zeros(K, np.float32))
        batch_docs.clear(); batch_meta.clear()

    t0 = time.time()
    for d in read_chunks(src):
        batch_docs.append(d["t"] if d["t"] else ["\x01"])  # empty chunk placeholder
        batch_meta.append((d["eid"], d["i"], d["n"], len(d["t"])))
        if len(batch_docs) >= 100_000:
            flush()
            print(f"  inferred {len(eids)} chunks ({time.time() - t0:.0f}s)", flush=True)
    flush()
    np.savez_compressed(os.path.join(work, "doc_topics.npz"),
                        counts=np.vstack(counts), eid=np.array(eids), i=np.array(idx, np.int16),
                        n=np.array(ns, np.int16), length=np.array(lens, np.int32))
    print("done", len(eids), "chunks", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(int(x) for x in a[1:]))
