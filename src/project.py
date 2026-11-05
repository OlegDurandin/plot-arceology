"""Prose pipeline, stage 2a: project prose chunks onto an existing topic model (no retraining).

Default: the Russian TV-subtitle model work/RU/lda.bin. Tokens outside its vocabulary are dropped.
Usage:  python src/project.py TALES|PROZA [model_dir=RU] [max_texts=20000]
Output: work/PROJ_<SRC>/{topics.csv, episodes.csv, doc_topics.npz, coverage.json}
"""
import gzip
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import tomotopy as tp

from stopwords import STOP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKERS = os.cpu_count()
INFER_ITERS = int(os.environ.get("INFER_ITERS", 40))


def select_texts(ep, max_texts):
    if len(ep) <= max_texts:
        return ep
    if "bucket" in ep:                       # proza.ru: deterministic crc32 sample
        return ep.nsmallest(max_texts, "bucket")
    return ep.sample(max_texts, random_state=0)


def iter_chunks(src_dir, keep):
    for fn in sorted(os.listdir(os.path.join(src_dir, "raw"))):
        with gzip.open(os.path.join(src_dir, "raw", fn), "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["eid"] in keep:
                    yield d


def infer_all(mdl, chunks, vocab, out_npz, K):
    eids, idx, ns, lens, counts = [], [], [], [], []
    tot = cov = 0
    batch, meta = [], []

    def flush():
        if not batch:
            return
        res, _ = mdl.infer([mdl.make_doc(t) for t in batch], iterations=INFER_ITERS, workers=WORKERS)
        for (e, i, n, L), dist in zip(meta, res):
            eids.append(e); idx.append(i); ns.append(n); lens.append(L)
            counts.append(np.asarray(dist, np.float32) * L if L else np.zeros(K, np.float32))
        batch.clear(); meta.clear()

    t0 = time.time()
    for d in chunks:
        toks = [w for w in d["t"] if not w.startswith("\x00") and w not in STOP["RU"] and len(w) > 1]
        kept = [w for w in toks if w in vocab]
        tot += len(toks); cov += len(kept)
        batch.append(kept or ["\x01"])
        meta.append((d["eid"], d["i"], d["n"], len(kept)))
        if len(batch) >= 100_000:
            flush()
            print(f"  inferred {len(eids)} chunks ({time.time() - t0:.0f}s)", flush=True)
    flush()
    np.savez_compressed(out_npz, counts=np.vstack(counts), eid=np.array(eids),
                        i=np.array(idx, np.int16), n=np.array(ns, np.int16),
                        length=np.array(lens, np.int32))
    return len(eids), tot, cov


def main(src, model_dir="RU", max_texts=20000):
    src = src.upper()
    src_dir = os.path.join(ROOT, "work", src)
    mdir = os.path.join(ROOT, "work", model_dir)
    out = os.path.join(ROOT, "work", f"PROJ_{src}" if model_dir == "RU" else f"{model_dir}_{src}")
    os.makedirs(out, exist_ok=True)
    mdl = tp.LDAModel.load(os.path.join(mdir, "lda.bin"))
    vocab = set(mdl.used_vocabs)
    ep = select_texts(pd.read_csv(os.path.join(src_dir, "episodes.csv")), int(max_texts))
    ep.to_csv(os.path.join(out, "episodes.csv"), index=False)
    shutil.copy(os.path.join(mdir, "topics.csv"), os.path.join(out, "topics.csv"))
    n, tot, cov = infer_all(mdl, iter_chunks(src_dir, set(ep.eid)), vocab,
                            os.path.join(out, "doc_topics.npz"), mdl.k)
    c = dict(texts=len(ep), chunks=n, content_tokens=tot, in_vocab=cov,
             coverage=round(cov / tot, 4), model=model_dir)
    json.dump(c, open(os.path.join(out, "coverage.json"), "w"), indent=1)
    print(c, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
