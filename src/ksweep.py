"""Sensitivity of the results to the number of topics K.

For every K the LDA is trained exactly as in topics.py (300k chunks, 500 Gibbs iterations,
alpha optimisation) but on chunks from training episodes only; evaluation uses a fixed set of
held-out episodes (the same for every K), so all numbers are comparable across K.

Per K:
  topic quality  : NPMI and C_v coherence (top-10 words, on a fixed 30k-chunk reference sample of the
                   training corpus), topic diversity, held-out log-likelihood/word (see REPORT: not
                   comparable across K, tomotopy's infer ll includes the document-topic prior)
  plot arc       : share of topics with |r_lin| > 0.9 / 0.7, arcing topics, 6-topic path r,
                   PCA variance of 2 PCs, |r(PC1, time)|, |r(PC2, middle)|, z(PC2) vs random walks,
                   agreement of the 12-part PC1/PC2 trajectory with the main K=127 model
Usage:  python src/ksweep.py RU|EN K1,K2,... [seeds=42] [n_eval_episodes=20000]
Output: work/KSWEEP/<LANG>_K<k>_s<seed>/{topics.csv,doc_topics.npz,metrics.json}
        results/ksweep/ksweep_<LANG>.csv
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

from analyze import PLACE_SEEDS, TRUTH_SEEDS, corr, find_topic, pca, remaining_ratio, segment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKERS = os.cpu_count()
N_TRAIN, ITERS, INFER_ITERS = 300_000, 500, 100
X12 = np.arange(1, 13, dtype=float)


def read_chunks(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def rw_sims(k, rng, n=1000, std=True):
    out = []
    for _ in range(n):
        W = np.cumsum(rng.standard_normal((12, k)), axis=0)
        if std:
            W = (W - W.mean(0)) / W.std(0)
        out.append(remaining_ratio(pca(W)[2]))
    return np.array(out)


def arc_metrics(dt, topics, lang, ref_scores, sims_cache, rng):
    _, C, L = segment(dt, 12)
    sh = C.sum(0) / L.sum(0)[:, None]
    K = sh.shape[1]
    rl, ra = corr(X12, sh), corr(-(X12 - 6.5) ** 2, sh)
    used, place, truth = set(), [], []
    for seeds in PLACE_SEEDS[lang]:
        k, _ = find_topic(topics, seeds, used); used.add(k); place.append(k)
    for seeds in TRUTH_SEEDS[lang]:
        k, _ = find_topic(topics, seeds, used); used.add(k); truth.append(k)
    sc, _, ev = pca(sh)
    m = dict(
        frac_lin09=float((np.abs(rl) > .9).mean()), frac_lin07=float((np.abs(rl) > .7).mean()),
        n_lin09=int((np.abs(rl) > .9).sum()), n_arc07=int((np.abs(ra) > .7).sum()),
        r_truth=float(np.corrcoef(X12, sh[:, truth].sum(1))[0, 1]),
        r_place=float(np.corrcoef(X12, sh[:, place].sum(1))[0, 1]),
        fig4_truth=[topics.label[k] for k in truth], fig4_place=[topics.label[k] for k in place],
        var2=float(ev[:2].sum() / ev.sum()),
        pc1_time=float(abs(np.corrcoef(X12, sc[:, 0])[0, 1])),
        pc2_arc=float(abs(np.corrcoef((X12 - 6.5) ** 2, sc[:, 1])[0, 1])))
    for mode in ("raw", "std"):
        kk = min(100, K)
        top = np.argsort(-sh.mean(0))[:kk]
        X = sh[:, top]
        if mode == "std":
            X = (X - X.mean(0)) / X.std(0)
        rr = remaining_ratio(pca(X)[2])
        key = (kk, mode)
        if key not in sims_cache:
            sims_cache[key] = rw_sims(kk, rng, std=(mode == "std"))
        s = sims_cache[key]
        m[f"z_pc2_{mode}"] = float((rr[1] - s[:, 1].mean()) / s[:, 1].std())
    if ref_scores is not None:
        m["agree_pc1_ref"] = float(abs(np.corrcoef(sc[:, 0], ref_scores[:, 0])[0, 1]))
        m["agree_pc2_ref"] = float(abs(np.corrcoef(sc[:, 1], ref_scores[:, 1])[0, 1]))
    return m


N_COH = 30_000   # fixed reference corpus for coherence (full-corpus c_v is single-threaded and O(K))


def coherence(top10, ref):
    """mean NPMI and C_v of the topics' top-10 words, estimated on the reference corpus."""
    targets = sorted({w for t in top10 for w in t})
    out = {}
    for c in ("c_npmi", "c_v"):
        coh = tp.coherence.Coherence(ref, coherence=c, top_n=10, targets=targets)
        out[c] = float(np.mean([coh.get_score(words=t) for t in top10]))
    return out


def main(lang, ks, seeds="42", n_eval=20000):
    lang = lang.upper()
    ks = [int(k) for k in ks.split(",")]
    seeds = [int(s) for s in str(seeds).split(",")]
    work = os.path.join(ROOT, "work", lang)
    out_root = os.path.join(ROOT, "work", "KSWEEP")
    res_dir = os.path.join(ROOT, "results", "ksweep")
    os.makedirs(out_root, exist_ok=True); os.makedirs(res_dir, exist_ok=True)

    ep = pd.read_csv(os.path.join(work, "episodes.csv"))
    eval_ep = set(ep.sample(int(n_eval), random_state=0).eid)
    rng = random.Random(42)
    train, eval_docs, seen = [], [], 0
    for d in read_chunks(os.path.join(work, "chunks.jsonl.gz")):
        if d["eid"] in eval_ep:
            eval_docs.append((d["eid"], d["i"], d["n"], d["t"]))
            continue
        if len(d["t"]) < 30:
            continue
        seen += 1
        if len(train) < N_TRAIN:
            train.append(d["t"])
        else:
            j = rng.randrange(seen)
            if j < N_TRAIN:
                train[j] = d["t"]
    print(f"{lang}: train {len(train)} chunks, eval {len(eval_docs)} chunks of {len(eval_ep)} episodes",
          flush=True)
    ref = tp.utils.Corpus()
    for j in random.Random(7).sample(range(len(train)), min(N_COH, len(train))):
        ref.add_doc(train[j])

    # reference trajectory: the main K=127 model on the same evaluation episodes
    main_dt = dict(np.load(os.path.join(work, "doc_topics.npz")))
    m = np.isin(main_dt["eid"], np.array(sorted(eval_ep)))
    _, Cr, Lr = segment({k: v[m] for k, v in main_dt.items()}, 12)
    ref_scores = pca(Cr.sum(0) / Lr.sum(0)[:, None])[0]
    del main_dt

    out_csv = os.path.join(res_dir, f"ksweep_{lang}.csv")
    rows = pd.read_csv(out_csv).to_dict("records") if os.path.exists(out_csv) else []
    for r in rows:      # (re)compute coherence of finished runs on the same reference corpus
        tdf = pd.read_csv(os.path.join(out_root, f"{lang}_K{r['K']}_s{r['seed']}", "topics.csv"))
        coh = coherence([t.split() for t in tdf.top10], ref)
        r["npmi"], r["c_v"] = coh["c_npmi"], coh["c_v"]
    done = {(r["K"], r["seed"]) for r in rows}
    sims_cache, nrng = {}, np.random.default_rng(0)
    for K in ks:
        for seed in seeds:
            if (K, seed) in done:
                continue
            t0 = time.time()
            mdl = tp.LDAModel(k=K, alpha=0.1, eta=0.01, seed=seed)
            mdl.optim_interval, mdl.burn_in = 20, 100
            for t in train:
                mdl.add_doc(t)
            mdl.train(ITERS, workers=WORKERS)
            t_train = time.time() - t0
            topics = pd.DataFrame([dict(topic=k, label=" ".join(w for w, _ in mdl.get_topic_words(k, top_n=7)),
                                        top10=" ".join(w for w, _ in mdl.get_topic_words(k, top_n=10)))
                                   for k in range(K)])
            coh = coherence([[w for w, _ in mdl.get_topic_words(k, top_n=10)] for k in range(K)], ref)
            top_words = [w for k in range(K) for w, _ in mdl.get_topic_words(k, top_n=10)]
            diversity = len(set(top_words)) / len(top_words)
            # held-out inference
            docs = [mdl.make_doc(t if t else ["\x01"]) for (_, _, _, t) in eval_docs]
            res, ll = mdl.infer(docs, iterations=INFER_ITERS, workers=WORKERS)
            lens = np.array([len(t) for (_, _, _, t) in eval_docs])
            ll = np.asarray(ll)
            ll_word = float(ll[lens > 0].sum() / lens.sum())
            counts = np.vstack([np.asarray(r, np.float32) * n for r, n in zip(res, lens)])
            dt = dict(counts=counts, eid=np.array([e for e, *_ in eval_docs]),
                      i=np.array([i for _, i, *_ in eval_docs], np.int16),
                      n=np.array([n for _, _, n, _ in eval_docs], np.int16), length=lens.astype(np.int32))
            d = os.path.join(out_root, f"{lang}_K{K}_s{seed}")
            os.makedirs(d, exist_ok=True)
            topics.to_csv(os.path.join(d, "topics.csv"), index=False)
            np.savez_compressed(os.path.join(d, "doc_topics.npz"), **dt)
            am = arc_metrics(dt, topics, lang, ref_scores, sims_cache, nrng)
            row = dict(lang=lang, K=K, seed=seed, npmi=coh["c_npmi"], c_v=coh["c_v"],
                       diversity=diversity, heldout_ll_word=ll_word, train_ll_word=float(mdl.ll_per_word),
                       train_sec=round(t_train), total_sec=round(time.time() - t0),
                       **{k: v for k, v in am.items() if not k.startswith("fig4")})
            json.dump(dict(row, fig4_truth=am["fig4_truth"], fig4_place=am["fig4_place"]),
                      open(os.path.join(d, "metrics.json"), "w"), ensure_ascii=False, indent=1)
            rows.append(row)
            pd.DataFrame(rows).sort_values(["K", "seed"]).to_csv(out_csv, index=False)
            print(f"K={K} seed={seed}: npmi={coh['c_npmi']:.3f} c_v={coh['c_v']:.3f} div={diversity:.2f} "
                  f"ll={ll_word:.3f} lin09={am['n_lin09']} r_truth={am['r_truth']:+.2f} "
                  f"r_place={am['r_place']:+.2f} var2={am['var2']:.3f} z_std={am['z_pc2_std']:.1f} "
                  f"agree={am.get('agree_pc1_ref', 0):.3f}/{am.get('agree_pc2_ref', 0):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            del mdl, docs, res


if __name__ == "__main__":
    main(*sys.argv[1:])
