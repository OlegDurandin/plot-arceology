"""BERTopic and embedding arcs (hypothesis H6: the plot arc is not an artefact of LDA).

Inputs: work/EMB/<SRC>.npz from embed_chunks.py (multilingual-e5-small, one shared space for
TV_RU, TV_EN, TALES, PROZA, FICBOOK) and the lemmatised chunk tokens (for topic words only).

1. Embedding arcs (no clustering at all): the mean chunk embedding of each 12th of the text
   (word-weighted) -> PCA: variance of 2 PCs, |r(PC1, time)|, |r(PC2, middle)|, z(PC2) vs random
   walks; and a shuffle null (chunk order permuted inside texts). Equal n = 2000 texts per source.
2. BERTopic: UMAP(5d, cosine) + HDBSCAN fitted on a balanced sample; c-TF-IDF over lemmas.
   Variants: HDBSCAN ("auto" number of topics) and KMeans with 127 / 64 / 32 topics on the same
   UMAP coordinates (comparable to the LDA K sweep).
   Every chunk is assigned to the nearest topic centroid (cosine), i.e. no outliers.
   Output in the LDA format -> work/BERT<variant>_{TV,TVEN,TALES,PROZA,FICBOOK}/ so that
   analyze.py and compare_prose.py run unchanged.
Usage: python src/bertopic_run.py
"""
import gzip
import json
import os
import time

import numpy as np
import pandas as pd

from analyze import pca, remaining_ratio, segment
from compare_prose import shuffled
from stopwords import STOP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = os.path.join(ROOT, "work", "EMB")
SRCS = {"TV_RU": ("RU", "TV"), "TV_EN": ("EN", "TVEN"), "TALES": ("TALES", "TALES"),
        "PROZA": ("PROZA", "PROZA"), "FICBOOK": ("FICBOOK", "FICBOOK")}
FIT_N = {"TV_RU": 40000, "TV_EN": 40000, "TALES": 20000, "PROZA": 30000, "FICBOOK": 30000}
X12 = np.arange(1, 13, dtype=float)
N_EQ, PERMS = 2000, 100
# RESID=1: cluster within-text residuals (chunk embedding minus the mean embedding of its text), so that
# topics describe what happens at this point of the text rather than which text / show / fandom it is
RESID = os.environ.get("RESID", "0") == "1"


def residualize(e):
    X = e["emb"].astype(np.float32)
    eid = e["eid"]
    uniq, inv = np.unique(eid, return_inverse=True)
    w = e["words"].astype(np.float32)[:, None]
    mean = np.zeros((len(uniq), X.shape[1]), np.float32)
    np.add.at(mean, inv, X * w)
    mean /= np.bincount(inv, weights=w[:, 0])[:, None]
    R = X - mean[inv]
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
    out = dict(e)
    out["emb"] = R
    return out


def load_emb(src):
    z = np.load(os.path.join(EMB, f"{src}.npz"))
    return {k: z[k] for k in z.files}


def lemma_docs(src, keys):
    """'eid#i' -> space-joined lemmas (stop words / names removed)."""
    wdir, _ = SRCS[src]
    stop_ru, stop_en = STOP["RU"], STOP["EN"]
    out = {}
    if src.startswith("TV_"):
        files = [os.path.join(ROOT, "work", wdir, "chunks.jsonl.gz")]   # vocabulary-filtered
    else:
        d = os.path.join(ROOT, "work", wdir, "raw")
        files = [os.path.join(d, f) for f in sorted(os.listdir(d))]
    for fn in files:
        with gzip.open(fn, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                k = f"{d['eid']}#{d['i']}"
                if k in keys:
                    out[k] = " ".join(w for w in d["t"] if not w.startswith("\x00")
                                      and w not in stop_ru and w not in stop_en and len(w) > 2)
    return out


def emb_arc_metrics(e, rng, sims):
    """e: dict(emb, eid, i, n, words) -> metrics of the mean-embedding arc."""
    dt = dict(eid=e["eid"], i=e["i"], n=e["n"], length=e["words"].astype(np.float32),
              counts=e["emb"].astype(np.float32) * e["words"][:, None])
    E, C, L = segment(dt, 12)

    def m(idx, C_, L_):
        A = C_[idx].sum(0) / L_[idx].sum(0)[:, None]          # [12, 384] mean embedding per 12th
        sc, _, ev = pca(A)
        X = (A - A.mean(0)) / (A.std(0) + 1e-12)
        rr = remaining_ratio(pca(X)[2])
        return dict(var2=float(ev[:2].sum() / ev.sum()),
                    pc1_time=float(abs(np.corrcoef(X12, sc[:, 0])[0, 1])),
                    pc2_arc=float(abs(np.corrcoef((X12 - 6.5) ** 2, sc[:, 1])[0, 1])),
                    z_pc2=float((rr[1] - sims[:, 1].mean()) / sims[:, 1].std())), sc

    full, sc_full = m(np.arange(len(E)), C, L)
    reps = [m(rng.choice(len(E), min(N_EQ, len(E)), replace=False), C, L)[0] for _ in range(10)]
    eq = pd.DataFrame(reps).mean().to_dict()
    pick = set(rng.choice(E, min(N_EQ, len(E)), replace=False))
    msk = np.isin(dt["eid"], np.array(sorted(pick)))
    sub = {k: v[msk] for k, v in dt.items()}
    _, Cs, Ls = segment(sub, 12)
    obs = m(np.arange(Cs.shape[0]), Cs, Ls)[0]
    nulls = []
    for _ in range(PERMS):
        _, Cp, Lp = segment(shuffled(sub, rng), 12)
        nulls.append(m(np.arange(Cp.shape[0]), Cp, Lp)[0])
    nd = pd.DataFrame(nulls)
    null = {f"null_{k}": float(nd[k].mean()) for k in obs}
    null.update({f"zshuf_{k}": float((obs[k] - nd[k].mean()) / (nd[k].std() + 1e-12)) for k in obs})
    return dict(texts=len(E), **{f"all_{k}": v for k, v in full.items()},
                **{f"n{N_EQ}_{k}": v for k, v in eq.items()}, **null), sc_full


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(ROOT, "results", "bertopic")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    sims = []
    for _ in range(1000):
        W = np.cumsum(rng.standard_normal((12, 384)), axis=0)
        W = (W - W.mean(0)) / W.std(0)
        sims.append(remaining_ratio(pca(W)[2]))
    sims = np.array(sims)

    embs = {s: load_emb(s) for s in SRCS if os.path.exists(os.path.join(EMB, f"{s}.npz"))}
    print("sources:", {s: len(e["eid"]) for s, e in embs.items()}, flush=True)
    if RESID:
        embs = {s: residualize(e) for s, e in embs.items()}
        print("using within-text residual embeddings", flush=True)

    # ---- 1. embedding arcs ---------------------------------------------------------------
    rows, arcs = [], {}
    for s, e in ({} if RESID else embs).items():
        mt, sc = emb_arc_metrics(e, rng, sims)
        rows.append(dict(source=s, **mt)); arcs[s] = sc
        print("emb-arc", s, {k: round(v, 3) for k, v in mt.items()}, flush=True)
    if not RESID:
        pd.DataFrame(rows).to_csv(os.path.join(out, "embedding_arcs.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    for s, sc in arcs.items():
        sg1 = np.sign(np.corrcoef(X12, sc[:, 0])[0, 1]) or 1
        sg2 = np.sign(np.corrcoef((X12 - 6.5) ** 2, sc[:, 1])[0, 1]) or 1
        p1, p2 = sc[:, 0] * sg1 / np.abs(sc[:, 0]).max(), sc[:, 1] * sg2 / np.abs(sc[:, 1]).max()
        ax.plot(p2, p1, label=s)
        for j in (0, 11):
            ax.text(p2[j], p1[j], str(j + 1), fontsize=8)
    ax.set_xlabel("PC2 (normalised)"); ax.set_ylabel("PC1 (normalised)"); ax.grid(alpha=.3); ax.legend()
    ax.set_title("Mean-embedding arc (multilingual-e5), 12 parts, per source")
    if not RESID:
        fig.tight_layout(); fig.savefig(os.path.join(out, "embedding_arcs.png"), dpi=130)
    plt.close(fig)

    # ---- 2. BERTopic ------------------------------------------------------------------------
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    t0 = time.time()
    fit_idx = {s: np.sort(rng.choice(len(e["eid"]), min(FIT_N[s], len(e["eid"])), replace=False))
               for s, e in embs.items()}
    keys_all = {s: np.char.add(np.char.add(e["eid"].astype(str), "#"), e["i"].astype(str))
                for s, e in embs.items()}
    lem = {}
    for s in embs:
        lem.update({f"{s}|{k}": v for k, v in lemma_docs(s, set(keys_all[s])).items()})
    fit_docs, fit_emb = [], []
    for s, idx in fit_idx.items():
        fit_docs += [lem.get(f"{s}|{k}", "") for k in keys_all[s][idx]]
        fit_emb.append(embs[s]["emb"][idx].astype(np.float32))
    fit_emb = np.vstack(fit_emb)
    print(f"BERTopic fit on {len(fit_docs)} chunks", flush=True)
    # UMAP once; every variant clusters the same 5-d coordinates
    from bertopic.dimensionality import BaseDimensionalityReduction
    from sklearn.cluster import KMeans
    red = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
               random_state=42, low_memory=True).fit_transform(fit_emb)
    print(f"UMAP done ({time.time() - t0:.0f}s)", flush=True)

    def fit_variant(variant):
        cl = (HDBSCAN(min_cluster_size=150, min_samples=10, metric="euclidean",
                      cluster_selection_method="eom", prediction_data=False)
              if variant == "auto" else KMeans(n_clusters=variant, n_init=4, random_state=42))
        m_ = BERTopic(umap_model=BaseDimensionalityReduction(), hdbscan_model=cl,
                      vectorizer_model=CountVectorizer(min_df=2),   # fitted on one document per topic
                      language="multilingual",   # "english" would strip all non-Latin characters
                      calculate_probabilities=False, verbose=False)
        m_.fit(fit_docs, red)
        return m_

    summary = []
    for variant in ([127, 64, 32] if RESID else ["auto", 127, 64, 32]):
        m_ = fit_variant(variant)
        if variant == "auto":
            tt = np.array(m_.topics_)
            print(f"HDBSCAN found {len(set(tt) - {-1})} topics, outliers {np.mean(tt == -1):.1%} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        t_fit = np.array(m_.topics_)
        tids = sorted(t for t in set(t_fit) if t != -1)
        K = len(tids)
        cent = np.vstack([fit_emb[t_fit == t].mean(0) for t in tids])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True)
        labels = []
        for t in tids:
            words = [w for w, _ in (m_.get_topic(t) or [])][:10]
            labels.append(dict(topic=len(labels), label=" ".join(words[:7]), top10=" ".join(words)))
        tag = f"BERTres{variant}" if RESID else f"BERT{variant}"
        for s, e in embs.items():
            X = e["emb"].astype(np.float32)
            assign = np.concatenate([np.argmax(X[a:a + 50000] @ cent.T, axis=1)
                                     for a in range(0, len(X), 50000)])
            counts = np.zeros((len(X), K), np.float32)
            counts[np.arange(len(X)), assign] = e["words"]
            d = os.path.join(ROOT, "work", f"{tag}_{SRCS[s][1]}")
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, "doc_topics.npz"), counts=counts, eid=e["eid"],
                                i=e["i"], n=e["n"], length=e["words"].astype(np.int32))
            pd.DataFrame(labels).to_csv(os.path.join(d, "topics.csv"), index=False)
            wdir = SRCS[s][0] if not s == "PROZA" else "PROJ_PROZA"
            ep = pd.read_csv(os.path.join(ROOT, "work", wdir, "episodes.csv"))
            ep[ep.eid.isin(set(e["eid"]))].to_csv(os.path.join(d, "episodes.csv"), index=False)
        summary.append(dict(variant=str(variant), topics=K))
        print(f"{tag}: {K} topics written", flush=True)
    pd.DataFrame(summary).to_csv(os.path.join(out, "variants_resid.csv" if RESID else "variants.csv"),
                                 index=False)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
