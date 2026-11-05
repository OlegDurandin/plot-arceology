"""Plot arcs by genre and by decade (uses metadata from ororo.sqlite via meta.py).

* every group's 12-part arc is projected into the PCA space fitted on the whole corpus;
* per group: variance of first 2 PCs, PC2 z-score vs 1000 random walks, trends of the
  "place" and "truth" topic bundles from Fig. 4;
* robustness: main statistics on fiction only (no documentary / reality / talk shows).
Usage: python src/genres.py EN|RU
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import corr, pca, remaining_ratio, segment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_EPISODES = 1500
RNG_SIMS = 1000
SUB_N, SUB_REPS = 5000, 20     # equal-size subsamples: z depends on sample size (noise)


def rw_baseline(n_dims, rng):
    sims = []
    for _ in range(RNG_SIMS):
        W = np.cumsum(rng.standard_normal((12, n_dims)), axis=0)
        W = (W - W.mean(0)) / W.std(0)
        sims.append(remaining_ratio(pca(W)[2]))
    return np.array(sims)


def z_pc2(C, L, idx, top100, sims):
    sh = C[idx].sum(0) / L[idx].sum(0)[:, None]
    X = sh[:, top100]
    X = (X - X.mean(0)) / X.std(0)
    ev = pca(X)[2]
    rr = remaining_ratio(ev)
    return (rr[1] - sims[:, 1].mean()) / sims[:, 1].std(), ev[:2].sum() / ev.sum()


def group_stats(C, L, mask, top100, sims, truth, place):
    c, l = C[mask].sum(0), L[mask].sum(0)
    sh = c / l[:, None]
    X = sh[:, top100]
    X = (X - X.mean(0)) / X.std(0)
    ev = pca(X)[2]
    rr = remaining_ratio(ev)
    x = np.arange(1, 13, dtype=float)
    rng = np.random.default_rng(1)
    pool = np.flatnonzero(mask)
    sub = [z_pc2(C, L, rng.choice(pool, SUB_N, replace=False), top100, sims)
           for _ in range(SUB_REPS)] if len(pool) >= SUB_N else [(np.nan, np.nan)]
    sub = np.array(sub)
    return sh, dict(
        z_pc2_sub=round(float(np.mean(sub[:, 0])), 1),
        var_first2_sub=round(float(np.mean(sub[:, 1])), 3),
        episodes=int(mask.sum()),
        var_first2=round(float(ev[:2].sum() / ev.sum()), 3),
        z_pc1=round(float((rr[0] - sims[:, 0].mean()) / sims[:, 0].std()), 1),
        z_pc2=round(float((rr[1] - sims[:, 1].mean()) / sims[:, 1].std()), 1),
        r_truth_time=round(float(np.corrcoef(x, sh[:, truth].sum(1))[0, 1]), 3),
        r_place_time=round(float(np.corrcoef(x, sh[:, place].sum(1))[0, 1]), 3),
        n_lin_gt_09=int((np.abs(corr(x, sh)) > .9).sum()),
        n_arc_gt_09=int((np.abs(corr(-(x - 6.5) ** 2, sh)) > .9).sum()),
    )


def arcs_plot(groups, base_sc, Vt, mean, title, path, ncol=4):
    n = len(groups)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow), sharex=True, sharey=True)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (name, sh, st) in zip(axes.flat, groups):
        sc = (sh - mean) @ Vt[:2].T
        ax.plot(base_sc[:, 1], base_sc[:, 0], color="#cccccc", lw=1.2)
        ax.plot(sc[:, 1], sc[:, 0], color="C3", lw=1.2)
        for j in (0, 5, 11):
            ax.text(sc[j, 1], sc[j, 0], str(j + 1), fontsize=8, color="C3")
        ax.set_title(f"{name} (n={st['episodes']})\nz(PC2)={st['z_pc2']}, at n=5000: {st['z_pc2_sub']}", fontsize=9)
        ax.grid(alpha=.3); ax.tick_params(labelsize=6)
    fig.supxlabel("PC2 (corpus PCA)"); fig.supylabel("PC1 (corpus PCA)")
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main(lang):
    lang = lang.upper()
    work = os.path.join(ROOT, "work", lang)
    out = os.path.join(ROOT, "results", lang)
    meta = pd.read_csv(os.path.join(work, "episodes_meta.csv")).set_index("eid")
    dt = dict(np.load(os.path.join(work, "doc_topics.npz")))
    summ = json.load(open(os.path.join(out, "summary.json")))
    topics = pd.read_csv(os.path.join(work, "topics.csv"))
    lab = topics.label.tolist()
    truth = [lab.index(l) for l in summ["fig4_truth_topics"]]
    place = [lab.index(l) for l in summ["fig4_place_topics"]]

    E, C, L = segment(dt, 12)
    m = meta.loc[E]
    share = C.sum(0) / L.sum(0)[:, None]
    base_sc, Vt, _ = pca(share)
    mean = share.mean(0)
    top100 = np.argsort(-share.mean(0))[:100]
    sims = rw_baseline(100, np.random.default_rng(0))

    rows, garcs = [], []
    allmask = np.ones(len(E), bool)
    for name, mask in [("ALL", allmask), ("fiction only", m.fiction.fillna(False).values.astype(bool))]:
        sh, st = group_stats(C, L, mask, top100, sims, truth, place)
        rows.append(dict(group=name, kind="all", **st))

    genres = m.genres.fillna("").str.split("|")
    gcount = pd.Series([g for gs in genres for g in gs if g]).value_counts()
    for g in gcount[gcount >= MIN_EPISODES].index:
        mask = genres.map(lambda gs: g in gs).values
        sh, st = group_stats(C, L, mask, top100, sims, truth, place)
        rows.append(dict(group=g, kind="genre", **st))
        garcs.append((g, sh, st))
    arcs_plot(garcs, base_sc, Vt, mean,
              f"[{lang}] Aggregate plot arc by genre in corpus PC space (grey = whole corpus)",
              os.path.join(out, "genre_arcs.png"))

    darcs = []
    dec = (m.air_year // 10 * 10)
    for d in sorted(dec.dropna().unique()):
        mask = (dec == d).values
        if mask.sum() < MIN_EPISODES:
            continue
        sh, st = group_stats(C, L, mask, top100, sims, truth, place)
        rows.append(dict(group=f"{int(d)}s", kind="decade", **st))
        darcs.append((f"{int(d)}s", sh, st))
    arcs_plot(darcs, base_sc, Vt, mean,
              f"[{lang}] Aggregate plot arc by decade of airing", os.path.join(out, "decade_arcs.png"),
              ncol=len(darcs))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "genre_stats.csv"), index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
