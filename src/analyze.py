"""Stage 3: reproduce the figures of Schmidt, "Plot Arceology".

Usage: python src/analyze.py EN|RU
Writes figures to results/<lang>/ and a summary json.
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Shows plotted in Fig. 5 of the paper -> closest equivalents in this corpus
# (CSI -> CSI Miami; reality show Survivor -> The Traitors US; The Fugitive (1963) is absent,
#  the 2020 Quibi remake has ~10-min episodes -> Prison Break, also a show about fugitives)
FIG5_SHOWS = ["CSI Miami", "Grey's Anatomy", "NCIS", "Stargate SG-1",
              "The Traitors US", "Prison Break"]
STREAM_SHOW = "Law and Order US"

# Seed words to locate the six topics of Fig. 4 (place vs. truth/apology)
PLACE_SEEDS = {
    "EN": [["house", "apartment", "room", "live", "place"],
           ["school", "class", "teacher", "college", "kids"],
           ["work", "job", "office", "boss", "company"]],
    "RU": [["дом", "квартира", "комната", "жить", "место"],
           ["школа", "класс", "учитель", "колледж", "урок"],
           ["работа", "работать", "офис", "босс", "компания"]],
}
TRUTH_SEEDS = {
    "EN": [["truth", "lie", "believe", "true", "lying"],
           ["sorry", "fault", "wrong", "mistake", "forgive"],
           ["leave", "stay", "away", "promise", "please", "care"]],   # the paper's third topic
    "RU": [["правда", "врать", "ложь", "верить", "лгать"],
           ["прости", "простить", "вина", "виноватый", "извинить", "ошибка"],
           ["любовь", "сердце", "любить", "чувствовать", "счастливый", "обещать", "простить"]],
}


def overlap_matrix(n, m):
    """W[j, i] = fraction of chunk i (of n equal parts) falling in segment j (of m)."""
    W = np.zeros((m, n))
    for i in range(n):
        a, b = i / n, (i + 1) / n
        for j in range(m):
            c, d = j / m, (j + 1) / m
            ov = max(0.0, min(b, d) - max(a, c))
            W[j, i] = ov * n
    return W


def segment(dt, m):
    """Aggregate chunk topic-counts into m equal segments of screen time per episode.
    Returns (episode ids, counts[E, m, K], tokens[E, m])."""
    counts, length, eid, n, i = dt["counts"], dt["length"], dt["eid"], dt["n"], dt["i"]
    order = np.lexsort((i, eid))
    counts, length, eid, n, i = counts[order], length[order], eid[order], n[order], i[order]
    starts = np.flatnonzero(np.r_[True, eid[1:] != eid[:-1]])
    ends = np.r_[starts[1:], len(eid)]
    K = counts.shape[1]
    E = len(starts)
    C = np.zeros((E, m, K), np.float32)
    L = np.zeros((E, m), np.float32)
    cache = {}
    for e, (s, t) in enumerate(zip(starts, ends)):
        nn = int(n[s])
        if t - s != nn:
            continue
        W = cache.get(nn)
        if W is None:
            W = cache[nn] = overlap_matrix(nn, m)
        C[e] = W @ counts[s:t]
        L[e] = W @ length[s:t]
    return eid[starts], C, L


def corr(x, y):
    x = x - x.mean(); y = y - y.mean(axis=0)
    return (x @ y) / np.sqrt((x @ x) * (y * y).sum(axis=0))


def small_multiples(share, topics, labels, title, path, xlab="12th"):
    fig, axes = plt.subplots(4, 3, figsize=(12, 11))
    x = np.arange(1, share.shape[0] + 1)
    for ax, k in zip(axes.flat, topics):
        ax.plot(x, share[:, k] * 100, color="black", lw=1.3)
        ax.set_title(labels[k][:48], fontsize=8.5, backgroundcolor="#dddddd")
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f%%"))
        ax.grid(alpha=.3)
        ax.set_xticks([1, 3, 6, 9, 12])
    fig.supxlabel(xlab); fig.supylabel("Percent of all non-stopwords in chunk assigned to topic")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def find_topic(topics_df, seeds, exclude):
    """Topic whose top-10 words overlap most with the seed list (rank-weighted)."""
    best, best_s = None, -1
    for _, r in topics_df.iterrows():
        if r.topic in exclude:
            continue
        words = r.top10.split()
        s = sum((10 - words.index(w)) for w in seeds if w in words)
        if s > best_s:
            best, best_s = r.topic, s
    return int(best), best_s


def pca(X):
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return U * S, Vt, S ** 2


def remaining_ratio(ev):
    ev = ev[:11]
    rem = np.cumsum(ev[::-1])[::-1]
    return ev / rem


def main(lang):
    lang = lang.upper()
    work = os.path.join(ROOT, "work", lang)
    out = os.path.join(ROOT, "results", lang)
    os.makedirs(out, exist_ok=True)
    topics = pd.read_csv(os.path.join(work, "topics.csv"))
    labels = topics.label.tolist()
    K = len(labels)
    dt = dict(np.load(os.path.join(work, "doc_topics.npz")))
    eps = pd.read_csv(os.path.join(work, "episodes.csv")).set_index("eid")
    summary = {"lang": lang, "episodes": int(len(eps)), "series": int(eps.series.nunique()),
               "chunks": int(len(dt["eid"])), "content_tokens": int(dt["length"].sum())}

    E12, C12, L12 = segment(dt, 12)
    E6, C6, L6 = segment(dt, 6)
    share12 = C12.sum(0) / L12.sum(0)[:, None]          # [12, K] corpus-level share
    share6 = C6.sum(0) / L6.sum(0)[:, None]
    np.save(os.path.join(work, "share12.npy"), share12)

    # ---------------- Fig 1 / Fig 2: linear and arcing trends -----------------------
    x = np.arange(1, 13, dtype=float)
    r_lin = corr(x, share12)
    r_arc = corr(-(x - 6.5) ** 2, share12)             # +: peak in the middle
    tr = topics.assign(r_linear=r_lin, r_arc=r_arc,
                       first12=share12[0], last12=share12[-1],
                       mean_share=share12.mean(0))
    tr.to_csv(os.path.join(out, "topic_trends.csv"), index=False)
    lin_top = np.argsort(-np.abs(r_lin))[:12]
    arc_top = np.argsort(-np.abs(r_arc))[:12]
    small_multiples(share12, lin_top, labels,
                    f"[{lang}] The 12 topics showing the strongest linear trend through a script",
                    os.path.join(out, "fig1_linear.png"))
    small_multiples(share12, arc_top, labels,
                    f"[{lang}] The 12 topics showing the strongest arced trend through a script",
                    os.path.join(out, "fig2_arc.png"))
    summary["linear_top"] = [dict(topic=int(k), label=labels[k], r=round(float(r_lin[k]), 3))
                             for k in lin_top]
    summary["arc_top"] = [dict(topic=int(k), label=labels[k], r=round(float(r_arc[k]), 3))
                          for k in arc_top]
    summary["n_topics_abs_r_lin_gt_0.9"] = int((np.abs(r_lin) > .9).sum())
    summary["n_topics_abs_r_arc_gt_0.9"] = int((np.abs(r_arc) > .9).sum())

    # ---------------- Fig 3: streamgraph of Law and Order by sixth -------------------
    ser6 = eps.loc[E6, "series"].values
    sel = ser6 == STREAM_SHOW
    if sel.any():
        c = C6[sel].sum(0)                                # [6, K] total words per topic
        top = np.argsort(-c.sum(0))[:10][::-1]
        fig, ax = plt.subplots(figsize=(11, 5))
        xs = np.arange(1, 7)
        ax.stackplot(xs, c[:, top].T, labels=[labels[k][:55] for k in top],
                     baseline="wiggle", colors=plt.cm.tab20(np.linspace(0, 1, 10)))
        h, l = ax.get_legend_handles_labels()
        ax.legend(h[::-1], l[::-1], fontsize=8, loc="center left", bbox_to_anchor=(1, .5))
        ax.set_xlabel("sixth of episode")
        ax.set_title(f"[{lang}] Top topics in {STREAM_SHOW} ({sel.sum()} episodes), "
                     "total words in topic per sixth")
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_law_and_order.png"), dpi=130)
        plt.close(fig)
        summary["law_and_order_episodes"] = int(sel.sum())

    # ---------------- Fig 4: path through a 2-D space of 3+3 topics ------------------
    used, place, truth = set(), [], []
    seed_lang = "EN" if lang == "EN" else "RU"   # prose / joint models are Russian
    for seeds in PLACE_SEEDS[seed_lang]:
        k, _ = find_topic(topics, seeds, used); used.add(k); place.append(k)
    for seeds in TRUTH_SEEDS[seed_lang]:
        k, _ = find_topic(topics, seeds, used); used.add(k); truth.append(k)
    summary["fig4_place_topics"] = [labels[k] for k in place]
    summary["fig4_truth_topics"] = [labels[k] for k in truth]
    xs, ys = share12[:, truth].sum(1), share12[:, place].sum(1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(xs * 100, ys * 100, color="grey", lw=.8)
    for j in range(12):
        ax.text(xs[j] * 100, ys[j] * 100, str(j + 1), fontsize=11, ha="center", va="center")
    ax.set_xlabel("Share of vocabulary from 3 truth & reconciliation topics, %")
    ax.set_ylabel("Share of vocabulary from 3 location topics, %")
    ax.set_title(f"[{lang}] Path of aggregate scripts, by twelfth,\nthrough a space defined by 6 topics")
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_path6.png"), dpi=130); plt.close(fig)
    summary["fig4_corr_x_time"] = round(float(np.corrcoef(x, xs)[0, 1]), 3)
    summary["fig4_corr_y_time"] = round(float(np.corrcoef(x, ys)[0, 1]), 3)

    # ---------------- Fig 5: six individual shows in that space ---------------------
    has_shows = any((ser6 == s_).any() for s_ in FIG5_SHOWS)
    fig, axes = plt.subplots(3, 2, figsize=(10, 11))
    summary["fig5"] = {}
    for ax, show in zip(axes.flat, FIG5_SHOWS):
        m = ser6 == show
        if not m.any():
            ax.set_title(f"{show}: not in corpus"); continue
        c, L = C6[m].sum(0), L6[m].sum(0)
        sx, sy = c[:, truth].sum(1) / L, c[:, place].sum(1) / L
        ax.plot(sx * 100, sy * 100, color="grey", lw=.8)
        for j in range(6):
            ax.text(sx[j] * 100, sy[j] * 100, str(j + 1), fontsize=11, ha="center", va="center")
        ax.set_title(f"{show} (n={m.sum()})", fontsize=10, backgroundcolor="#dddddd")
        ax.grid(alpha=.3); ax.tick_params(labelsize=8)
        summary["fig5"][show] = dict(n=int(m.sum()),
                                     r_truth_time=round(float(np.corrcoef(np.arange(6), sx)[0, 1]), 3),
                                     r_place_time=round(float(np.corrcoef(np.arange(6), sy)[0, 1]), 3))
    fig.supxlabel("truth & reconciliation topics, %"); fig.supylabel("location topics, %")
    fig.suptitle(f"[{lang}] Aggregate path of several individual shows, by 6th")
    if has_shows:
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig5_shows.png"), dpi=130)
    plt.close(fig)

    # ---------------- Fig 6: PCA of the 12 x K plot-arc matrix -----------------------
    scores, Vt, ev = pca(share12)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.plot(scores[:, 1], scores[:, 0], color="#e07070", lw=1)
    for j in range(12):
        ax.text(scores[j, 1], scores[j, 0], f"part {j + 1}", fontsize=9)
    load = Vt[:2].T                                       # [K, 2]
    imp = np.argsort(-np.hypot(load[:, 0], load[:, 1]))[:15]
    scale = np.abs(scores[:, :2]).max() / np.abs(load[imp]).max()
    cols = plt.cm.tab20(np.linspace(0, 1, 15))
    for c, k in zip(cols, imp):
        ax.plot([0, load[k, 1] * scale], [0, load[k, 0] * scale], color=c, lw=1)
        ax.text(load[k, 1] * scale, load[k, 0] * scale, " ".join(labels[k].split()[:4]),
                color=c, fontsize=7.5)
    ax.set_xlabel("PC2"); ax.set_ylabel("PC1"); ax.grid(alpha=.3)
    ax.set_title(f"[{lang}] Plot arcs with the 15 most important topics to the Principal Components")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig6_pca.png"), dpi=130); plt.close(fig)
    summary["pca_var_explained_first2_allK"] = round(float(ev[:2].sum() / ev.sum()), 4)
    summary["pc1_corr_time"] = round(float(abs(np.corrcoef(x, scores[:, 0])[0, 1])), 3)
    summary["pc2_corr_arc"] = round(float(abs(np.corrcoef((x - 6.5) ** 2, scores[:, 1])[0, 1])), 3)

    # ---------------- Fig 7: variance explained vs. random walks ---------------------
    # 100 most frequent topics with non-zero variance over the 12 parts (BERTopic topics can be
    # absent from a source, e.g. English-only topics in Russian texts)
    top100 = [k for k in np.argsort(-share12.mean(0)) if share12[:, k].std() > 0][:100]
    kdim = len(top100)
    rng = np.random.default_rng(0)
    res = {}
    for mode in ["raw", "standardized"]:
        def prep(X):
            if mode == "standardized":
                X = (X - X.mean(0)) / X.std(0)
            return X
        X = prep(share12[:, top100])
        _, _, ev_r = pca(X)
        rr = remaining_ratio(ev_r)
        sims, f2 = [], []
        for _ in range(1000):
            W = np.cumsum(rng.standard_normal((12, kdim)), axis=0)
            _, _, ev_w = pca(prep(W))
            sims.append(remaining_ratio(ev_w))
            f2.append(ev_w[:2].sum() / ev_w.sum())
        sims = np.array(sims)
        z = (rr - sims.mean(0)) / sims.std(0)
        res[mode] = dict(real=rr, sims=sims, z=z,
                         first2=float(ev_r[:2].sum() / ev_r.sum()),
                         rw_first2=float(np.mean(f2)))
        summary[f"fig7_{mode}"] = dict(
            var_explained_first2=round(res[mode]["first2"], 4),
            rw_var_explained_first2=round(res[mode]["rw_first2"], 4),
            remaining_ratio_real=[round(float(v), 3) for v in rr[:6]],
            remaining_ratio_rw_mean=[round(float(v), 3) for v in sims.mean(0)[:6]],
            z_scores=[round(float(v), 1) for v in z[:6]],
            pc_beats_all_1000_rw=[bool(rr[k] > sims[:, k].max()) for k in range(min(6, len(rr)))])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, mode in zip(axes, res):
        r = res[mode]
        nc = min(11, len(r["real"]), r["sims"].shape[1])      # rank can be < 11 for few topics
        ax.boxplot([r["sims"][:, k] for k in range(nc)], positions=range(1, nc + 1), widths=.6,
                   flierprops=dict(markersize=2))
        ax.scatter(range(1, nc + 1), r["real"][:nc], marker="^", color="red", s=60, zorder=3)
        ax.set_xlabel("component"); ax.set_ylabel("share of remaining variance explained")
        ax.set_title(f"{mode} topics: first 2 PCs explain {r['first2']:.0%} of variance\n"
                     f"z(PC1)={r['z'][0]:.1f}, z(PC2)={r['z'][1]:.1f} vs 1000 random walks")
        ax.grid(alpha=.3)
    fig.suptitle(f"[{lang}] Remaining variance explained by each PC (red) vs random walks (boxplots)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig7_random_walk.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
