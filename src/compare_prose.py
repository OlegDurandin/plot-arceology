"""Prose pipeline, stage 3: compare TV subtitles, fairy tales and proza.ru stories
in ONE topic space (default: projection onto the Russian TV model).

* equal-size subsamples (N texts, REPS repetitions): trend counts, 6-topic path, PCA variance, z(PC2)
* shuffle null: chunk order permuted inside each text (keeps each text's vocabulary, destroys order)
* all sources' 12-part arcs in the TV PCA space; dialogue-share arc; proza length strata
Usage: python src/compare_prose.py [prefix=PROJ] [tv_dir=RU]
   prefix PROJ -> work/PROJ_TALES, work/PROJ_PROZA ; prefix JOINT -> work/JOINT_{TV,TALES,PROZA}
"""
import gzip
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
N, REPS, PERMS = 2000, 20, 200
N_CTRL = 600                      # texts per source in the equal-chunk-count control
X12 = np.arange(1, 13, dtype=float)


def load(d):
    w = os.path.join(ROOT, "work", d)
    dt = dict(np.load(os.path.join(w, "doc_topics.npz")))
    ep = pd.read_csv(os.path.join(w, "episodes.csv"))
    return dt, ep


def subset(dt, eids):
    m = np.isin(dt["eid"], np.asarray(list(eids)))
    return {k: v[m] for k, v in dt.items()}


def shuffled(dt, rng):
    """permute chunk positions inside every text"""
    eid = dt["eid"]
    order = np.lexsort((rng.random(len(eid)), eid))
    new_i = np.empty_like(dt["i"])
    starts = np.r_[0, np.flatnonzero(eid[order][1:] != eid[order][:-1]) + 1]
    pos = np.arange(len(eid)) - np.repeat(starts, np.diff(np.r_[starts, len(eid)]))
    new_i[order] = pos
    out = dict(dt)
    out["i"] = new_i
    return out


def rw_sims(rng, n=1000, k=100):
    s = []
    for _ in range(n):
        W = np.cumsum(rng.standard_normal((12, k)), axis=0)
        s.append(remaining_ratio(pca(W)[2]))
    return np.array(s)


def metrics(C, L, truth, place, top, sims):
    sh = C.sum(0) / L.sum(0)[:, None]
    rl, ra = corr(X12, sh), corr(-(X12 - 6.5) ** 2, sh)
    ev = pca(sh[:, top])[2]
    rr = remaining_ratio(ev)
    return dict(n_lin09=int((np.abs(rl) > .9).sum()), n_lin07=int((np.abs(rl) > .7).sum()),
                n_arc07=int((np.abs(ra) > .7).sum()),
                r_truth=float(np.corrcoef(X12, sh[:, truth].sum(1))[0, 1]),
                r_place=float(np.corrcoef(X12, sh[:, place].sum(1))[0, 1]),
                var2=float(ev[:2].sum() / ev.sum()),
                z_pc2=float((rr[1] - sims[:, 1].mean()) / sims[:, 1].std())), sh


def dialogue_arc(src_dir, eids):
    """share of dialogue words by 12th (from the raw chunk files)"""
    rows = []
    raw = os.path.join(ROOT, "work", src_dir, "raw")
    for fn in sorted(os.listdir(raw)):
        with gzip.open(os.path.join(raw, fn), "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["eid"] in eids:
                    rows.append((d["eid"], d["i"], d["n"], d["w"], d["d"] * d["w"]))
    df = pd.DataFrame(rows, columns=["eid", "i", "n", "w", "dw"])
    dt = dict(eid=df.eid.values, i=df.i.values, n=df.n.values,
              counts=df[["dw"]].values.astype(np.float32), length=df.w.values.astype(np.float32))
    _, C, L = segment(dt, 12)
    return C.sum(0)[:, 0] / L.sum(0)


def main(prefix="PROJ", tv_dir="RU"):
    out = os.path.join(ROOT, "results", f"prose_{prefix.lower()}")
    os.makedirs(out, exist_ok=True)
    srcs = {"TV (subtitles RU)": tv_dir if prefix == "PROJ" else f"{prefix}_TV",
            "Fairy tales": f"{prefix}_TALES", "proza.ru": f"{prefix}_PROZA",
            "ficbook.net": f"{prefix}_FICBOOK"}
    srcs = {k: v for k, v in srcs.items()
            if os.path.exists(os.path.join(ROOT, "work", v, "doc_topics.npz"))}
    summ_tv = json.load(open(os.path.join(ROOT, "results", srcs["TV (subtitles RU)"], "summary.json")))
    topics = pd.read_csv(os.path.join(ROOT, "work", srcs["TV (subtitles RU)"], "topics.csv"))
    lab = topics.label.tolist()
    truth = [lab.index(l) for l in summ_tv["fig4_truth_topics"]]
    place = [lab.index(l) for l in summ_tv["fig4_place_topics"]]
    rng = np.random.default_rng(0)
    sims = rw_sims(rng)

    data, rows, arcs, null_rows = {}, [], {}, []
    for name, d in srcs.items():
        dt, ep = load(d)
        data[name] = (dt, ep)
        E, C, L = segment(dt, 12)
        full = C.sum(0) / L.sum(0)[:, None]
        arcs[name] = full
        top = np.argsort(-full.mean(0))[:100]
        m_full, _ = metrics(C, L, truth, place, top, sims)
        rows.append(dict(source=name, sample="all", texts=len(E), **m_full))
        reps = []
        for r in range(REPS):
            idx = rng.choice(len(E), min(N, len(E)), replace=False)
            reps.append(metrics(C[idx], L[idx], truth, place, top, sims)[0])
        rows.append(dict(source=name, sample=f"n={N} (mean of {REPS})", texts=min(N, len(E)),
                         **pd.DataFrame(reps).mean().to_dict()))
        # control for chunking resolution: only texts with exactly 6 / 12 chunks, equal n
        n_of = pd.Series(dt["n"], index=dt["eid"]).groupby(level=0).first().reindex(E).values
        for nc in (6, 12):
            pool = np.flatnonzero(n_of == nc)
            if len(pool) < N_CTRL:
                continue
            reps = [metrics(C[i_], L[i_], truth, place, top, sims)[0]
                    for i_ in (rng.choice(pool, N_CTRL, replace=False) for _ in range(REPS))]
            rows.append(dict(source=name, sample=f"{nc} chunks only, n={N_CTRL}", texts=N_CTRL,
                             **pd.DataFrame(reps).mean().to_dict()))
        # shuffle null on one equal-size subsample
        pick = set(rng.choice(E, min(N, len(E)), replace=False))
        sub = subset(dt, pick)
        _, Cs, Ls = segment(sub, 12)
        obs, _ = metrics(Cs, Ls, truth, place, top, sims)
        nulls = []
        for p in range(PERMS):
            _, Cp, Lp = segment(shuffled(sub, rng), 12)
            nulls.append(metrics(Cp, Lp, truth, place, top, sims)[0])
        nd = pd.DataFrame(nulls)
        for k in ["n_lin09", "n_lin07", "n_arc07", "r_truth", "r_place", "var2", "z_pc2"]:
            o, mu, sd = obs[k], nd[k].mean(), nd[k].std()
            null_rows.append(dict(source=name, metric=k, observed=o, null_mean=mu, null_sd=sd,
                                  z=(o - mu) / sd if sd > 0 else np.nan,
                                  # one-sided, in the direction of the observed deviation
                                  p=float(((np.sign(o - mu) * (nd[k] - mu) >= abs(o - mu)).sum() + 1)
                                          / (PERMS + 1))))
        print(name, "done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "comparison.csv"), index=False)
    nres = pd.DataFrame(null_rows)
    nres.to_csv(os.path.join(out, "shuffle_null.csv"), index=False)

    # proza length strata
    dt, ep = data["proza.ru"]
    E, C, L = segment(dt, 12)
    nw = ep.set_index("eid").loc[E, "n_words_clean"].values
    strata = []
    for lo, hi in [(0, 1500), (1500, 3000), (3000, 99999)]:
        m = (nw >= lo) & (nw < hi)
        idx = rng.choice(np.flatnonzero(m), min(N, m.sum()), replace=False)
        top = np.argsort(-arcs["proza.ru"].mean(0))[:100]
        strata.append(dict(stratum=f"{lo}-{hi} words", texts=len(idx),
                           **metrics(C[idx], L[idx], truth, place, top, sims)[0]))
    pd.DataFrame(strata).to_csv(os.path.join(out, "proza_length_strata.csv"), index=False)

    # figure: arcs of all sources in the TV PCA space + 6-topic paths + dialogue arc
    sc_tv, Vt, _ = pca(arcs["TV (subtitles RU)"])
    mean = arcs["TV (subtitles RU)"].mean(0)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    cols = {"TV (subtitles RU)": "C0", "Fairy tales": "C2", "proza.ru": "C3", "ficbook.net": "C4"}
    for name, sh in arcs.items():
        sc = (sh - sh.mean(0)) @ Vt[:2].T          # centred per source: compare shapes
        axes[0].plot(sc[:, 1], sc[:, 0], color=cols[name], label=name)
        for j in (0, 5, 11):
            axes[0].text(sc[j, 1], sc[j, 0], str(j + 1), color=cols[name], fontsize=9)
        xs, ys = sh[:, truth].sum(1) * 100, sh[:, place].sum(1) * 100
        axes[1].plot(xs - xs.mean(), ys - ys.mean(), color=cols[name], label=name)
        for j in (0, 11):
            axes[1].text(xs[j] - xs.mean(), ys[j] - ys.mean(), str(j + 1), color=cols[name])
    axes[0].set_xlabel("PC2 of TV arc"); axes[0].set_ylabel("PC1 of TV arc")
    axes[0].set_title("12-part arcs projected on the TV principal components\n(each source centred)")
    axes[1].set_xlabel("truth & reconciliation topics, % (centred)")
    axes[1].set_ylabel("location topics, % (centred)")
    axes[1].set_title("6-topic path (Fig. 4 analogue)")
    for name in [n for n in ["Fairy tales", "proza.ru", "ficbook.net"] if n in srcs]:
        dt_, ep_ = data[name]
        da = dialogue_arc(srcs[name].split("_", 1)[1], set(ep_.eid))
        axes[2].plot(X12, da * 100, color=cols[name], label=name, marker="o")
    axes[2].set_xlabel("12th of text"); axes[2].set_ylabel("dialogue words, %")
    axes[2].set_title("Dialogue share by 12th (TV = 100 %)")
    for ax in axes:
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "arcs_by_source.png"), dpi=130); plt.close(fig)

    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    print(nres.round(3).to_string(index=False))
    print(pd.DataFrame(strata).round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
