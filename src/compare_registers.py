"""P2: matched fairy tales vs proza.ru; post-stratified group bootstrap.

Fixed, covariate-only matched cohorts; simultaneous representation comparisons.
Intervals condition on matching. Repeated independent matching is a sensitivity.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analyze import segment, overlap_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/p2"
WORK = ROOT / "work/P2"
BINS = np.array([999, 1200, 1500, 1800, 2200, 2700, 3600, 5000, 7200, 10000])
BOOT = 2000
NULL_REPS = 100


def eigstats(x, standardized=False):
    """Batch of [12, dimensions] profiles; optional per-coordinate standardization."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x[None]
    x = x - x.mean(1, keepdims=True)
    if standardized:
        sd = x.std(1, keepdims=True)
        x = np.divide(x, sd, out=np.zeros_like(x), where=sd > 1e-12)
    gram = x @ x.transpose(0, 2, 1)
    ev = np.maximum(np.linalg.eigvalsh(gram)[:, ::-1], 0)
    total = ev.sum(1)
    return dict(pc2_remaining=ev[:, 1] / ev[:, 1:].sum(1),
                var2=ev[:, :2].sum(1) / total)


def quotas(counts, n):
    counts = np.asarray(counts, dtype=int)
    if counts.sum() < n:
        raise ValueError("Insufficient common stratum capacity")
    exact = n * counts / counts.sum()
    q = np.floor(exact).astype(int)
    for j in np.argsort(-(exact - q))[:n - q.sum()]:
        q[j] += 1
    assert q.sum() == n and (q <= counts).all()
    return q


def bootstrap_weights(groups, strata, rng, n=BOOT, equal_author=False):
    """Resample whole units, then restore fixed length-stratum totals.

    The same unit multiplicity applies to all its texts, across strata. Weights
    are fractional after post-stratification; nominal sample size stays fixed.
    """
    groups = np.asarray(groups).astype(str)
    _, inv = np.unique(groups, return_inverse=True)
    units = int(inv.max()) + 1
    strata = np.asarray(strata)
    base = np.ones(len(inv))
    if equal_author:
        for s in np.unique(strata):
            mask = strata == s
            size = np.bincount(inv[mask], minlength=units)
            base[mask] = 1 / size[inv[mask]]
    output = np.empty((n, len(inv)))
    rejected = 0
    for start in range(0, n, 200):
        size = min(200, n - start)
        mult = rng.multinomial(units, np.full(units, 1 / units), size=size)
        w = mult[:, inv] * base
        valid = np.ones(size, bool)
        for s in np.unique(strata):
            valid &= w[:, strata == s].sum(1) > 0
        while not valid.all():
            bad = np.flatnonzero(~valid); rejected += len(bad)
            mult[bad] = rng.multinomial(units, np.full(units, 1 / units), size=len(bad))
            w[bad] = mult[bad][:, inv] * base
            valid[bad] = True
            for s in np.unique(strata):
                valid[bad] &= w[bad][:, strata == s].sum(1) > 0
        for s in np.unique(strata):
            mask = strata == s
            w[:, mask] *= mask.sum() / w[:, mask].sum(1)[:, None]
        output[start:start + size] = w
    point = base.copy()
    for s in np.unique(strata):
        mask = strata == s
        point[mask] *= mask.sum() / point[mask].sum()
    return output, point, rejected


def load_source(source):
    ep = pd.read_csv(ROOT / f"work/JOINT_{source}/episodes.csv")
    representations = {}
    for name in ["LDA", "E5"]:
        if name == "LDA":
            with np.load(ROOT / f"work/JOINT_{source}/doc_topics.npz") as z:
                dt = {k: z[k] for k in z.files}
        else:
            with np.load(ROOT / f"work/EMB/{source}.npz") as z:
                dt = dict(eid=z["eid"], i=z["i"], n=z["n"], length=z["words"].astype(np.float32),
                          counts=z["emb"].astype(np.float32) * z["words"][:, None])
        order = np.lexsort((dt["i"], dt["eid"]))
        dt = {k: v[order] for k, v in dt.items()}
        eid, c, length = segment(dt, 12)
        valid = (length > 0).all(1)
        shares = np.divide(c, length[:, :, None], out=np.zeros_like(c), where=length[:, :, None] > 0)
        starts = np.flatnonzero(np.r_[True, dt["eid"][1:] != dt["eid"][:-1]])
        representations[name] = dict(eid=eid, valid=valid, shares=shares, length=length,
                                     raw=dt, starts=starts)
    common = set(ep[ep.n_words_clean.between(1000, 10000)].eid)
    for rep in representations.values():
        common &= set(rep["eid"][rep["valid"]])
    ep = ep[ep.eid.isin(common)].sort_values("eid").reset_index(drop=True)
    ep["stratum"] = np.searchsorted(BINS[1:], ep.n_words_clean.to_numpy(), side="left")
    ep["source"] = source
    ep["unit_author"] = ep["author" if source == "TALES" else "series"].fillna("unknown").astype(str)
    ep["unit_collection"] = ep.category.fillna("unknown").astype(str) if source == "TALES" else ep.unit_author
    for name, rep in representations.items():
        idx = pd.Index(rep["eid"]).get_indexer(ep.eid)
        rep["shares"] = rep["shares"][idx]
        rep["length"] = rep["length"][idx]
        rep["starts"] = rep["starts"][idx]
        rep["n"] = rep["raw"]["n"][rep["starts"]]
        del rep["valid"], rep["eid"]
    return ep, representations


def matched(ep, n, rng, exact_chunks=None, omit=None, q=None):
    pool = np.arange(len(ep))
    if exact_chunks:
        pool = pool[ep.n_chunks.to_numpy()[pool] == exact_chunks]
    if omit:
        pool = pool[ep.unit_collection.to_numpy()[pool] != omit]
    counts = np.bincount(ep.stratum.to_numpy()[pool], minlength=len(BINS) - 1)
    if q is None:
        q = quotas(counts, n)
    if (q > counts).any():
        raise ValueError(f"Not enough texts: quotas={q}, capacities={counts}")
    selected = np.concatenate([rng.choice(pool[ep.stratum.to_numpy()[pool] == s], count, replace=False)
                               for s, count in enumerate(q) if count])
    return selected, q


def profiles(shares, length, w, weighting):
    n, parts, dim = shares.shape
    if weighting == "text":
        a = (w @ shares.reshape(n, -1)).reshape(len(w), parts, dim)
        return a / w.sum(1)[:, None, None]
    counts = shares * length[:, :, None]
    a = (w @ counts.reshape(n, -1)).reshape(len(w), parts, dim)
    return a / (w @ length)[:, :, None]


def evaluate(a, baselines):
    raw, std = eigstats(a), eigstats(a, standardized=True)
    mu, sd = baselines
    return dict(pc2_remaining_std=std["pc2_remaining"], pc2_remaining_raw=raw["pc2_remaining"],
                var2_raw=raw["var2"], var2_std=std["var2"],
                z_pc2_std=(std["pc2_remaining"] - mu) / sd)


def shuffled_profiles(rep, idx, rng):
    starts, sizes = rep["starts"][idx], rep["n"][idx]
    dim = rep["shares"].shape[-1]
    count = np.zeros((len(idx), 12, dim), np.float32)
    length = np.zeros((len(idx), 12), np.float32)
    for nn in np.unique(sizes):
        mask = np.flatnonzero(sizes == nn)
        take = starts[mask, None] + np.arange(nn)[None]
        perm = np.argsort(rng.random(take.shape), axis=1)
        take = np.take_along_axis(take, perm, axis=1)
        w = overlap_matrix(int(nn), 12).astype(np.float32)
        count[mask] = np.einsum("mn,bnk->bmk", w, rep["raw"]["counts"][take], optimize=True)
        length[mask] = np.einsum("mn,bn->bm", w, rep["raw"]["length"][take], optimize=True)
    return count / length[:, :, None], length


def plot_effects(effects, metric="pc2_remaining_std", output="effects.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, name in zip(axes, ["LDA", "E5"]):
        e = effects[(effects.representation == name) & (effects.weighting == "text") &
                    (effects.metric == metric)].reset_index(drop=True)
        ax.hlines(np.arange(len(e)), e.ci975_low, e.ci975_high, color="C0")
        ax.scatter(e.difference, np.arange(len(e)))
        ax.axvline(0, color="grey", ls="--")
        ax.set_yticks(np.arange(len(e)), e.case)
        label = "Delta standardised PC2 remaining variance" if metric == "pc2_remaining_std" else "Delta variance of first 2 PCs (raw)"
        ax.set(title=name, xlabel=label)
        ax.grid(alpha=.2)
    fig.suptitle("Fairy tales minus proza.ru: 97.5% conditional group-bootstrap intervals", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / output, dpi=140); plt.close(fig)


def interval_diagnostics(effects):
    """Post-hoc sensitivity to reflecting percentile intervals about the point."""
    frame=effects.copy()
    for level in ['95','975']:
        frame[f'basic_ci{level}_low']=2*frame.difference-frame[f'ci{level}_high']
        frame[f'basic_ci{level}_high']=2*frame.difference-frame[f'ci{level}_low']
    frame['status']='post-hoc basic-bootstrap sensitivity; does not replace primary percentile protocol'
    frame.to_csv(OUT/'interval_sensitivity.csv',index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
    json.dump(dict(primary_contrast="TALES minus PROZA", primary_statistic="pc2_remaining_std",
                   primary_weight="equal_text", primary_representations=["LDA127", "E5_384"],
                   primary_n=1000, chunk_control_n=600, length_range=[1000, 10000], bins=BINS.tolist(),
                   bootstrap=BOOT, primary_interval=.975, secondary_interval=.95,
                   matching_seed=0, matching_repeats=20, null_repeats=NULL_REPS,
                   bootstrap_method="whole author resampling then restore fixed length-stratum totals",
                   inference_scope="conditional on covariate-only fixed matched cohort and trained representations",
                   primary_support="both representation lower 97.5% bounds >0; no point reversal in 6/12-chunk or equal-author controls",
                   note="not a direct measure of narrative formulaicity; no topic/embedding refitting"),
              open(OUT / "protocol.json", "w"), ensure_ascii=False, indent=2)
    data = {s: load_source(s) for s in ["TALES", "PROZA"]}
    pools = pd.concat([d[0] for d in data.values()])
    pools.to_csv(OUT / "eligible.csv", index=False)
    print("ELIGIBLE",pools.groupby("source").size().to_dict(),flush=True)
    baselines = {}
    for name in ["LDA", "E5"]:
        k = data["TALES"][1][name]["shares"].shape[-1]
        walks = np.cumsum(np.random.default_rng(700).standard_normal((5000, 12, k)), axis=1)
        rr = eigstats(walks, standardized=True)["pc2_remaining"]
        baselines[name] = (float(rr.mean()), float(rr.std()))
        del walks
    json.dump(baselines, open(OUT / "rw_baselines.json", "w"), indent=2)
    largest = data["TALES"][0].unit_collection.value_counts().index[0]
    cases = [("matched", 1000, None, False, False, None),
             ("chunks6", 600, 6, False, False, None),
             ("chunks12", 600, 12, False, False, None),
             ("equal_author", 1000, None, True, False, None),
             ("collection_units", 1000, None, False, True, None),
             ("omit_largest_collection", 1000, None, False, False, largest)]
    rows, balance, assignments, repeats, nulls = [], [], [], [], []
    for case, n, chunks, author_equal, collection, omit in cases:
        print("START",case,flush=True)
        idxs, weights, pts, estimates, points = {}, {}, {}, {}, {}
        # Same matching seed -> same cohort for alternative unit/weight analyses.
        rng = np.random.default_rng(0)
        idxs["TALES"], q = matched(data["TALES"][0], n, rng, chunks, omit)
        idxs["PROZA"], _ = matched(data["PROZA"][0], n, rng, chunks, q=q)
        for source in ["TALES", "PROZA"]:
            ep, reps = data[source]; idx = idxs[source]; cohort = ep.iloc[idx].copy()
            cohort["case"] = case; assignments.append(cohort)
            groups = cohort["unit_collection" if collection else "unit_author"].to_numpy()
            w, pt, rejected = bootstrap_weights(groups, cohort.stratum.to_numpy(),
                                                np.random.default_rng(42 if source == "TALES" else 43),
                                                equal_author=author_equal)
            weights[source], pts[source] = w, pt
            for s in sorted(cohort.stratum.unique()):
                sub = cohort[cohort.stratum == s]
                balance.append(dict(case=case, source=source, stratum=int(s), texts=len(sub),
                                    median_words=float(sub.n_words_clean.median()),
                                    mean_words=float(sub.n_words_clean.mean()),
                                    mean_words_per_chunk=float((sub.n_words_clean/sub.n_chunks).mean()),
                                    authors=sub.unit_author.nunique(), collections=sub.unit_collection.nunique(),
                                    rejected_bootstrap_draws=rejected))
            for name, rep in reps.items():
                sh, le = rep["shares"][idx], rep["length"][idx]
                for weighting in ["text", "word"]:
                    key = name, weighting
                    points[source, key] = evaluate(profiles(sh, le, pt[None], weighting), baselines[name])
                    estimates[source, key] = evaluate(profiles(sh, le, w, weighting), baselines[name])
        for name in ["LDA", "E5"]:
            for weighting in ["text", "word"]:
                key = name, weighting
                for metric in points["TALES", key]:
                    delta = estimates["TALES", key][metric] - estimates["PROZA", key][metric]
                    low, high = np.quantile(delta, [.025, .975])
                    slo, shi = np.quantile(delta, [.0125, .9875])
                    a, b = (float(points[s, key][metric][0]) for s in ["TALES", "PROZA"])
                    rows.append(dict(case=case, representation=name, weighting=weighting, metric=metric,
                                     n_per_source=n, tales=a, proza=b, difference=a-b,
                                     ci95_low=low, ci95_high=high, ci975_low=slo, ci975_high=shi,
                                     primary=case=="matched" and weighting=="text" and metric=="pc2_remaining_std"))
        # Matching sensitivity: independent covariate-only cohorts, no new k/model choices.
        if case in ["matched", "chunks6", "chunks12"]:
            for seed in range(1, 21):
                r = np.random.default_rng(seed)
                it, qq = matched(data["TALES"][0], n, r, chunks)
                ip, _ = matched(data["PROZA"][0], n, r, chunks, q=qq)
                for name in ["LDA", "E5"]:
                    val = {}
                    for source, idx in [("TALES", it), ("PROZA", ip)]:
                        rep = data[source][1][name]
                        val[source] = evaluate(profiles(rep["shares"][idx], rep["length"][idx],
                                                        np.ones((1, len(idx))), "text"), baselines[name])
                    for metric in val["TALES"]:
                        repeats.append(dict(case=case, seed=seed, representation=name, metric=metric,
                                            difference=float(val["TALES"][metric][0]-val["PROZA"][metric][0])))
        # Descriptive permutation check on each fixed cohort, preserving raw chunk lengths.
        if case in ["matched", "chunks6", "chunks12"]:
            for source in ["TALES", "PROZA"]:
                ep, reps = data[source]; idx = idxs[source]
                for name, rep in reps.items():
                    rng = np.random.default_rng(111)
                    values = []
                    for _ in range(NULL_REPS):
                        sh, le = shuffled_profiles(rep, idx, rng)
                        values.append(evaluate(profiles(sh, le, np.ones((1, len(idx))), "text"), baselines[name]))
                    for metric in values[0]:
                        v = np.array([a[metric][0] for a in values])
                        obs = float(points[source, (name, "text")][metric][0])
                        nulls.append(dict(case=case, source=source, representation=name, metric=metric,
                                          observed=obs, null_mean=float(v.mean()), null_sd=float(v.std()),
                                          difference=obs-float(v.mean()), permutations=NULL_REPS))
        pd.DataFrame(rows).to_csv(OUT / "effects.csv", index=False)
        print("DONE",case,flush=True)
    effects = pd.DataFrame(rows)
    pd.DataFrame(balance).to_csv(OUT / "balance.csv", index=False)
    pd.concat(assignments).to_csv(OUT / "cohorts.csv", index=False)
    pd.DataFrame(repeats).to_csv(OUT / "matching_sensitivity.csv", index=False)
    pd.DataFrame(nulls).to_csv(OUT / "shuffle_controls.csv", index=False)
    selected = effects[effects.primary]
    controls = effects[(effects.case.isin(["chunks6", "chunks12", "equal_author"])) &
                       (effects.weighting=="text") & (effects.metric=="pc2_remaining_std")]
    json.dump(dict(primary_supported=bool((selected.ci975_low>0).all()),
                   no_control_reversal=bool((controls.difference>0).all()),
                   strong_D2_support=bool((selected.ci975_low>0).all() and (controls.difference>0).all()),
                   primary=selected.to_dict("records"), omitted_collection=largest,
                   limitations=["intervals conditional on matching and fitted representations",
                                "author/collection labels are imperfect dependence units",
                                "fixed length bins do not match every word count exactly",
                                "no direct formulaicity or narrative-event measure",
                                "TV and ficbook not included in this primary contrast"]),
              open(OUT / "conclusion.json", "w"), ensure_ascii=False, indent=2)
    plot_effects(effects)
    plot_effects(effects, "var2_raw", "variance_effects.png")
    interval_diagnostics(effects)
    print("ALLDONE",flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
