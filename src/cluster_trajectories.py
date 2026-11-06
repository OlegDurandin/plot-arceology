"""H5: individual JOINT-LDA trajectories; run from any directory.

Six parts, text-centred topic shares, balanced shared PCA, per-source KMeans.
Labels are never used for selecting k. Outputs results/h5 and work/H5.
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
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from threadpoolctl import threadpool_limits

from analyze import segment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/h5"
WORK = ROOT / "work/H5"
SOURCES = ["TV", "TALES", "PROZA", "FICBOOK"]
REPS = 10
PERMS = 199


def fit(x, k, seed=42):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)


def silhouette(x, y):
    return float(silhouette_score(x, y, sample_size=min(1200, len(x)), random_state=0))


def stability(x, groups, k):
    """Disjoint reference texts; train group-subsamples without reference authors/shows."""
    rng = np.random.default_rng(19)
    units = np.unique(groups)
    reference_units = rng.choice(units, max(1, len(units) // 5), replace=False)
    reference = np.isin(groups, reference_units)
    train_units = units[~np.isin(units, reference_units)]
    pred = []
    for rep in range(REPS):
        selected = rng.choice(train_units, max(1, int(.8 * len(train_units))), replace=False)
        mask = np.isin(groups, selected)
        model = fit(x[mask], k, rep + 100)
        pred.append(model.predict(x[reference]))
    values = [adjusted_rand_score(pred[a], pred[b])
              for a in range(REPS) for b in range(a)]
    return float(np.mean(values)), float(np.min(values)), int(reference.sum())


def prepare():
    data = {}
    for source in SOURCES:
        directory = ROOT / "work" / f"JOINT_{source}"
        with np.load(directory / "doc_topics.npz") as z:
            dt = {k: z[k] for k in z.files}
        eid, c, length = segment(dt, 6)
        ep = pd.read_csv(directory / "episodes.csv").set_index("eid").loc[eid].reset_index()
        valid = (ep.n_chunks.to_numpy() >= 6) & (length > 0).all(1)
        ep = ep.loc[valid].reset_index(drop=True)
        shares = c[valid] / length[valid, :, None]
        # Unweighted mean of the six equal narrative-time parts: removes constant topic offset.
        residual = shares - shares.mean(1, keepdims=True)
        if source == "TV":
            meta = pd.read_csv(ROOT / "work/RU/episodes_meta.csv")
            ep = ep.merge(meta[["eid", "genres"]], on="eid", how="left", validate="one_to_one")
            groups = ep.series.astype(str).to_numpy()
        elif source == "TALES":
            groups = ep.author.fillna(ep.category).fillna("unknown").astype(str).to_numpy()
        else:
            groups = ep.series.astype(str).to_numpy()
        data[source] = dict(ep=ep, shares=shares, residual=residual, groups=groups)
        print(source, len(ep), "eligible texts", len(set(groups)), "groups", flush=True)
    rng = np.random.default_rng(0)
    balanced = np.concatenate([d["residual"][rng.choice(len(d["ep"]), min(1500, len(d["ep"])),
                                                       replace=False)].reshape(-1, 127)
                               for d in data.values()])
    pca = PCA(n_components=20, svd_solver="full").fit(balanced)
    np.savez_compressed(WORK / "pca.npz", components=pca.components_, mean=pca.mean_,
                        explained_variance_ratio=pca.explained_variance_ratio_)
    return data, pca


def bh(values):
    values = np.asarray(values)
    order = np.argsort(values)
    result = np.empty(len(values))
    result[order] = np.minimum.accumulate((values[order] * len(values) /
                                          np.arange(1, len(values) + 1))[::-1])[::-1].clip(0, 1)
    return result


def association(source, ep, labels, groups):
    """Group-level label-preserving permutations; one random text per dependence unit."""
    rng = np.random.default_rng(32)
    # One representative per author/show avoids treating related texts as independent.
    indices = pd.DataFrame({"g": groups, "j": np.arange(len(ep))}).sample(frac=1, random_state=32)
    indices = indices.drop_duplicates("g").j.to_numpy()
    e = ep.iloc[indices].reset_index(drop=True)
    lab = labels[indices]
    fields = {"TV": ["genres"], "TALES": ["regions", "category"],
              "FICBOOK": ["tags", "direction", "original"],
              "PROZA": ["year"]}[source]
    rows, enrich = [], []
    for field in fields:
        series = e[field].fillna("").astype(str)
        tokens = sorted({v for cell in series for v in cell.split("|") if v})
        for token in tokens:
            y = series.map(lambda cell: token in cell.split("|")).to_numpy().astype(int)
            if min(y.sum(), len(y) - y.sum()) < 20:
                continue
            score = normalized_mutual_info_score(y, lab)
            null = [normalized_mutual_info_score(rng.permutation(y), lab) for _ in range(PERMS)]
            table = pd.crosstab(lab, y).reindex(columns=[0, 1], fill_value=0)
            v = float(np.sqrt(chi2_contingency(table, correction=False)[0] / len(y)))
            rows.append(dict(source=source, field=field, label=token, units=len(y), positives=int(y.sum()),
                             nmi=score, cramers_v=v, p=(1 + np.sum(np.array(null) >= score)) / (PERMS + 1)))
            for cluster in sorted(set(lab)):
                mask = lab == cluster
                enrich.append(dict(source=source, field=field, label=token, cluster=int(cluster),
                                   texts=int(mask.sum()), positives=int(y[mask].sum()),
                                   prevalence=float(y[mask].mean()), overall=float(y.mean())))
    return rows, enrich


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    WORK.mkdir(exist_ok=True, parents=True)
    data, pca = prepare()
    metrics, selected, associations, enrichments, profiles, controls = [], [], [], [], [], []
    topic_labels = pd.read_csv(ROOT / "work/JOINT/topics.csv").label.tolist()
    for source, d in data.items():
        ep, shares, residual, groups = (d[k] for k in ["ep", "shares", "residual", "groups"])
        scores = pca.transform(residual.reshape(-1, 127)).reshape(len(ep), 6, 20)
        models = {}
        for dim in [10, 20]:
            x = scores[:, :, :dim].reshape(len(ep), -1)
            for k in range(2, 9):
                model = fit(x, k)
                avg, minimum, reference_n = stability(x, groups, k)
                sizes = np.bincount(model.labels_, minlength=k)
                row = dict(source=source, dim=dim, k=k, silhouette=silhouette(x, model.labels_),
                           stability_ari=avg, stability_min_ari=minimum, reference_texts=reference_n,
                           smallest_fraction=float(sizes.min() / len(x)), inertia=float(model.inertia_))
                metrics.append(row)
                models[dim, k] = model
                print(source, dim, k, "sil", round(row["silhouette"], 3), "ARI", round(avg, 3), flush=True)
        candidates = [r for r in metrics if r["source"] == source and r["dim"] == 10]
        acceptable = [r for r in candidates if r["stability_ari"] >= .6 and r["smallest_fraction"] >= .05]
        # No stable solution: show best silhouette for exploration, explicitly mark H5 unsupported.
        best = max(acceptable or candidates, key=lambda r: r["silhouette"])
        k = best["k"]
        model = models[10, k]
        labels = model.labels_
        x = scores[:, :, :10].reshape(len(ep), -1)
        rng = np.random.default_rng(51)
        perm = np.argsort(rng.random((len(ep), 6)), axis=1)
        shuffled = scores[np.arange(len(ep))[:, None], perm, :10].reshape(len(ep), -1)
        norm = np.linalg.norm(x, axis=1)
        norm_x = x / np.maximum(norm[:, None], 1e-12)
        mean_model = fit(shares.mean(1), k)
        norm_model = fit(norm_x, k)
        null_model = fit(shuffled, k)
        null_ari, _, _ = stability(shuffled, groups, k)
        normalized_ari, _, _ = stability(norm_x, groups, k)
        dim_ari = adjusted_rand_score(labels, models[20, k].labels_)
        choice = dict(best, supported=bool(acceptable), dimension_ari=float(dim_ari),
                      mean_topic_ari=float(adjusted_rand_score(labels, mean_model.labels_)),
                      amplitude_normalized_ari=float(adjusted_rand_score(labels, norm_model.labels_)),
                      normalized_stability=normalized_ari,
                      shuffle_silhouette=silhouette(shuffled, null_model.labels_), shuffle_stability=null_ari,
                      texts=len(ep), units=len(set(groups)))
        selected.append(choice)
        assigned = ep.copy()
        assigned["cluster"] = labels
        assigned["mean_topic_cluster"] = mean_model.labels_
        assigned["normalized_cluster"] = norm_model.labels_
        assigned["trajectory_norm"] = norm
        assigned.to_csv(OUT / f"assignments_{source}.csv", index=False)
        ar, en = association(source, ep, labels, groups)
        associations.extend(ar); enrichments.extend(en)
        # Baseline metadata associations distinguish trajectory effects from average topic effects.
        for representation, baseline_labels in [("mean_topic", mean_model.labels_),
                                                  ("normalized", norm_model.labels_),
                                                  ("shuffled", null_model.labels_)]:
            ar_base, _ = association(source, ep, baseline_labels, groups)
            controls.extend(dict(r, representation=representation) for r in ar_base)
        fig, axes = plt.subplots(k, 2, figsize=(12, 3 * k), squeeze=False)
        for cluster in range(k):
            mask = labels == cluster
            profile = residual[mask].mean(0)
            top = np.argsort(-np.ptp(profile, axis=0))[:5]
            for topic in top:
                axes[cluster, 0].plot(range(1, 7), profile[:, topic], marker="o", label=topic_labels[topic][:48])
                profiles.extend(dict(source=source, cluster=cluster, topic=int(topic), label=topic_labels[topic],
                                     part=j + 1, mean_residual=float(profile[j, topic])) for j in range(6))
            axes[cluster, 0].axhline(0, color="grey", lw=.5)
            axes[cluster, 0].legend(fontsize=6)
            axes[cluster, 0].set_title(f"{source} cluster {cluster}: n={mask.sum()}")
            path = scores[mask].mean(0)
            axes[cluster, 1].plot(path[:, 0], path[:, 1], marker="o")
            for j in range(6):
                axes[cluster, 1].annotate(str(j + 1), path[j, :2])
            axes[cluster, 1].set(xlabel="shared PC1", ylabel="shared PC2")
        fig.tight_layout(); fig.savefig(OUT / f"centroids_{source}.png", dpi=140); plt.close(fig)
        np.savez_compressed(WORK / f"{source}.npz", scores=scores, labels=labels,
                            centres=model.cluster_centers_, eid=ep.eid.to_numpy().astype(str))
        pd.DataFrame(metrics).to_csv(OUT / "model_selection.csv", index=False)
        print("SELECTED", source, choice, flush=True)
    pd.DataFrame(selected).to_csv(OUT / "selected.csv", index=False)
    a = pd.DataFrame(associations)
    if len(a):
        a["q_bh"] = bh(a.p)
    a.to_csv(OUT / "associations.csv", index=False)
    pd.DataFrame(enrichments).to_csv(OUT / "enrichment.csv", index=False)
    baseline = pd.DataFrame(controls)
    if len(baseline):
        baseline["q_bh"] = bh(baseline.p)
    baseline.to_csv(OUT / "association_controls.csv", index=False)
    pd.DataFrame(profiles).to_csv(OUT / "profiles.csv", index=False)
    json.dump(dict(parts=6, min_chunks=6, dims=[10, 20], ks=list(range(2, 9)),
                   stability_reps=REPS, permutations=PERMS, seed=42,
                   selection="dim10; stability ARI>=.6 and smallest cluster>=5%; maximum silhouette",
                   pca_variance=pca.explained_variance_ratio_.tolist(),
                   metadata_unit="one fixed random text per author/show; tales grouped by author/category"),
              open(OUT / "protocol.json", "w"), indent=2)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
