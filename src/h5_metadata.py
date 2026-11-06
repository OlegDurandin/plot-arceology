"""Complement H5 with all-text descriptions and group-level metadata permutations.

Each dependence unit receives equal weight. Permute the unit-level vector of
cluster frequencies against unit-level metadata prevalence; no iid-text assumption.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_trajectories import OUT, WORK, bh, fit, silhouette
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def statistic(a, b):
    cov = a.T @ b / len(a) - a.mean(0) * b.mean()
    return float(np.sum(cov ** 2))


def main():
    rng = np.random.default_rng(72)
    rows, descriptions, nuisance, chunk_controls = [], [], [], []
    for source in ["TV", "TALES", "PROZA", "FICBOOK"]:
        ep = pd.read_csv(OUT / f"assignments_{source}.csv")
        group_col = "author" if source == "TALES" else "series"
        group = ep[group_col].astype(str)
        with np.load(WORK / f"{source}.npz") as saved:
            x = saved["scores"][:, :, :10].reshape(len(ep), -1)
        for n_chunks in [6, 12]:
            mask = ep.n_chunks.to_numpy() == n_chunks
            if mask.sum() < 100:
                continue
            model = fit(x[mask], ep.cluster.nunique())
            chunk_controls.append(dict(source=source, n_chunks=n_chunks, texts=int(mask.sum()),
                                       ari_with_main=float(adjusted_rand_score(ep.cluster[mask], model.labels_)),
                                       silhouette=silhouette(x[mask], model.labels_)))
        representations = {"trajectory": "cluster", "mean_topic": "mean_topic_cluster",
                           "normalized": "normalized_cluster"}
        features = {"TV": ["genres"], "TALES": ["regions", "category"],
                    "PROZA": ["year"], "FICBOOK": ["tags", "direction", "original"]}[source]
        masks = {}
        for field in features:
            strings = ep[field].fillna("").astype(str)
            for token in sorted({t for cell in strings for t in cell.split("|") if t}):
                y = strings.map(lambda s: token in s.split("|")).astype(float)
                if min(y.sum(), len(y) - y.sum()) < 20:
                    continue
                masks[field, token] = y
                for c, sub in ep.groupby("cluster"):
                    descriptions.append(dict(source=source, field=field, label=token, cluster=int(c),
                                             texts=len(sub), positives=int(y.loc[sub.index].sum()),
                                             prevalence=float(y.loc[sub.index].mean()), overall=float(y.mean())))
        for rep, column in representations.items():
            cluster = pd.get_dummies(ep[column], dtype=float).groupby(group).mean()
            a = cluster.to_numpy()
            # Same permutation bank for comparable representations.
            permutations = np.array([np.random.default_rng(j + 72).permutation(len(a)) for j in range(999)])
            for (field, token), y in masks.items():
                b = y.groupby(group).mean().reindex(cluster.index).to_numpy()
                positive_units = int((b > 0).sum())
                negative_units = int((b < 1).sum())
                if min(positive_units, negative_units) < 5:
                    continue
                score = statistic(a, b)
                # Reuse matrix multiplication instead of repeated per-permutation Python calls.
                centred_a = a - a.mean(0)
                b_perm = b[permutations] - b.mean()
                null = np.sum((b_perm @ centred_a / len(a)) ** 2, axis=1)
                rows.append(dict(source=source, representation=rep, field=field, label=token,
                                 units=len(a), positive_units=positive_units, texts=int(y.sum()),
                                 covariance_norm=score,
                                 p=float((1 + (null >= score - 1e-15).sum()) / 1000)))
        for c, sub in ep.groupby("cluster"):
            row = dict(source=source, cluster=int(c), texts=len(sub), units=sub[group_col].nunique(),
                       median_chunks=float(sub.n_chunks.median()),
                       median_norm=float(sub.trajectory_norm.median()))
            row["length_cluster_nmi"] = float(normalized_mutual_info_score(ep.cluster, ep.n_chunks))
            for col in ["n_words_clean", "dialogue", "duration"]:
                if col in sub:
                    row[f"median_{col}"] = float(sub[col].median())
            nuisance.append(row)
    result = pd.DataFrame(rows)
    result["q_bh"] = np.nan
    # Primary family: all tested source/label pairs, trajectory representation only.
    for rep in result.representation.unique():
        mask = result.representation == rep
        result.loc[mask, "q_bh"] = bh(result.loc[mask, "p"])
    result.to_csv(OUT / "group_associations.csv", index=False)
    pd.DataFrame(descriptions).to_csv(OUT / "enrichment_all_texts.csv", index=False)
    pd.DataFrame(nuisance).to_csv(OUT / "cluster_covariates.csv", index=False)
    pd.DataFrame(chunk_controls).to_csv(OUT / "chunk_controls.csv", index=False)
    json.dump(dict(permutations=999, unit="show / author; tales author",
                   statistic="squared covariance norm between cluster frequencies and label prevalence",
                   min_text_positives=20, min_units=5,
                   adjustment="BH across all source/label pairs separately per representation",
                   caveat="marginal associations; not adjusted for topic, length or other labels"),
              open(OUT / "metadata_protocol.json", "w"), indent=2)
    selection = pd.read_csv(OUT / "model_selection.csv")
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
    for col, source in enumerate(["TV", "TALES", "PROZA", "FICBOOK"]):
        for dim, sub in selection[selection.source == source].groupby("dim"):
            axes[0, col].plot(sub.k, sub.silhouette, marker="o", label=f"{dim} PCs")
            axes[1, col].plot(sub.k, sub.stability_ari, marker="o", label=f"{dim} PCs")
        axes[0, col].set_title(source)
        axes[0, col].legend(fontsize=8)
        axes[1, col].axhline(.6, color="grey", ls="--", lw=.8)
        axes[1, col].set(xlabel="k", ylim=(0, 1))
        for ax in axes[:, col]:
            ax.grid(alpha=.25)
    axes[0, 0].set_ylabel("silhouette")
    axes[1, 0].set_ylabel("mean stability ARI")
    fig.tight_layout(); fig.savefig(OUT / "selection.png", dpi=140); plt.close(fig)
    print(result[result.representation == "trajectory"].sort_values(["q_bh", "p"]).head(30).to_string(index=False))


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
