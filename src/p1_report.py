"""Paired author-bootstrap intervals and diagnostics for P1 fixed OOF predictions."""
import json
import hashlib
import importlib.metadata
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from validate_endings import OUT, ROOT


def weighted_ap_many(y, probability, weights):
    """Exact sklearn-style AP for repeated group bootstrap weights, including ties."""
    order = np.argsort(-probability, kind="stable")
    ys = np.asarray(y)[order]
    ends = np.r_[np.flatnonzero(np.diff(np.asarray(probability)[order]) != 0), len(y) - 1]
    w = np.asarray(weights)[:, order]
    tp = np.cumsum(w * ys, axis=1)[:, ends]
    total = np.cumsum(w, axis=1)[:, ends]
    increments = np.diff(np.column_stack([np.zeros(len(w)), tp]), axis=1)
    precision = np.divide(tp, total, out=np.zeros_like(tp, dtype=float), where=total > 0)
    return np.divide(np.sum(increments * precision, axis=1), tp[:, -1],
                     out=np.full(len(w), np.nan), where=tp[:, -1] > 0)


def main():
    protocol = json.load(open(OUT / "protocol.json"))
    predictions = pd.read_csv(OUT / "predictions.csv")
    cohort = pd.read_csv(OUT / "cohort.csv").set_index("eid")
    expected = len(cohort)
    assert all(len(g) == expected and g.eid.is_unique
               for _, g in predictions.groupby(["seed", "target", "model"])), "Incomplete OOF predictions"
    assert set(predictions.seed) == set(protocol["outer_seeds"]), "Missing split seeds"
    manifest = dict(python=platform.python_version(), packages={p: importlib.metadata.version(p)
                    for p in ["numpy", "pandas", "scipy", "scikit-learn", "transformers"]},
                    sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [ROOT / "src/validate_endings.py", ROOT / "src/p1_report.py",
                                      ROOT / "src/p1_audit.py", ROOT / "work/EMB/FICBOOK.npz",
                                      ROOT / "work/FICBOOK/episodes.csv"]})
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2)
    metrics, intervals, sensitivity = [], [], []
    contrasts = ["B2", "B4_s101", "B4_s202", "B4_s303", "B1_TFIDF", "B2_TFIDF"]
    for (seed, target), group in predictions.groupby(["seed", "target"]):
        wide = group.pivot(index="eid", columns="model", values="probability").sort_index()
        info = group.drop_duplicates("eid").set_index("eid").loc[wide.index]
        y = info.y.to_numpy()
        authors, inv = np.unique(info.author.to_numpy(), return_inverse=True)
        samples = {name: np.empty(protocol["bootstrap_reps"]) for name in wide.columns}
        rng = np.random.default_rng(900 + int(seed))
        for start in range(0, protocol["bootstrap_reps"], 250):
            size = min(250, protocol["bootstrap_reps"] - start)
            units = rng.multinomial(len(authors), np.full(len(authors), 1 / len(authors)), size=size)
            weights = units[:, inv]
            for name in wide.columns:
                samples[name][start:start + size] = weighted_ap_many(y, wide[name].to_numpy(), weights)
        for name in wide.columns:
            point = average_precision_score(y, wide[name])
            low, high = np.nanquantile(samples[name], [.025, .975])
            metrics.append(dict(seed=seed, target=target, model=name, ap=point,
                                ap_ci95_low=low, ap_ci95_high=high, prevalence=float(y.mean())))
        for baseline in contrasts:
            delta = samples["B3"] - samples[baseline]
            point = average_precision_score(y, wide.B3) - average_precision_score(y, wide[baseline])
            lo, hi = np.nanquantile(delta, [.025, .975])
            slo, shi = np.nanquantile(delta, [.00625, .99375])
            intervals.append(dict(seed=seed, target=target, comparison=f"B3-{baseline}", delta_ap=point,
                                  ci95_low=lo, ci95_high=hi, ci9875_low=slo, ci9875_high=shi,
                                  primary=(seed == 0 and target in ["happy", "open"] and
                                           baseline in ["B2", "B4_s101"]),
                                  practical=point >= protocol["practical_ap_gain"],
                                  adjusted_interval_positive=slo > 0))
        b4_values = [samples[f"B4_s{s}"] for s in protocol["shuffle_seeds"]]
        diff = samples["B3"] - np.mean(b4_values, axis=0)
        intervals.append(dict(seed=seed, target=target, comparison="B3-mean_AP_B4",
                              delta_ap=average_precision_score(y, wide.B3) - np.mean([
                                  average_precision_score(y, wide[f"B4_s{s}"])
                                  for s in protocol["shuffle_seeds"]]),
                              ci95_low=np.nanquantile(diff, .025), ci95_high=np.nanquantile(diff, .975),
                              ci9875_low=np.nanquantile(diff, .00625), ci9875_high=np.nanquantile(diff, .99375),
                              primary=False, practical=False,
                              adjusted_interval_positive=np.nanquantile(diff, .00625) > 0))
        subcohort = cohort.loc[wide.index]
        masks = {"all": np.ones(len(y), bool),
                 "without_PWP": ~subcohort.tags.fillna("").map(lambda s: "PWP" in s.split("|")).to_numpy(),
                 "originals_only": subcohort.original.to_numpy().astype(bool)}
        audit_path = OUT / "token_window_audit.json"
        if audit_path.exists():
            audit = json.load(open(audit_path))
            if "affected_eids" in audit:
                masks["without_truncated_windows"] = ~wide.index.isin(audit["affected_eids"])
            if "marker_eids" in audit:
                masks["without_label_markers"] = ~wide.index.isin(audit["marker_eids"])
        for subset, mask in masks.items():
            for name in wide.columns:
                if len(set(y[mask])) < 2:
                    continue
                sensitivity.append(dict(seed=seed, target=target, subset=subset, model=name,
                                        texts=int(mask.sum()), positives=int(y[mask].sum()),
                                        prevalence=float(y[mask].mean()),
                                        ap=average_precision_score(y[mask], wide[name].to_numpy()[mask]),
                                        roc_auc=roc_auc_score(y[mask], wide[name].to_numpy()[mask])))
        print(seed, target, "bootstrap complete", flush=True)
    metric = pd.DataFrame(metrics); contrast = pd.DataFrame(intervals)
    contrast["primary_design_comparison"] = contrast.primary
    if str(protocol.get("status", "")).startswith("post-hoc"):
        contrast["primary"] = False
    metric.to_csv(OUT / "metric_intervals.csv", index=False)
    contrast.to_csv(OUT / "contrasts.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUT / "sensitivity.csv", index=False)
    conclusion = {}
    for target in ["happy", "open"]:
        rows = contrast[contrast.primary_design_comparison & (contrast.target == target)]
        conclusion[target] = dict(supported=bool((rows.adjusted_interval_positive & rows.practical).all()),
                                  contrasts=rows.to_dict("records"))
    json.dump(dict(analysis_phase=protocol.get("status", "initial fixed protocol"), primary_results=conclusion,
                   limitations=["fixed OOF author-bootstrap does not refit models",
                                "three author split seeds are sensitivity, not independent samples",
                                "author tags are not independently verified story endings",
                                "same corpus already explored in H5",
                                "frozen e5 windows may be truncated; no new encoding in this experiment" ]),
              open(OUT / "conclusion.json", "w"), ensure_ascii=False, indent=2)
    names = ["B0", "B1", "B2", "B3", "B4_s101", "B1_TFIDF", "B2_TFIDF"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, target in zip(axes, ["happy", "open"]):
        sub = metric[(metric.seed == 0) & (metric.target == target)].set_index("model").loc[names]
        ax.vlines(np.arange(len(names)), sub.ap_ci95_low, sub.ap_ci95_high, color="C0")
        ax.scatter(np.arange(len(names)), sub.ap, color="C0")
        ax.axhline(sub.prevalence.iloc[0], color="grey", ls="--", label="tag prevalence")
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        ax.set(title=target, ylabel="OOF average precision (95% author-bootstrap CI)")
        ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "primary_ap.png", dpi=140); plt.close(fig)
    primary = contrast[contrast.primary_design_comparison].copy()
    fig, ax = plt.subplots(figsize=(9, 4))
    pos = np.arange(len(primary))
    ax.hlines(pos, primary.ci9875_low, primary.ci9875_high, color="C0", lw=2)
    ax.scatter(primary.delta_ap, pos, color="C0", zorder=3)
    ax.axvline(0, color="grey", ls="--")
    ax.axvline(protocol["practical_ap_gain"], color="C2", ls=":", label="practical target +0.02 AP")
    ax.set_yticks(pos, [f"{r.target}: {r.comparison}" for r in primary.itertuples()])
    ax.set_xlabel("Paired AP gain; 98.75% author-bootstrap intervals (4 design comparisons)")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "primary_deltas.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
