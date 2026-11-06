"""Post-hoc sensitivity: extend C grid after boundary selections in the primary run.

The original fixed protocol/results are retained. Same seed-0 author folds, primary
targets, features and models; this run is exploratory and cannot replace the primary.
"""
import json
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from validate_endings import OUT, MODELS, prepare, design, model, splits


def main():
    directory = OUT / "extended_grid"
    directory.mkdir(exist_ok=True)
    original = json.load(open(OUT / "protocol.json"))
    cs = [.0001, .001, .01, .1, 1., 10., 100.]
    protocol = dict(original, primary_targets=["happy", "open"], secondary_targets=[],
                    outer_seeds=[0], cs=cs,
                    status="post-hoc sensitivity; primary run often selected grid boundary C",
                    pca_dimensions=16)
    json.dump(protocol, open(directory / "protocol.json", "w"), ensure_ascii=False, indent=2)
    ep, parts, mean, cov, texts, endings, targets, strata, groups = prepare()
    for name in ["cohort.csv", "tag_audit.csv", "data_audit.json", "token_window_audit.json"]:
        shutil.copy2(OUT / name, directory / name)
    targets = {k: targets[k] for k in ["happy", "open"]}
    predictions, choices, fold_rows, scores = [], [], [], []
    for fold, (train, test) in enumerate(splits(strata, groups, 0)):
        tuning = {(tag, name, c): [] for tag in targets for name in MODELS for c in cs}
        for itrain, ival in splits(strata[train], groups[train], fold + 100, n=3):
            tr, va = train[itrain], train[ival]
            features, _ = design(parts, mean, cov, texts, endings, tr)
            for tag, y in targets.items():
                for name in MODELS:
                    for c in cs:
                        fitted = model(c).fit(features[name][tr], y[tr])
                        pred = fitted.predict_proba(features[name][va])[:, 1]
                        tuning[tag, name, c].append(average_precision_score(y[va], pred))
        features, _ = design(parts, mean, cov, texts, endings, train)
        for tag, y in targets.items():
            fold_rows.append(dict(seed=0, fold=fold, target=tag, train=len(train), test=len(test)))
            for name in MODELS:
                c = max(cs, key=lambda cc: np.mean(tuning[tag, name, cc]))
                fitted = model(c).fit(features[name][train], y[train])
                pred = fitted.predict_proba(features[name][test])[:, 1]
                choices.append(dict(seed=0, fold=fold, target=tag, model=name, c=c,
                                    inner_ap=float(np.mean(tuning[tag, name, c]))))
                predictions.extend(dict(seed=0, fold=fold, target=tag, model=name, eid=ep.eid.iloc[j],
                                        author=groups[j], y=int(y[j]), probability=float(p))
                                   for j, p in zip(test, pred))
        pd.DataFrame(predictions).to_csv(directory / "predictions.csv", index=False)
        print("EXTENDED DONE",fold,flush=True)
    for (tag, name), g in pd.DataFrame(predictions).groupby(["target", "model"]):
        scores.append(dict(seed=0, target=tag, model=name, texts=len(g), positives=int(g.y.sum()),
                           prevalence=float(g.y.mean()), ap=average_precision_score(g.y, g.probability),
                           roc_auc=roc_auc_score(g.y, g.probability)))
    pd.DataFrame(scores).to_csv(directory / "scores.csv", index=False)
    pd.DataFrame(choices).to_csv(directory / "tuning.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(directory / "folds.csv", index=False)
    import p1_report
    p1_report.OUT = directory
    p1_report.main()


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
