"""P1: author-disjoint nested CV for content, ending and order controls.

Frozen e5 embeddings; every learned PCA, scaler, vocabulary and C uses train only.
Run: python src/validate_endings.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from analyze import segment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/p1"
WORK = ROOT / "work/P1"
TAGS = {"happy": "Счастливый финал", "open": "Открытый финал",
        "death": "Смерть основных персонажей", "tragedy": "Трагедия"}
SEEDS = [0, 17, 42]
SHUFFLES = [101, 202, 303]
CS = [.01, .1, 1., 10.]
MODELS = ["B0", "B1", "B2", "B3"] + [f"B4_s{s}" for s in SHUFFLES] + ["B1_TFIDF", "B2_TFIDF"]


def permute_internal(parts, seed):
    rng = np.random.default_rng(seed)
    out = parts.copy()
    order = np.argsort(rng.random((len(parts), 5)), axis=1)
    out[:, :5] = parts[np.arange(len(parts))[:, None], order]
    return out


def prepare():
    with np.load(ROOT / "work/EMB/FICBOOK.npz") as z:
        dt = dict(eid=z["eid"], i=z["i"], n=z["n"], length=z["words"].astype(np.float32),
                  counts=z["emb"].astype(np.float32) * z["words"][:, None])
    eid, c, length = segment(dt, 6)
    ep = pd.read_csv(ROOT / "work/FICBOOK/episodes.csv").set_index("eid").loc[eid].reset_index()
    ok = (length > 0).all(1) & (ep.n_chunks.to_numpy() >= 6)
    ep = ep[ok].reset_index(drop=True)
    c, length = c[ok], length[ok]
    parts = c / length[:, :, None]
    mean = c.sum(1) / length.sum(1)[:, None]
    residual = parts - parts.mean(1, keepdims=True)
    cov = np.column_stack([np.log1p(ep.n_words_clean), np.log1p(ep.n_chunks), ep.dialogue,
                           np.linalg.norm(residual, axis=(1, 2))]).astype(np.float32)
    text_parts = {}
    with gzip.open(ROOT / "work/EMB/FICBOOK_texts.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            text_parts.setdefault(r["eid"], []).append((r["i"], r["text"]))
    texts = np.array([" ".join(t for _, t in sorted(text_parts[e])) for e in ep.eid], dtype=object)
    # Final sixth by the same equal-chunk overlap convention, retaining whole boundary chunks.
    endings = np.array([" ".join(t for i, t in sorted(text_parts[e]) if (i + 1) / n > 5 / 6)
                        for e, n in zip(ep.eid, ep.n_chunks)], dtype=object)
    hashes = np.array([hashlib.sha256(" ".join(t.lower().split()).encode()).hexdigest() for t in texts])
    assert len(set(hashes)) == len(hashes), "Exact duplicates require connected author groups before CV"
    targets = {key: ep.tags.fillna("").map(lambda s: tag in s.split("|")).to_numpy().astype(int)
               for key, tag in TAGS.items()}
    strata = targets["happy"] + 2 * targets["open"]
    groups = ep.series.astype(str).to_numpy()
    ep["exact_text_sha256"] = hashes
    ep.to_csv(OUT / "cohort.csv", index=False)
    # Blind audit materials are local, contain no author tags and never enter model features.
    audit_idx = []
    rng = np.random.default_rng(811)
    for key in ["happy", "open"]:
        audit_idx.extend(rng.choice(np.flatnonzero(targets[key]), 6, replace=False).tolist())
    audit_idx.extend(rng.choice(np.flatnonzero(strata == 0), 12, replace=False).tolist())
    audit_idx = rng.permutation(np.unique(audit_idx))
    pd.DataFrame(dict(audit_id=np.arange(len(audit_idx)), eid=ep.eid.iloc[audit_idx].to_numpy(),
                      ending=endings[audit_idx], whole_text=texts[audit_idx])).to_json(
                          WORK / "audit_blind.jsonl", orient="records", lines=True, force_ascii=False)
    pd.DataFrame(dict(audit_id=np.arange(len(audit_idx)), **{k: v[audit_idx] for k, v in targets.items()})).to_csv(
        WORK / "audit_key.csv", index=False)
    audit = []
    for key, y in targets.items():
        audit.append(dict(target=key, tag=TAGS[key], texts=len(ep), positives=int(y.sum()),
                          positive_authors=int(ep.loc[y.astype(bool), "series"].nunique()),
                          prevalence=float(y.mean())))
    pd.DataFrame(audit).to_csv(OUT / "tag_audit.csv", index=False)
    overlaps = {a: {b: int((targets[a] & targets[b]).sum()) for b in targets} for a in targets}
    json.dump(dict(overlaps=overlaps, duplicate_text_sets=0, manual_audit="pending; blind packet in work/P1",
                   excluded=int((~ok).sum())), open(OUT / "data_audit.json", "w"), ensure_ascii=False, indent=2)
    return ep, parts, mean, cov, texts, endings, targets, strata, groups


def embedding_features(parts, mean, cov, train):
    pca = PCA(n_components=16, svd_solver="full").fit(parts[train].reshape(-1, parts.shape[-1]))
    projected = pca.transform(parts.reshape(-1, parts.shape[-1])).reshape(len(parts), 6, 16)
    m = pca.transform(mean)
    b1 = np.column_stack([cov, m])
    b2 = np.column_stack([b1, projected[:, -1]])
    result = {"B0": cov, "B1": b1, "B2": b2,
              "B3": np.column_stack([b2, (projected - projected.mean(1, keepdims=True)).reshape(len(parts), -1)])}
    for seed in SHUFFLES:
        perm = permute_internal(projected, seed)
        result[f"B4_s{seed}"] = np.column_stack([b2, (perm - perm.mean(1, keepdims=True)).reshape(len(parts), -1)])
    return result, pca


def design(parts, mean, cov, texts, endings, train):
    features, pca = embedding_features(parts, mean, cov, train)
    # Both vocabularies and IDF learned on training texts only.
    full_v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=12000, sublinear_tf=True)
    tail_v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=6000, sublinear_tf=True)
    full_v.fit(texts[train]); tail_v.fit(endings[train])
    full = full_v.transform(texts); tail = tail_v.transform(endings)
    cov_scaler = StandardScaler().fit(cov[train])
    cscaled = sparse.csr_matrix(cov_scaler.transform(cov))
    features["B1_TFIDF"] = sparse.hstack([cscaled, full], format="csr")
    features["B2_TFIDF"] = sparse.hstack([cscaled, full, tail], format="csr")
    # Embedding columns standardised within train; sparse TF-IDF keeps its conventional scale.
    for key in list(features):
        if "TFIDF" not in key:
            features[key] = StandardScaler().fit(features[key][train]).transform(features[key])
    return features, float(pca.explained_variance_ratio_.sum())


def model(c):
    return LogisticRegression(C=c, solver="liblinear", max_iter=2000, random_state=0)


def splits(strata, groups, seed, n=5):
    output = list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(
        np.zeros(len(strata)), strata, groups))
    for train, test in output:
        assert not set(groups[train]) & set(groups[test])
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
    # Written before fitting any predictive model: exploratory protocol, not a preregistration.
    protocol = dict(primary_targets=["happy", "open"], secondary_targets=["death", "tragedy"],
                    outer_seeds=SEEDS, outer_folds=5, inner_folds=3, cs=CS, pca_dimensions=16,
                    shuffle_seeds=SHUFFLES, primary_seed=0, primary_shuffle=101,
                    primary_metric="average_precision", primary_comparisons=["B3-B2", "B3-B4_s101"],
                    practical_ap_gain=.02, primary_family_size=4,
                    inference="author bootstrap of fixed OOF predictions; 98.75% Bonferroni intervals",
                    bootstrap_reps=5000, representation="frozen multilingual-e5-small; train-only PCA/scalers/IDF",
                    models=MODELS, labels_are="author tags, absent tag is not verified negative ending",
                    manual_audit="blind packet prepared; independent human review pending",
                    prior_exploration="same corpus and tags previously analysed in H5; not unseen external validation")
    json.dump(protocol, open(OUT / "protocol.json", "w"), ensure_ascii=False, indent=2)
    ep, parts, mean, cov, texts, endings, targets, strata, groups = prepare()
    predictions, choices, folds, scores, variance = [], [], [], [], []
    print("COHORT",len(ep),len(set(groups)),"authors",flush=True)
    for seed in SEEDS:
        for fold, (train, test) in enumerate(splits(strata, groups, seed)):
            print("START",seed,fold,flush=True)
            inner = splits(strata[train], groups[train], seed + fold + 100, n=3)
            tuning = {(tag, name, c): [] for tag in targets for name in MODELS for c in CS}
            for itrain, ival in inner:
                tr, va = train[itrain], train[ival]
                features, _ = design(parts, mean, cov, texts, endings, tr)
                for tag, y in targets.items():
                    assert len(set(y[tr])) == 2 and len(set(y[va])) == 2
                    for name in MODELS:
                        for c in CS:
                            m = model(c).fit(features[name][tr], y[tr])
                            pred = m.predict_proba(features[name][va])[:, 1]
                            tuning[tag, name, c].append(average_precision_score(y[va], pred))
            features, var = design(parts, mean, cov, texts, endings, train)
            variance.append(dict(seed=seed, fold=fold, pca_variance=var))
            for tag, y in targets.items():
                assert y[test].sum() > 0 and y[test].sum() < len(test)
                folds.append(dict(seed=seed, fold=fold, target=tag, train=len(train), test=len(test),
                                  train_authors=len(set(groups[train])), test_authors=len(set(groups[test])),
                                  train_positive=int(y[train].sum()), test_positive=int(y[test].sum())))
                for name in MODELS:
                    c = max(CS, key=lambda cc: np.mean(tuning[tag, name, cc]))
                    m = model(c).fit(features[name][train], y[train])
                    pred = m.predict_proba(features[name][test])[:, 1]
                    choices.append(dict(seed=seed, fold=fold, target=tag, model=name, c=c,
                                        inner_ap=float(np.mean(tuning[tag, name, c]))))
                    predictions.extend(dict(seed=seed, fold=fold, target=tag, model=name, eid=ep.eid.iloc[j],
                                            author=groups[j], y=int(y[j]), probability=float(p))
                                       for j, p in zip(test, pred))
            pd.DataFrame(predictions).to_csv(OUT / "predictions.csv", index=False)
            pd.DataFrame(choices).to_csv(OUT / "tuning.csv", index=False)
            pd.DataFrame(folds).to_csv(OUT / "folds.csv", index=False)
            print("DONE",seed,fold,"PCA",round(var,3),flush=True)
    pred = pd.DataFrame(predictions)
    for (seed, tag, name), g in pred.groupby(["seed", "target", "model"]):
        assert len(g)==len(ep) and g.eid.is_unique
        scores.append(dict(seed=seed, target=tag, model=name, texts=len(g), positives=int(g.y.sum()),
                           prevalence=float(g.y.mean()), ap=average_precision_score(g.y,g.probability),
                           roc_auc=roc_auc_score(g.y,g.probability)))
    pd.DataFrame(scores).to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame(variance).to_csv(OUT / "pca_variance.csv", index=False)
    print("ALLDONE",flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
