"""Stage 4: EN vs RU comparison on the parallel part of the corpus.

Russian subtitles are mostly translations of the English ones with identical timing,
so chunk (episode, i) is the same stretch of screen time in both languages.
  * topics are aligned EN<->RU by correlating their per-chunk shares over parallel chunks
    (Hungarian matching);
  * linear / arc trends of aligned topics are compared;
  * PCA arcs of both languages are compared.
Usage: python src/compare.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from analyze import corr, pca

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(lang):
    dt = dict(np.load(os.path.join(ROOT, "work", lang, "doc_topics.npz")))
    key = np.char.add(np.char.add(dt["eid"].astype(str), "#"), dt["i"].astype(str))
    return dt, key


def main():
    out = os.path.join(ROOT, "results", "compare")
    os.makedirs(out, exist_ok=True)
    ten = pd.read_csv(os.path.join(ROOT, "work", "EN", "topics.csv"))
    tru = pd.read_csv(os.path.join(ROOT, "work", "RU", "topics.csv"))
    den, ken = load("EN")
    dru, kru = load("RU")
    # parallel chunks: same episode, same chunking, same index
    common, ia, ib = np.intersect1d(ken, kru, return_indices=True)
    ok = (den["n"][ia] == dru["n"][ib]) & (den["length"][ia] >= 30) & (dru["length"][ib] >= 30)
    ia, ib = ia[ok], ib[ok]
    print("parallel chunks:", len(ia))
    A = den["counts"][ia] / den["length"][ia, None]
    B = dru["counts"][ib] / dru["length"][ib, None]
    A = (A - A.mean(0)) / A.std(0)
    B = (B - B.mean(0)) / B.std(0)
    C = A.T @ B / len(A)                                   # [K_en, K_ru] correlations
    r, c = linear_sum_assignment(-C)
    pairs = pd.DataFrame(dict(en=r, ru=c, corr=C[r, c],
                              en_label=ten.label.values[r], ru_label=tru.label.values[c]))

    sh_en = np.load(os.path.join(ROOT, "work", "EN", "share12.npy"))
    sh_ru = np.load(os.path.join(ROOT, "work", "RU", "share12.npy"))
    x = np.arange(1, 13, dtype=float)
    lin_en, lin_ru = corr(x, sh_en), corr(x, sh_ru)
    arc_en, arc_ru = corr(-(x - 6.5) ** 2, sh_en), corr(-(x - 6.5) ** 2, sh_ru)
    pairs["r_lin_en"], pairs["r_lin_ru"] = lin_en[r], lin_ru[c]
    pairs["r_arc_en"], pairs["r_arc_ru"] = arc_en[r], arc_ru[c]
    pairs["profile_corr"] = [np.corrcoef(sh_en[:, a], sh_ru[:, b])[0, 1] for a, b in zip(r, c)]
    pairs = pairs.sort_values("corr", ascending=False)
    pairs.to_csv(os.path.join(out, "topic_alignment.csv"), index=False)

    good = pairs["corr"] > 0.5
    s = dict(parallel_chunks=int(len(ia)),
             aligned_pairs_corr_gt_0_5=int(good.sum()),
             median_alignment_corr=round(float(pairs["corr"].median()), 3),
             r_lin_agreement_all=round(float(np.corrcoef(pairs.r_lin_en, pairs.r_lin_ru)[0, 1]), 3),
             r_lin_agreement_good=round(float(np.corrcoef(pairs.r_lin_en[good], pairs.r_lin_ru[good])[0, 1]), 3),
             r_arc_agreement_good=round(float(np.corrcoef(pairs.r_arc_en[good], pairs.r_arc_ru[good])[0, 1]), 3),
             median_profile_corr_good=round(float(pairs.profile_corr[good].median()), 3))
    sc_en, _, ev_en = pca(sh_en)
    sc_ru, _, ev_ru = pca(sh_ru)
    s["pc1_en_vs_ru"] = round(float(abs(np.corrcoef(sc_en[:, 0], sc_ru[:, 0])[0, 1])), 3)
    s["pc2_en_vs_ru"] = round(float(abs(np.corrcoef(sc_en[:, 1], sc_ru[:, 1])[0, 1])), 3)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    ax.hist(pairs["corr"], bins=30, color="grey")
    ax.set_title("EN↔RU topic alignment: correlation over parallel chunks")
    for ax, a, b, name in [(axes[1], "r_lin_en", "r_lin_ru", "linear trend r"),
                           (axes[2], "r_arc_en", "r_arc_ru", "arc trend r")]:
        ax.scatter(pairs[a][~good], pairs[b][~good], s=10, color="#bbbbbb", label="corr ≤ .5")
        ax.scatter(pairs[a][good], pairs[b][good], s=14, color="C3", label="corr > .5")
        ax.axline((0, 0), slope=1, color="k", lw=.5)
        ax.set_xlabel(f"EN {name}"); ax.set_ylabel(f"RU {name}")
        ax.set_title(f"{name} of aligned topics"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "alignment.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for sc, name, col in [(sc_en, "EN", "C0"), (sc_ru, "RU", "C3")]:
        sgn = np.sign(np.corrcoef(x, sc[:, 0])[0, 1]); sgn2 = np.sign(np.corrcoef((x - 6.5) ** 2, sc[:, 1])[0, 1])
        p1, p2 = sc[:, 0] * sgn / np.abs(sc[:, 0]).max(), sc[:, 1] * sgn2 / np.abs(sc[:, 1]).max()
        ax.plot(p2, p1, color=col, label=name)
        for j in range(12):
            ax.text(p2[j], p1[j], str(j + 1), color=col, fontsize=9)
    ax.set_xlabel("PC2 (normalised)"); ax.set_ylabel("PC1 (normalised)"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("Aggregate plot arc in PC space: EN vs RU")
    fig.tight_layout(); fig.savefig(os.path.join(out, "pca_en_ru.png"), dpi=130); plt.close(fig)

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    print(json.dumps(s, indent=1))
    print(pairs.head(25)[["corr", "en_label", "ru_label", "r_lin_en", "r_lin_ru"]].to_string())


if __name__ == "__main__":
    main()
