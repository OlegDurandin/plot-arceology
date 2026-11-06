"""Figure + table for the K sweep (results/ksweep/ksweep_<LANG>.csv from ksweep.py)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "ksweep")

PANELS = [
    ("npmi", "NPMI coherence (top-10)"),
    ("c_v", "C_v coherence (top-10)"),
    ("diversity", "topic diversity (unique top-10 words)"),
    ("n_lin09", "topics with |r_lin| > 0.9"),
    ("frac_lin07", "share of topics with |r_lin| > 0.7"),
    ("r_truth", "r(truth topics, time)"),
    ("r_place", "r(place topics, time)"),
    ("var2", "variance of 2 PCs"),
    ("z_pc2_std", "z(PC2) vs random walks (standardised)"),
    ("agree_pc1_ref", "|r| of PC1 trajectory with K=127 main model"),
]


def main():
    frames = [pd.read_csv(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT))
              if f.startswith("ksweep_") and f.endswith(".csv")]
    df = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(2, 5, figsize=(21, 7.5))
    for ax, (col, title) in zip(axes.flat, PANELS):
        for (lang, seed), g in df.groupby(["lang", "seed"]):
            g = g.sort_values("K")
            ax.plot(g.K, g[col], marker="o" if seed == 42 else "x",
                    ls="-" if seed == 42 else "--", color="C0" if lang == "RU" else "C3",
                    label=f"{lang} seed {seed}", alpha=1 if seed == 42 else .7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df.K.unique()))
        ax.set_xticklabels(sorted(df.K.unique()), fontsize=7)
        ax.set_title(title, fontsize=9); ax.grid(alpha=.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Sensitivity to the number of LDA topics K (held-out 20k episodes)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "ksweep.png"), dpi=120); plt.close(fig)
    cols = ["lang", "K", "seed", "npmi", "c_v", "diversity", "n_lin09", "frac_lin09", "frac_lin07",
            "n_arc07", "r_truth", "r_place", "var2", "pc1_time", "pc2_arc", "z_pc2_raw", "z_pc2_std",
            "agree_pc1_ref", "agree_pc2_ref"]
    pd.set_option("display.width", 250)
    print(df.sort_values(["lang", "seed", "K"])[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
