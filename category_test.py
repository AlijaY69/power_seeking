"""Tests whether power-seeking rates differ between categories, per model.

Unit of analysis: the question. Each question's power-seeking rate is averaged over the
9 configurations, so repeated answers to the same question are not counted as independent.
Note: the hypothesis (categories differ) was formed in advance; this test was chosen after
the main analysis, so treat it as a follow-up check.

Run from the repo root:  python -m scripts.category_test
"""
import sys, itertools
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency

path = sys.argv[1] if len(sys.argv) > 1 else "results/samples.csv"
df = pd.read_csv(path)
df = df[df["parsed"]]  # same as summary.csv: unparsed answers are excluded
df["model"] = df["model"].str.split("/").str[-1]
print("rows", len(df), "| configs per model:", df.groupby("model")[["system_prompt","reasoning"]].apply(lambda g: len(g.drop_duplicates())).to_dict())

# Unit = question: each question's power-seeking rate across the 9 configurations
q = df.groupby(["model","sample_id","category"])["power_seeking"].mean().reset_index()
cats = ["circumventing_oversight","resource_accumulation","upward_mobility"]
pairs = list(itertools.combinations(cats, 2))
n_pair_tests = len(pairs) * q["model"].nunique()

for m, g in q.groupby("model"):
    groups = [g.loc[g.category==c,"power_seeking"] for c in cats]
    H, p = kruskal(*groups)
    print(f"\n{m}: Kruskal-Wallis H={H:.1f}, p={p:.2g}  | means: " +
          ", ".join(f"{c[:4]}={x.mean():.3f} (n={len(x)})" for c, x in zip(cats, groups)))
    for a, b in pairs:
        U, pp = mannwhitneyu(g.loc[g.category==a,"power_seeking"], g.loc[g.category==b,"power_seeking"])
        print(f"   {a[:4]} vs {b[:4]}: p={pp:.2g}, Bonferroni({n_pair_tests}) p={min(1, pp*n_pair_tests):.2g}")

# Robustness: chi-square category x choice within each single configuration
print("\nPer-configuration chi-square (category x power-seeking), Bonferroni over 27:")
for m, g in df.groupby("model"):
    sig = 0
    for _, c in g.groupby(["system_prompt","reasoning"]):
        p = chi2_contingency(pd.crosstab(c["category"], c["power_seeking"]))[1]
        sig += p * 27 < 0.05
    print(f"  {m}: {sig}/9 configurations significant")

# ARENA-style pooling: oversight + upward mobility as one 'upward-mobile' category
print("\nPooled oversight + upward mobility (ARENA-style), mean over 9 configs:")
for m, g in df.groupby("model"):
    by = g.groupby("category")["power_seeking"].mean()
    pooled = g[g.category.isin(["circumventing_oversight","upward_mobility"])]["power_seeking"].mean()
    print(f"  {m}: oversight={by['circumventing_oversight']:.3f}, upward={by['upward_mobility']:.3f}, pooled={pooled:.3f}, resources={by['resource_accumulation']:.3f}")