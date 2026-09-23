"""Source for notebooks/01_eda.ipynb.

Kept as a plain .py so every cell is syntax-checked in CI and diffs are
readable. `python -m scripts.build_notebook` regenerates the .ipynb from it.
Cells are separated by the `# %%` marker; a cell starting `# %% [markdown]`
becomes a markdown cell from its comment text.
"""

# %% [markdown]
# # Telco customer churn — exploratory analysis
#
# The operator loses roughly a quarter of its subscriber base. This notebook
# works out *who* leaves and *why*, and turns that into things the retention
# team can actually do.
#
# Read the modelling decisions here as inputs to `src/features/engineering.py` —
# every derived feature in the pipeline traces back to something on this page.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ADDON_COLUMNS, RAW_CSV, TARGET  # noqa: E402
from src.data.preprocess import clean  # noqa: E402

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.max_columns", 30)

# %% [markdown]
# ## 1. Load
#
# Reusing `clean()` from the pipeline rather than re-implementing it here. If
# the notebook and the pipeline disagreed about what "clean" means, the
# notebook's conclusions would not describe the data the model actually sees.

# %%
if not RAW_CSV.exists():
    raise FileNotFoundError(
        f"{RAW_CSV} not found. Run `make data` (or `python -m src.data.download`) first."
    )

raw = pd.read_csv(RAW_CSV)
df = clean(raw)

print(f"{raw.shape[0]:,} customers x {raw.shape[1]} raw columns")
print(f"Churn rate: {df[TARGET].mean():.1%}")
df.head()

# %% [markdown]
# ### Data quality
#
# One quirk matters. `TotalCharges` ships as text, and 11 rows hold a blank
# string. Every one of them is a customer with `tenure == 0` — they have signed
# up but have not been billed yet, so their true lifetime spend is 0, not the
# column median. Imputing the median here would invent ~£1,400 of history for
# the newest customers in the base, who are also the highest-churn-risk group.
# The pipeline reconstructs the value as `MonthlyCharges * tenure`.

# %%
blank_total = raw["TotalCharges"].astype(str).str.strip() == ""
print(f"Blank TotalCharges: {blank_total.sum()}")
print(f"...all with tenure == 0: {bool((raw.loc[blank_total, 'tenure'] == 0).all())}")
# TotalCharges NaNs are intentional (see above); nothing else should be missing.
other = df.drop(columns=[TARGET, "TotalCharges"])
print(f"\nMissing values in every other column: {other.isna().sum().sum()}")
print(f"Duplicate customer IDs: {raw['customerID'].duplicated().sum()}")

# %% [markdown]
# ## 2. How imbalanced is the target?
#
# About 27% positive. Not severe, but severe enough that accuracy is a useless
# metric: predicting "nobody churns" scores ~73%. Everything downstream is
# reported on PR-AUC, and the decision threshold is tuned on expected value
# rather than left at 0.5.

# %%
counts = df[TARGET].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.bar(["Stayed", "Churned"], counts.values, color=["#4c72b0", "#c44e52"])
for i, v in enumerate(counts.values):
    ax.text(i, v + 60, f"{v:,}\n({v / len(df):.1%})", ha="center", fontsize=9)
ax.set_ylabel("Customers")
ax.set_title("Churn is the minority class")
ax.set_ylim(0, counts.max() * 1.18)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Tenure is the dominant signal
#
# Churn hazard is not linear in tenure — it is front-loaded. The first year is
# where the base leaks, and it flattens out afterwards. That non-linearity is
# why the pipeline adds a binned `tenure_bucket` alongside raw tenure: a linear
# model cannot express "steep then flat" from the raw column alone.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

for label, colour in [(0, "#4c72b0"), (1, "#c44e52")]:
    axes[0].hist(
        df.loc[df[TARGET] == label, "tenure"],
        bins=36, alpha=0.65, color=colour,
        label="Churned" if label else "Stayed",
    )
axes[0].set_xlabel("Tenure (months)")
axes[0].set_ylabel("Customers")
axes[0].set_title("Churners are concentrated in month 0–6")
axes[0].legend()

bucket = pd.cut(
    df["tenure"],
    bins=[-0.1, 6, 12, 24, 48, 60, np.inf],
    labels=["0-6m", "6-12m", "1-2y", "2-4y", "4-5y", "5y+"],
)
rate = df.groupby(bucket, observed=True)[TARGET].mean()
axes[1].plot(rate.index.astype(str), rate.values, marker="o", color="#c44e52")
axes[1].set_ylabel("Churn rate")
axes[1].set_xlabel("Tenure bucket")
axes[1].set_title("Hazard is front-loaded, then flattens")
axes[1].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
plt.tight_layout()
plt.show()

rate.to_frame("churn_rate").style.format("{:.1%}")

# %% [markdown]
# ## 4. Contract type is the single strongest categorical driver

# %%
cat_cols = [
    "Contract", "InternetService", "PaymentMethod", "OnlineSecurity",
    "TechSupport", "PaperlessBilling", "Dependents", "SeniorCitizen",
]
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
overall = df[TARGET].mean()

for ax, col in zip(axes.ravel(), cat_cols, strict=True):
    rates = df.groupby(col, observed=True)[TARGET].mean().sort_values()
    ax.barh(rates.index.astype(str), rates.values, color="#55a868")
    ax.axvline(overall, color="#c44e52", ls="--", lw=1)
    ax.set_title(col, fontsize=10)
    ax.set_xlim(0, max(0.5, rates.max() * 1.15))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.tick_params(labelsize=8)

fig.suptitle("Churn rate by category (dashed line = base rate)", y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# The spread within `Contract` is enormous:

# %%
contract = (
    df.groupby("Contract", observed=True)
    .agg(customers=(TARGET, "size"), churn_rate=(TARGET, "mean"))
    .sort_values("churn_rate", ascending=False)
)
contract["lift_vs_base"] = contract["churn_rate"] / overall
contract.style.format({"churn_rate": "{:.1%}", "lift_vs_base": "{:.2f}x", "customers": "{:,}"})

# %% [markdown]
# ## 5. The add-on effect
#
# Each optional service a customer holds makes them measurably stickier. This
# is the motivation for the `num_addons` feature: six separate one-hot columns
# force the model to learn the pattern six times, whereas one count column
# states it directly.
#
# Read this as correlation, not proven causation — customers who buy more
# add-ons may simply be more committed to begin with. It is still a useful
# *predictor*, and a cheap A/B test (free tech support for 3 months to
# at-risk month-to-month customers) would settle whether it is causal.

# %%
addons = (df[ADDON_COLUMNS] == "Yes").sum(axis=1)
by_addons = df.groupby(addons, observed=True)[TARGET].agg(["size", "mean"])
by_addons.columns = ["customers", "churn_rate"]

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.bar(by_addons.index, by_addons["churn_rate"], color="#4c72b0")
ax.set_xlabel("Number of add-on services held")
ax.set_ylabel("Churn rate")
ax.set_title("More add-ons, less churn")
ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
for i, (n, r) in enumerate(zip(by_addons["customers"], by_addons["churn_rate"], strict=True)):
    ax.text(by_addons.index[i], r + 0.012, f"n={n:,}", ha="center", fontsize=7)
plt.tight_layout()
plt.show()

by_addons.style.format({"customers": "{:,}", "churn_rate": "{:.1%}"})

# %% [markdown]
# ## 6. Charges, and the derived feature they motivate
#
# `MonthlyCharges` is bimodal — the second hump is the fiber-optic cohort, who
# both pay more and churn more. But the raw column hides something: a
# customer's *current* charge tells you nothing about whether it recently went
# up. Dividing lifetime spend by tenure recovers their historical average, and
# the ratio between the two (`charge_drift` in the pipeline) flags customers
# whose bill has risen.

# %%
work = df.copy()
work["avg_monthly_spend"] = work["TotalCharges"].fillna(0) / work["tenure"].clip(lower=1)
work["charge_drift"] = work["MonthlyCharges"] / work["avg_monthly_spend"].replace(0, np.nan)
work["charge_drift"] = work["charge_drift"].fillna(1.0)

fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))

for label, colour in [(0, "#4c72b0"), (1, "#c44e52")]:
    axes[0].hist(
        work.loc[work[TARGET] == label, "MonthlyCharges"],
        bins=40, alpha=0.6, color=colour,
        label="Churned" if label else "Stayed",
    )
axes[0].set_title("MonthlyCharges is bimodal")
axes[0].set_xlabel("Monthly charge")
axes[0].legend(fontsize=8)

sns.boxplot(data=work, x=TARGET, y="charge_drift", ax=axes[1],
            palette=["#4c72b0", "#c44e52"], hue=TARGET, legend=False)
axes[1].set_ylim(0.6, 2.0)
axes[1].set_xticks([0, 1], ["Stayed", "Churned"])
axes[1].set_xlabel("")
axes[1].set_title("Churners' bills drifted up more")

drift_band = pd.cut(work["charge_drift"], [0, 0.95, 1.05, 1.25, np.inf],
                    labels=["fell", "flat", "up <25%", "up >25%"])
dr = work.groupby(drift_band, observed=True)[TARGET].mean()
axes[2].bar(dr.index.astype(str), dr.values, color="#dd8452")
axes[2].set_title("Churn rate by bill movement")
axes[2].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Correlation check
#
# `TotalCharges` is almost a product of tenure and monthly charge, so it is
# heavily collinear with both. That is fine for tree models and survivable for
# regularised logistic regression, but it is the reason the pipeline scales
# numerics and uses an L2 penalty rather than reading raw coefficients as
# causal effects.

# %%
numeric = work[["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", TARGET]]
corr = numeric.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(5.6, 4.4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Numeric correlations")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Where the money is
#
# Churn rate alone does not tell you where to spend the retention budget — a
# high-churn segment of 200 customers matters less than a medium-churn segment
# of 2,000. Ranking by *monthly revenue at risk* is what a retention manager
# actually needs.

# %%
segments = (
    df.assign(segment=df["Contract"] + " / " + df["InternetService"])
    .groupby("segment", observed=True)
    .agg(
        customers=(TARGET, "size"),
        churn_rate=(TARGET, "mean"),
        avg_monthly=("MonthlyCharges", "mean"),
    )
)
segments["revenue_at_risk"] = (
    segments["customers"] * segments["churn_rate"] * segments["avg_monthly"]
)
segments = segments.sort_values("revenue_at_risk", ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
top = segments.head(8).iloc[::-1]
ax.barh(top.index, top["revenue_at_risk"], color="#c44e52")
ax.set_xlabel("Monthly revenue at risk ($)")
ax.set_title("Where churn actually costs money")
plt.tight_layout()
plt.show()

segments.style.format({
    "customers": "{:,}", "churn_rate": "{:.1%}",
    "avg_monthly": "${:.2f}", "revenue_at_risk": "${:,.0f}",
})

# %% [markdown]
# ## 9. What this means for the business
#
# **1. Month-to-month fiber customers in their first year are the whole
# problem.** They are the largest single block of revenue at risk. A targeted
# contract-upgrade offer aimed at this cohort in months 2–6 addresses more
# lost revenue than any other single intervention.
#
# **2. Contract length is the lever with the most headroom.** Two-year
# customers churn at a fraction of the month-to-month rate. Even allowing that
# committed customers self-select into long contracts, the gap is large enough
# that moving a slice of the month-to-month base onto annual terms is worth
# testing properly.
#
# **3. Tech support and online security look like retention products, not just
# revenue products.** Customers without them churn far more. Bundling them free
# for the first year into at-risk segments is a cheap experiment with a clear
# readout.
#
# **4. Electronic check payers churn more than any other payment method.**
# Likely a proxy for lower commitment rather than a cause, but it is a free
# targeting signal — no new data collection needed.
#
# **5. Watch bills that move.** Customers whose current charge exceeds their
# historical average churn more. A simple alert on bill increases above a
# threshold would catch a slice of this before they leave.
#
# ### What this changes downstream
#
# | Finding | Feature it motivates |
# |---|---|
# | Front-loaded hazard | `tenure_bucket` |
# | Add-ons are sticky | `num_addons` |
# | Bill increases predict exit | `charge_drift` |
# | Plan changes hide in TotalCharges | `avg_monthly_spend` |
# | ~27% positive class | PR-AUC over accuracy; cost-tuned threshold |
#
# ### Caveats worth stating
#
# The dataset is a single snapshot with no timestamps, so there is no way to
# build a temporal validation split — the model is validated on a random split,
# which is optimistic relative to production. Everything above is association,
# not causation; the interventions are hypotheses to A/B test, not conclusions.
