---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: pyrigi-Cuq7u1lD-py3.12
  language: python
  name: python3
---

# Benchmark Result Analysis

Edit **only § 1 (Configuration)**, then `Kernel → Restart & Run All`.

---

## § 0  Imports & helpers

```{code-cell} ipython3
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.dpi": 110,
    "legend.framealpha": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def _si_time(x, _=None):
    if x == 0: return "0"
    a = abs(x)
    if a >= 1:     return f"{x:.3g} s"
    if a >= 1e-3:  return f"{x*1e3:.3g} ms"
    if a >= 1e-6:  return f"{x*1e6:.3g} µs"
    return f"{x*1e9:.3g} ns"

SI_TIME_FMT = FuncFormatter(_si_time)
```

---

## § 1  Configuration

```{code-cell} ipython3
# =====================================================================
#  CONFIGURATION — edit only this cell, then "Restart & Run All"
# =====================================================================

RESULTS_FILE = "./benchmark_results.json"

# ── Filters (None = no filter) ────────────────────────────────────────
FILTER_FUNCTION  = "is_min_rigid"   # e.g. "is_min_rigid"  or  None
FILTER_CONFIGS   = None             # e.g. ["algorithm=graphic, dim=1"]
FILTER_MIN_NODES = None             # e.g. 10
FILTER_MAX_NODES = None             # e.g. 100

# ── Plot axes ─────────────────────────────────────────────────────────
X_AXIS  = "num_nodes"   # "num_nodes"  or  "num_edges"
X_SCALE = "log"         # "linear"  or  "log"
Y_SCALE = "log"         # "linear"  or  "log"

# ── Statistics ────────────────────────────────────────────────────────
CONFIDENCE       = 0.95   # 0.90, 0.95, or 0.99
MIN_CI_INSTANCES = 3      # t-CI requires at least this many instances

# Bootstrap fallback: when n < MIN_CI_INSTANCES use bootstrap instead
USE_BOOTSTRAP = True
N_BOOTSTRAP   = 2000

# ── Comparison chart (§ 7) ────────────────────────────────────────────
# None → auto-pick up to 4 representative sizes from the data
COMPARISON_SIZES = None   # e.g. [10, 30, 60]

# ── Figure export ─────────────────────────────────────────────────────
SAVE_FIGURES  = False        # True → save to ./figures/
FIGURE_FORMAT = "pdf"        # "png", "pdf", or "svg"
_FIG_DIR      = Path("./figures")

def _savefig(fig, name):
    if SAVE_FIGURES:
        _FIG_DIR.mkdir(exist_ok=True)
        fig.savefig(_FIG_DIR / f"{name}.{FIGURE_FORMAT}", bbox_inches="tight")
```

---

## § 2  Data loading

```{code-cell} ipython3
def _load(path):
    with open(path) as f:
        raw = json.load(f)
    rows = []
    for b in raw.get("benchmarks", []):
        if b.get("timed_out"):          # skip timeout markers — no stats to analyse
            continue
        params = b.get("params", {})
        cfg    = params.get("config", {})
        gi     = params.get("graph_info", {})
        st     = b.get("stats", {})
        fn     = b.get("function", b.get("group", "unknown"))
        if ":" in fn:
            fn = fn.rsplit(":", 1)[-1]
        rows.append({
            "function":     fn,
            "config_label": (", ".join(f"{k}={v}" for k, v in sorted(cfg.items()))
                             if cfg else "unknown"),
            "graph_name":   gi.get("file_name", "unknown"),
            "graph_idx":    gi.get("idx",        np.nan),
            "num_nodes":    gi.get("num_nodes",  np.nan),
            "num_edges":    gi.get("num_edges",  np.nan),
            "mean_time":    st.get("mean",       np.nan),
            "std_dev":      st.get("stddev",     np.nan),
            "rounds":       st.get("rounds",     np.nan),
            "min_time":     st.get("min",        np.nan),
            "max_time":     st.get("max",        np.nan),
            "iqr":          st.get("iqr",        np.nan),
            "source_hash":  b.get("source_hash", None),
            "timestamp":    b.get("timestamp",   None),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        denom            = df["num_nodes"] * (df["num_nodes"] - 1) / 2
        df["density"]    = df["num_edges"] / denom
        df["avg_degree"] = 2 * df["num_edges"] / df["num_nodes"]
        df["cv"]         = df["std_dev"] / df["mean_time"]
    return df

df_all = _load(RESULTS_FILE)
print(f"✓  Loaded {len(df_all):,} benchmark entries.")
print(f"\nFunctions : {sorted(df_all['function'].unique())}")
print(f"Configs   : {sorted(df_all['config_label'].unique())}")
```

### Staleness check

```{code-cell} ipython3
def _check_staleness(df):
    for fn in df["function"].unique():
        hashes = df.loc[df["function"] == fn, "source_hash"].dropna().unique()
        if len(hashes) > 1:
            warnings.warn(
                f"'{fn}' has {len(hashes)} different source versions in results. "
                f"Run check_staleness.py or --force-rerun to clean up.",
                stacklevel=2,
            )
        elif len(hashes) == 0:
            print(f"'{fn}': no source_hash stored — staleness unknown.")

_check_staleness(df_all)
```

---

## § 3  Data overview

Sample counts per (config × graph size).  CI bands require ≥ **MIN_CI_INSTANCES** instances.

```{code-cell} ipython3
count_piv = (
    df_all.groupby(["config_label", "num_nodes"])
          .size()
          .unstack(fill_value=0)
)
display(
    count_piv.style
        .background_gradient(cmap="Blues", axis=None)
        .format("{:d}")
        .set_caption(
            f"Sample counts per (config × node count)  |  "
            f"CI drawn when count ≥ {MIN_CI_INSTANCES}"
        )
)
```

---

## § 4  Filter & data quality

```{code-cell} ipython3
def _filter(df):
    d = df.copy()
    if FILTER_FUNCTION:  d = d[d["function"]     == FILTER_FUNCTION]
    if FILTER_CONFIGS:   d = d[d["config_label"].isin(FILTER_CONFIGS)]
    if FILTER_MIN_NODES: d = d[d["num_nodes"]    >= FILTER_MIN_NODES]
    if FILTER_MAX_NODES: d = d[d["num_nodes"]    <= FILTER_MAX_NODES]
    print(f"{len(d):,} entries after filtering  ({len(df)-len(d):,} removed).")
    if d.empty:
        raise ValueError("No data after filtering — check FILTER_* settings.")
    return d.reset_index(drop=True)

df = _filter(df_all)

# Consistent colour per config (used in every plot)
_cfgs    = sorted(df["config_label"].unique())
_pal     = sns.color_palette("tab10", len(_cfgs))
CONFIG_COLORS = dict(zip(_cfgs, _pal))
```

### Coefficient of Variation (CV = σ/μ)

Green ≤ 0.05 → stable measurement.  Red > 0.05 → noisy; interpret with caution.

```{code-cell} ipython3
cv_piv = (
    df.groupby(["config_label", "num_nodes"])["cv"]
      .mean()
      .unstack()
)
noisy = int((cv_piv > 0.05).sum().sum())
if noisy:
    print(f"{noisy} cell(s) have CV > 0.05.")

display(
    cv_piv.style
        .format("{:.3f}")
        .background_gradient(cmap="RdYlGn_r", vmin=0, vmax=0.10, axis=None)
        .set_caption("CV per (config × node count)  |  green ≤ 0.05  red > 0.05")
)
```

---

## § 5  Scaling analysis

For each (config, size) the shaded band is the **95 % Student-t CI on the mean**
across graph instances at that size.  When fewer than `MIN_CI_INSTANCES` instances
are available a lighter **bootstrap CI** is shown instead.
Dashed lines show the fitted power-law curve `T(n) = a · nᵇ`.

```{code-cell} ipython3
# ── CI helpers ────────────────────────────────────────────────────────────

def _tci(times, conf):
    n, mu = len(times), times.mean()
    se    = times.std(ddof=1) / np.sqrt(n)
    t     = stats.t.ppf((1 + conf) / 2, df=n - 1)
    return mu, mu - t * se, mu + t * se

def _bootci(times, n_boot, conf):
    rng   = np.random.default_rng(42)
    boots = rng.choice(times, size=(n_boot, len(times)), replace=True).mean(axis=1)
    a     = (1 - conf) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    return times.mean(), lo, hi

def _agg(df, x_col, conf, min_ci, use_boot, n_boot):
    rows = []
    for (cfg, xv), grp in df.groupby(["config_label", x_col]):
        times = grp["mean_time"].dropna().values
        n = len(times)
        if n == 0:
            continue
        if n >= min_ci:
            mu, lo, hi, src = *_tci(times, conf), "t"
        elif n >= 2 and use_boot:
            mu, lo, hi, src = *_bootci(times, n_boot, conf), "boot"
        elif n == 1:
            mu, lo, hi, src = times[0], np.nan, np.nan, "none"
        else:
            mu, lo, hi, src = times.mean(), np.nan, np.nan, "none"
        rows.append({"config_label": cfg, x_col: xv,
                     "mu": mu, "ci_lo": lo, "ci_hi": hi, "n": n, "src": src})
    return pd.DataFrame(rows).sort_values([x_col, "config_label"]).reset_index(drop=True)

# ── Power-law fit  T(n) = a · n^b ─────────────────────────────────────────

def _fit(stats_df, x_col):
    rows = []
    for cfg, grp in stats_df.groupby("config_label"):
        g = grp.dropna(subset=[x_col, "mu"]).sort_values(x_col)
        if len(g) < 4:
            rows.append({"config_label": cfg, "a": np.nan, "a_std": np.nan,
                         "b": np.nan, "b_std": np.nan})
            continue
        x, y = g[x_col].values.astype(float), g["mu"].values
        try:
            popt, pcov = curve_fit(lambda n, a, b: a * np.power(n, b),
                                   x, y, p0=[1e-6, 2.0], maxfev=10_000)
            perr = np.sqrt(np.diag(pcov))
            rows.append({"config_label": cfg,
                         "a": popt[0], "a_std": perr[0],
                         "b": popt[1], "b_std": perr[1]})
        except RuntimeError:
            rows.append({"config_label": cfg, "a": np.nan, "a_std": np.nan,
                         "b": np.nan, "b_std": np.nan})
    return pd.DataFrame(rows)

# ── Compute & plot ─────────────────────────────────────────────────────────

stats_df = _agg(df, X_AXIS, CONFIDENCE, MIN_CI_INSTANCES, USE_BOOTSTRAP, N_BOOTSTRAP)
fit_df   = _fit(stats_df, X_AXIS)

fig, ax = plt.subplots(figsize=(12, 6))
used_boot = False

for cfg, grp in stats_df.groupby("config_label"):
    color = CONFIG_COLORS[cfg]
    grp   = grp.sort_values(X_AXIS)

    ax.plot(grp[X_AXIS], grp["mu"], marker="o", ms=5,
            color=color, label=cfg, zorder=3)

    for mask, alpha, label in [
        (grp["src"] == "t",    0.22, None),
        (grp["src"] == "boot", 0.11, None),
    ]:
        sel = grp[mask & grp["ci_lo"].notna()]
        if not sel.empty:
            ax.fill_between(sel[X_AXIS], sel["ci_lo"], sel["ci_hi"],
                            color=color, alpha=alpha, linewidth=0)
            if grp["src"].eq("boot").any():
                used_boot = True

    fr = fit_df[fit_df["config_label"] == cfg]
    if not fr.empty and not np.isnan(fr["b"].values[0]):
        a, b  = fr["a"].values[0], fr["b"].values[0]
        x_fit = np.linspace(grp[X_AXIS].min(), grp[X_AXIS].max(), 300)
        ax.plot(x_fit, a * x_fit**b, "--", color=color, alpha=0.7,
                label=f"  fit {a:.2e}·n^{b:.2f}")

ax.set_xscale(X_SCALE)
ax.set_yscale(Y_SCALE)
ax.yaxis.set_major_formatter(SI_TIME_FMT)
ax.set_xlabel({"num_nodes": "Vertex count  n",
               "num_edges": "Edge count  m"}.get(X_AXIS, X_AXIS))
ax.set_ylabel("Execution time")
ax.set_title(
    f"Scaling analysis: {FILTER_FUNCTION or 'all functions'}\n"
    f"Band = {int(CONFIDENCE*100)} % CI across instances (excluding timeout instances)"
    + ("  (light = bootstrap)" if used_boot else "")
)
ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
ax.legend(title="Config / fit", bbox_to_anchor=(1.02, 1), loc="upper left",
          fontsize=10, title_fontsize=10)
fig.tight_layout()
_savefig(fig, "01_scaling")
plt.show()
```

---

## § 6  Power-law fit summary  `T(n) = a · nᵇ`

```{code-cell} ipython3
if fit_df.empty or fit_df["b"].isna().all():
    print("Not enough data for fitting (need ≥ 4 distinct sizes per config).")
else:
    print(f"{'Config':<45}  {'a':>12}  {'b':>8}  {'±2σ(b)':>8}  Implied")
    print("─" * 88)
    for _, row in fit_df.iterrows():
        if np.isnan(row["b"]):
            print(f"  {row['config_label']:<43}  (fit failed)")
            continue
        print(
            f"  {row['config_label']:<43}  {row['a']:>12.3e}"
            f"  {row['b']:>8.3f}  {row['b_std']*2:>8.3f}  O(n^{row['b']:.2f})"
        )
    print("─" * 88)
    print("\n— LaTeX —")
    for _, row in fit_df.iterrows():
        if not np.isnan(row["b"]):
            print(f"  {row['config_label']}: "
                  f"$T(n) \\propto n^{{{row['b']:.2f} \\pm {row['b_std']*2:.2f}}}$")
```

---

## § 7  Algorithm comparison at specific sizes

Side-by-side 95 % CI error bars.  Non-overlapping CIs → statistically distinguishable.
A Welch *t*-test *p*-value table is shown below.

```{code-cell} ipython3
_cfgs_list = sorted(df["config_label"].unique())

if len(_cfgs_list) < 2:
    print("Only one config present — comparison section skipped.")
else:
    _all_sizes = sorted(stats_df[X_AXIS].dropna().unique())
    if COMPARISON_SIZES is not None:
        _sizes = [s for s in COMPARISON_SIZES if s in _all_sizes]
    else:
        _idx   = np.round(np.linspace(0, len(_all_sizes)-1,
                                      min(4, len(_all_sizes)))).astype(int)
        _sizes = [_all_sizes[i] for i in _idx]

    sub    = stats_df[stats_df[X_AXIS].isin(_sizes)]
    n_cfg  = len(_cfgs_list)
    width  = 0.8 / n_cfg
    x_pos  = np.arange(len(_sizes))

    fig, ax = plt.subplots(figsize=(max(10, len(_sizes)*2.5), 6))

    for i, cfg in enumerate(_cfgs_list):
        color = CONFIG_COLORS[cfg]
        csub  = sub[sub["config_label"] == cfg].set_index(X_AXIS)
        mus   = [csub.loc[s, "mu"]    if s in csub.index else np.nan for s in _sizes]
        lo_e  = [csub.loc[s, "mu"] - csub.loc[s, "ci_lo"]
                 if s in csub.index and not np.isnan(csub.loc[s, "ci_lo"]) else 0
                 for s in _sizes]
        hi_e  = [csub.loc[s, "ci_hi"] - csub.loc[s, "mu"]
                 if s in csub.index and not np.isnan(csub.loc[s, "ci_hi"]) else 0
                 for s in _sizes]
        off   = (i - (n_cfg-1)/2) * width
        ax.bar(x_pos + off, mus, width=width*0.9, color=color, alpha=0.85,
               label=cfg, yerr=[lo_e, hi_e],
               error_kw={"capsize": 4, "elinewidth": 1.2})

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"n={s}" for s in _sizes])
    ax.yaxis.set_major_formatter(SI_TIME_FMT)
    ax.set_yscale(Y_SCALE)
    ax.set_ylabel("Execution time")
    ax.set_xlabel("Graph size")
    ax.set_title(
        f"Algorithm comparison — {FILTER_FUNCTION or 'all functions'}\n"
        f"Error bars = {int(CONFIDENCE*100)} % CI  (non-overlapping → distinguishable)"
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    _savefig(fig, "02_comparison")
    plt.show()

    print(f"\nWelch's t-test  (p < 0.05 = ✓ significant)")
    print("─" * 70)
    for size in _sizes:
        size_data = {
            cfg: df[(df["config_label"] == cfg) & (df[X_AXIS] == size)
                    ]["mean_time"].dropna().values
            for cfg in _cfgs_list
        }
        size_data = {k: v for k, v in size_data.items() if len(v) >= 2}
        if len(size_data) < 2:
            continue
        print(f"\n  n = {size}")
        for c1, c2 in combinations(size_data, 2):
            _, p = stats.ttest_ind(size_data[c1], size_data[c2], equal_var=False)
            mark = " ✓" if p < 0.05 else ""
            print(f"    {c1}  vs  {c2}:  p = {p:.4f}{mark}")
```

---

## § 8  Relative performance heatmap

Each cell = slowdown factor vs the fastest config **at that graph size** (1.00 = fastest).

```{code-cell} ipython3
pivot = stats_df.pivot_table(
    index="config_label", columns=X_AXIS, values="mu", aggfunc="first"
)

if pivot.empty or pivot.shape[0] < 2:
    print("Need ≥ 2 configs to build the heatmap.")
else:
    normalized = pivot.div(pivot.min(axis=0), axis=1)
    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns)*0.9), max(4, len(pivot)*0.8))
    )
    sns.heatmap(
        normalized, annot=True, fmt=".2f",
        cmap="YlOrRd", linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Slowdown factor  (1.00 = fastest at this size)"},
        ax=ax,
    )
    ax.set_title(
        f"Relative performance — {FILTER_FUNCTION or 'all functions'}\n"
        f"Lower is better"
    )
    ax.set_xlabel({"num_nodes": "Vertex count  n",
                   "num_edges": "Edge count  m"}.get(X_AXIS, X_AXIS))
    ax.set_ylabel("Configuration")
    fig.tight_layout()
    _savefig(fig, "03_heatmap")
    plt.show()
```

---

## § 9  Variability analysis

### 9a — CV heatmap (filtered data)

```{code-cell} ipython3
cv_filt = (
    df.groupby(["config_label", "num_nodes"])["cv"]
      .mean()
      .unstack()
)
fig, ax = plt.subplots(
    figsize=(max(10, len(cv_filt.columns)*0.9), max(4, len(cv_filt)*0.8))
)
sns.heatmap(
    cv_filt, annot=True, fmt=".3f",
    cmap="RdYlGn_r", vmin=0, vmax=0.10,
    linewidths=0.5, linecolor="white",
    cbar_kws={"label": "Mean CV = σ/μ  (≤ 0.05 = stable)"},
    ax=ax,
)
ax.set_title("Measurement stability (CV) — filtered data")
ax.set_xlabel({"num_nodes": "Vertex count  n",
               "num_edges": "Edge count  m"}.get(X_AXIS, X_AXIS))
ax.set_ylabel("Configuration")
fig.tight_layout()
_savefig(fig, "04_cv_heatmap")
plt.show()
```

### 9b — Distribution of instance timings (box plots)

Each box shows how `mean_time` varies across different graph instances at the same
size — capturing sensitivity to graph structure, not just timing noise.

```{code-cell} ipython3
_cfgs_var = sorted(df["config_label"].unique())
n_row = (len(_cfgs_var) + 1) // 2
n_col = min(2, len(_cfgs_var))

fig, axes = plt.subplots(n_row, n_col,
                          figsize=(n_col*7, n_row*4),
                          sharey=True, squeeze=False)

for ax, cfg in zip(axes.flat, _cfgs_var):
    sub   = df[df["config_label"] == cfg].copy()
    sub["sz"] = sub["num_nodes"].astype(int).astype(str)
    order = [str(s) for s in sorted(sub["num_nodes"].dropna().unique().astype(int))]
    sns.boxplot(data=sub, x="sz", y="mean_time", order=order,
                color=CONFIG_COLORS[cfg], ax=ax,
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.5})
    ax.set_title(cfg)
    ax.yaxis.set_major_formatter(SI_TIME_FMT)
    ax.set_yscale(Y_SCALE)
    ax.set_xlabel("Vertex count  n")
    ax.set_ylabel("mean_time per instance")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

for ax in axes.flat[len(_cfgs_var):]:
    ax.set_visible(False)

fig.suptitle(
    f"Instance-level timing distribution — {FILTER_FUNCTION or 'all functions'}\n"
    f"Box = IQR · whiskers = 1.5×IQR · dots = outliers",
    fontsize=12, y=1.01,
)
fig.tight_layout()
_savefig(fig, "05_boxplots")
plt.show()
```

---

## § 10  Timeout rate per (config × size)

Timed-out instances are excluded from every section above (they carry no timing
data).  This section is the only place they appear.  For each (config, size) the
rate is computed over **attempted** instances only — completed plus timed out —
so a value of 100 % means every graph that was run at that size exceeded the
budget.  A config disappears from the plot at the size where early-stop halted
it, since larger sizes were never attempted.

```{code-cell} ipython3
EARLY_STOP_THRESHOLD = 0.75   # matches benchmark config; reference line only

def _load_timeout_stats(path):
    with open(path) as f:
        raw = json.load(f)
    rows = []
    for b in raw.get("benchmarks", []):
        cfg = b.get("params", {}).get("config", {})
        gi  = b.get("params", {}).get("graph_info", {})
        fn  = b.get("function", "unknown")
        if ":" in fn:
            fn = fn.rsplit(":", 1)[-1]
        rows.append({
            "function":     fn,
            "config_label": (", ".join(f"{k}={v}" for k, v in sorted(cfg.items()))
                             if cfg else "unknown"),
            "num_nodes":    gi.get("num_nodes", np.nan),
            "timed_out":    bool(b.get("timed_out", False)),
        })
    return pd.DataFrame(rows)

to_df = _load_timeout_stats(RESULTS_FILE)
if FILTER_FUNCTION:
    to_df = to_df[to_df["function"] == FILTER_FUNCTION]
if FILTER_CONFIGS:
    to_df = to_df[to_df["config_label"].isin(FILTER_CONFIGS)]
if FILTER_MIN_NODES:
    to_df = to_df[to_df["num_nodes"] >= FILTER_MIN_NODES]
if FILTER_MAX_NODES:
    to_df = to_df[to_df["num_nodes"] <= FILTER_MAX_NODES]

if to_df.empty:
    print("No data available for timeout analysis.")
else:
    grp = (to_df.groupby(["config_label", "num_nodes"])
                .agg(attempted=("timed_out", "size"),
                     timeouts =("timed_out", "sum")))
    grp["pct"] = 100 * grp["timeouts"] / grp["attempted"]
    grp = grp.reset_index()

    total_to = int(grp["timeouts"].sum())
    if total_to == 0:
        print("No timed-out instances recorded for the current filter.")
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        for cfg, g in grp.groupby("config_label"):
            g = g.sort_values("num_nodes")
            ax.plot(g["num_nodes"], g["pct"], marker="o", ms=4,
                    label=cfg, color=CONFIG_COLORS.get(cfg))
        ax.axhline(EARLY_STOP_THRESHOLD * 100, ls="--", color="grey", lw=1,
                   label=f"early-stop threshold ({EARLY_STOP_THRESHOLD*100:.0f}%)")
        ax.set_xscale(X_SCALE)
        ax.set_xlabel({"num_nodes": "Vertex count  n",
                       "num_edges": "Edge count  m"}.get(X_AXIS, "Vertex count  n"))
        ax.set_ylabel("Timed-out instances (%)")
        ax.set_ylim(-2, 102)
        ax.set_title(
            f"Timeout rate per config — {FILTER_FUNCTION or 'all functions'}\n"
            f"Denominator = attempted instances (completed + timed out)"
        )
        ax.grid(True, ls="--", lw=0.4, alpha=0.5)
        ax.legend(title="Config", bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=10, title_fontsize=10)
        fig.tight_layout()
        _savefig(fig, "06_timeout_rate")
        plt.show()
```
