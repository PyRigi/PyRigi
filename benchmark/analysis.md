---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: pyrigi-Cuq7u1lD-py3.12
  language: python
  name: python3
---

# Benchmark Result Analysis

Run this notebook to visualize the results from `benchmark_results.json`.

```{code-cell} ipython3
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from pathlib import Path

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]

results_path = '../benchmark_results.json'
with open(results_path, 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data['benchmarks'])} benchmark runs.")
```

## Data Preprocessing
Extracting parameters and metrics from the raw JSON data.

```{code-cell} ipython3
rows = []

if data['benchmarks']:
    for b in data['benchmarks']:
        name = b['name']
        stats = b['stats']
        params = b.get('params', {})
        
        config = params.get('config', {})
        graph_info = params.get('graph_info', {})
        
        # Create readable config label
        if config:
            config_label = ', '.join(f"{k}={v}" for k, v in sorted(config.items()))
        else:
            config_label = "Unknown Config"
        
        # Extract graph metadata
        graph_name = graph_info.get('file_name', 'Unknown Graph')
        num_nodes = graph_info.get('num_nodes', None)
        num_edges = graph_info.get('num_edges', None)
        
        row = {
            'name': name,
            'config_label': config_label,
            'graph_name': graph_name,
            'N': num_nodes,
            'E': num_edges,
            'mean_time': stats['mean'],
            'std_dev': stats['stddev'],
            'min_time': stats['min'],
            'max_time': stats['max']
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    
    # Aggregate by config and graph (average across multiple graphs from same file)
    df_agg = df.groupby(['config_label', 'graph_name', 'N', 'E']).agg({
        'mean_time': 'mean',
        'std_dev': 'mean',
        'min_time': 'min',
        'max_time': 'max'
    }).reset_index()
    
    # Sort by N and config
    if df_agg['N'].notna().all():
        df_agg = df_agg.sort_values(['N', 'config_label'])
    
    print(f"\nAggregated to {len(df_agg)} unique (config, graph) combinations")
    display(df_agg.head(10))
else:
    print("No benchmark data found.")
    df_agg = pd.DataFrame()
```

## 1. Algorithm Comparison
Comparing mean execution time for different configurations.

```{code-cell} ipython3
if not df_agg.empty:
    plt.figure(figsize=(14, 6))
    
    sns.barplot(data=df_agg, x='graph_name', y='mean_time', hue='config_label')
    
    plt.xticks(rotation=45, ha='right')
    plt.title('Mean Execution Time by Graph')
    plt.ylabel('Time (s)')
    plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
else:
    print("No data to plot")
```

## 2. Scaling Analysis
Log-log plot to analyze time complexity (Time vs Graph Size).

```{code-cell} ipython3
if not df_agg.empty and df_agg['N'].notna().all():
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=df_agg, 
        x='N', 
        y='mean_time', 
        hue='config_label',
        style='config_label',
        markers=True, 
        dashes=False
    )
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Scaling Analysis: Performance vs Vertices')
    plt.xlabel('Number of Nodes (Log Scale)')
    plt.ylabel('Time (s) (Log Scale)')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
else:
    print("Skipping Scaling Analysis: Could not extract node counts (N).")
```

```{code-cell} ipython3
import numpy as np

if not df_agg.empty and df_agg['N'].notna().all():
    print("### Empirical Complexity Analysis ###\n")
    
    def r_squared(y_true, y_pred):
        """Calculate R² score (1.0 = perfect fit)"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Further aggregate by config and N (average across all graphs with same node count)
    df_complexity = df_agg.groupby(['config_label', 'N'], as_index=False).agg({
        'mean_time': 'mean'
    })
    
    for config in df_complexity['config_label'].unique():
        subset = df_complexity[df_complexity['config_label'] == config].sort_values('N')
        
        if len(subset) < 3:
            print(f"{config}: Not enough data points (need at least 3, have {len(subset)})\n")
            continue
        
        n = subset['N'].values
        t = subset['mean_time'].values
        
        # Debug: show the data points
        print(f"\n{config}: Analyzing {len(n)} data points")
        print(f"  N values: {n}")
        print(f"  Time values: {t}")
        
        models = {}
        
        # Model 1: O(log n)
        try:
            log_n = np.log(n)
            c = np.polyfit(log_n, t, 1)[0]
            t_pred = c * log_n
            models['O(log n)'] = r_squared(t, t_pred)
        except Exception as e:
            print(f"  O(log n) failed: {e}")
        
        # Model 2: O(n) - Linear
        try:
            c = np.polyfit(n, t, 1)[0]
            t_pred = c * n
            models['O(n)'] = r_squared(t, t_pred)
        except Exception as e:
            print(f"  O(n) failed: {e}")
        
        # Model 3: O(n log n)
        try:
            n_log_n = n * np.log(n)
            c = np.polyfit(n_log_n, t, 1)[0]
            t_pred = c * n_log_n
            models['O(n log n)'] = r_squared(t, t_pred)
        except Exception as e:
            print(f"  O(n log n) failed: {e}")
        
        # Model 4: O(n^k) - Polynomial (via log-log regression)
        try:
            log_n = np.log(n)
            log_t = np.log(t)
            k, log_c = np.polyfit(log_n, log_t, 1)
            t_pred = np.exp(log_c) * (n ** k)
            models[f'O(n^{k:.2f})'] = r_squared(t, t_pred)
        except Exception as e:
            print(f"  O(n^k) failed: {e}")
        
        # Model 5: O(2^n) - Exponential
        try:
            log_t = np.log(t)
            a, log_c = np.polyfit(n, log_t, 1)
            t_pred = np.exp(log_c) * (2 ** (a * n))
            models['O(2^n)'] = r_squared(t, t_pred)
        except Exception as e:
            print(f"  O(2^n) failed: {e}")
        
        if models:
            best_model = max(models, key=models.get)
            best_r2 = models[best_model]
            
            print(f"\n{config}:")
            print(f"  Best fit: {best_model} (R² = {best_r2:.3f})")
            print(f"  All models:")
            for model, r2 in sorted(models.items(), key=lambda x: x[1], reverse=True):
                print(f"    {model}: R² = {r2:.3f}")
            print()
        else:
            print(f"  No models could be fitted!\n")
else:
    print("Cannot perform complexity analysis: insufficient data or missing N values")
```

## 3. Distribution Analysis
Box plots showing the spread of execution times.

```{code-cell} ipython3
if not df_agg.empty:
    pivot_mean = df_agg.pivot(index='graph_name', columns='config_label', values='mean_time')
    pivot_std = df_agg.pivot(index='graph_name', columns='config_label', values='std_dev')
    
    pivot_mean.plot(
        kind='bar', 
        yerr=pivot_std, 
        capsize=4, 
        figsize=(14, 6),
        rot=45
    )
    
    plt.title('Performance Stability by Config (Mean +/- Std Dev)')
    plt.ylabel('Time (s)')
    plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
else:
    print("No data.")
```

## 4. Relative Performance Heatmap

```{code-cell} ipython3
if not df_agg.empty:
    pivot_df = df_agg.pivot(index='graph_name', columns='config_label', values='mean_time')
    pivot_normalized = pivot_df.div(pivot_df.min(axis=1), axis=0)
    
    plt.figure(figsize=(10, len(pivot_df) * 0.5 + 2))
    
    sns.heatmap(
        pivot_normalized, 
        annot=True, 
        fmt=".2f", 
        cmap="YlOrRd",
        cbar_kws={'label': 'Slowdown Factor (1.0 = Fastest)'}
    )
    
    plt.title('Relative Performance (Lower is Better)')
    plt.ylabel('Graph')
    plt.xlabel('Configuration')
    plt.tight_layout()
    plt.show()
else:
    print("No data.")
```
