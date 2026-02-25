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
import numpy as np
from pathlib import Path

# ========== CONFIGURATION ==========
# Edit these variables to customize analysis

# Filter by function (None = all functions)
# Example: 'is_min_rigid' or None
FILTER_FUNCTION = 'is_min_rigid'

# X-axis for scaling analysis
# Options: 'num_nodes', 'num_edges', 'density', 'avg_degree'
X_AXIS = 'num_edges'  

# Scale type
# Options: 'linear', 'log'
X_SCALE = 'linear'  
Y_SCALE = 'log'  

# Filter by configs (None = all)
# Example: ['dim=1, algorithm=graphic'] or None
FILTER_CONFIGS = ['algorithm=graphic, dim=1', 'algorithm=randomized, dim=1']  

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]
```

## Data Preprocessing
Extracting parameters and metrics from the raw JSON data.

```{code-cell} ipython3
results_path = './benchmark_results.json'

try:
    with open(results_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data.get('benchmarks', []))} benchmark runs.")
except FileNotFoundError:
    print(f"File not found: {results_path}")
    data = {'benchmarks': []}

# Convert to DataFrame
rows = []
for b in data.get('benchmarks', []):
    params = b.get('params', {})
    config = params.get('config', {})
    graph_info = params.get('graph_info', {})
    stats = b.get('stats', {})
    
    # Create readable config label
    if config:
        config_label = ', '.join(f"{k}={v}" for k, v in sorted(config.items()))
    else:
        config_label = "Unknown Config"
    
    # Extract graph metadata
    if not graph_info and 'graph_path' in params:
         graph_name = Path(params['graph_path']).stem
         num_nodes = params.get('num_nodes', np.nan)
         num_edges = params.get('num_edges', np.nan)
    else:
        graph_name = graph_info.get('file_name', 'Unknown Graph')
        num_nodes = graph_info.get('num_nodes', np.nan)
        num_edges = graph_info.get('num_edges', np.nan)

    # Determine function name
    func_name = b.get('function', b.get('group', 'unknown'))
    
    if ':' in func_name:
        func_name = func_name.split(':')[-1]

    row = {
        'function': func_name,
        'config_label': config_label,
        'graph_name': graph_name,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'timestamp': b.get('timestamp', None),
        'mean_time': stats.get('mean', np.nan),
        'std_dev': stats.get('stddev', np.nan),
        'min_time': stats.get('min', np.nan),
        'max_time': stats.get('max', np.nan)
    }
    rows.append(row)

df = pd.DataFrame(rows)

if not df.empty:
    df['density'] = df['num_edges'] / (df['num_nodes'] * (df['num_nodes'] - 1) / 2)
    df['avg_degree'] = 2 * df['num_edges'] / df['num_nodes']
    
    print("\nAvailable Functions:", df['function'].unique())
    print("\nAvailable Configs:", df['config_label'].unique())
else:
    print("DataFrame is empty.")

df.head()
```

```{code-cell} ipython3
# Apply Filters
df_filtered = df.copy()

if FILTER_FUNCTION:
    df_filtered = df_filtered[df_filtered['function'] == FILTER_FUNCTION]
    print(f"Filtered to function: {FILTER_FUNCTION}")

if FILTER_CONFIGS:
    df_filtered = df_filtered[df_filtered['config_label'].isin(FILTER_CONFIGS)]
    print(f"Filtered to configs: {FILTER_CONFIGS}")

# Aggregate by config and graph
group_cols = ['function', 'config_label', 'graph_name', 'num_nodes', 'num_edges']
df_agg = df_filtered.groupby(group_cols).agg({
    'mean_time': 'mean',
    'std_dev': 'mean',
    'min_time': 'min',
    'max_time': 'max'
}).reset_index()

# Sort for better plotting
if X_AXIS in df_agg.columns:
    df_agg = df_agg.sort_values([X_AXIS, 'config_label'])

print(f"\nAnalyzing {len(df_agg)} unique combinations after filtering.")
df_agg.head()
```

## 1. Algorithm Comparison
Comparing mean execution time for different configurations.

```{code-cell} ipython3
if not df_agg.empty:
    plt.figure(figsize=(14, 6))
    
    # Convert config_label to string to avoid categorical issues
    df_agg['config_label'] = df_agg['config_label'].astype(str)
    
    sns.barplot(data=df_agg, x='graph_name', y='mean_time', hue='config_label')
    
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Mean Execution Time by Graph ({FILTER_FUNCTION or "All Functions"})')
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
if not df_agg.empty and X_AXIS in df_agg.columns:
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=df_agg, 
        x=X_AXIS, 
        y='mean_time', 
        hue='config_label',
        style='config_label',
        markers=True,
        dashes=False
    )
    
    plt.xscale(X_SCALE)
    plt.yscale(Y_SCALE)
    
    x_label = X_AXIS.replace('_', ' ').title()
    plt.title(f'Scaling Analysis: Time vs {x_label}')
    plt.xlabel(f'{x_label} ({X_SCALE} scale)')
    plt.ylabel(f'Time (s) ({Y_SCALE} scale)')
    
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
else:
    print(f"Skipping Scaling Analysis: Could not extract {X_AXIS}.")
```

```{code-cell} ipython3
import numpy as np

complexity_axis = 'num_nodes'

if not df_agg.empty and complexity_axis in df_agg.columns:
    print("### Empirical Complexity Analysis ###\n")
    
    def r_squared(y_true, y_pred):
        """Calculate R² score (1.0 = perfect fit)"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Filter for the specific function if set
    df_complexity = df_agg.copy()
    
    # Further aggregate by config and N
    df_complexity = df_complexity.groupby(['config_label', complexity_axis], as_index=False).agg({
        'mean_time': 'mean'
    })
    
    for config in df_complexity['config_label'].unique():
        subset = df_complexity[df_complexity['config_label'] == config].sort_values(complexity_axis)
        
        if len(subset) < 3:
            print(f"{config}: Not enough data points (need at least 3, have {len(subset)})\n")
            continue
        
        n = subset[complexity_axis].values
        t = subset['mean_time'].values
        
        # Debug: show the data points
        print(f"\n{config}: Analyzing {len(n)} data points")
        # print(f"  N: {n}")
        # print(f"  T: {t}")
        
        models = {}
        
        # Model 1: O(log n)
        try:
            log_n = np.log(n)
            c = np.polyfit(log_n, t, 1)[0]
            t_pred = c * log_n
            models['O(log n)'] = r_squared(t, t_pred)
        except Exception as e:
            pass # print(f"  O(log n) failed")
        
        # Model 2: O(n) - Linear
        try:
            c = np.polyfit(n, t, 1)[0]
            t_pred = c * n
            models['O(n)'] = r_squared(t, t_pred)
        except Exception:
            pass
        
        # Model 3: O(n log n)
        try:
            n_log_n = n * np.log(n)
            c = np.polyfit(n_log_n, t, 1)[0]
            t_pred = c * n_log_n
            models['O(n log n)'] = r_squared(t, t_pred)
        except Exception:
            pass
        
        # Model 4: O(n^k) - Polynomial (via log-log regression)
        try:
            log_n = np.log(n)
            log_t = np.log(t)
            k, log_c = np.polyfit(log_n, log_t, 1)
            t_pred = np.exp(log_c) * (n ** k)
            models[f'O(n^{k:.2f})'] = r_squared(t, t_pred)
        except Exception:
            pass
        
        # Model 5: O(2^n) - Exponential
        try:
            log_t = np.log(t)
            a, log_c = np.polyfit(n, log_t, 1)
            t_pred = np.exp(log_c) * (2 ** (a * n))
            models['O(2^n)'] = r_squared(t, t_pred)
        except Exception:
            pass
        
        if models:
            best_model = max(models, key=models.get)
            best_r2 = models[best_model]
            
            print(f"\n{config}:")
            print(f"  Best fit: {best_model} (R² = {best_r2:.3f})")
            # print(f"  All models: {models}")
            
        else:
            print(f"  No models could be fitted!\n")
else:
    print(f"Cannot perform complexity analysis: insufficient data or missing {complexity_axis}")
```

## 3. Distribution Analysis
Box plots showing the spread of execution times.

```{code-cell} ipython3
if not df_agg.empty:
    if 'std_dev' in df_agg.columns and df_agg['std_dev'].notna().any():
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
        print("Standard deviation data not available/valid for plot.")
else:
    print("No data.")
```

## 4. Relative Performance Heatmap

```{code-cell} ipython3
if not df_agg.empty:
    pivot_df = df_agg.pivot(index='graph_name', columns='config_label', values='mean_time')
    
    # Normalize by the fastest config for each graph
    pivot_normalized = pivot_df.div(pivot_df.min(axis=1), axis=0)
    
    plt.figure(figsize=(10, max(6, len(pivot_df) * 0.5)))
    
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
