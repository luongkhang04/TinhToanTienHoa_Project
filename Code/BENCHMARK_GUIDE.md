# Benchmark Guide

This guide explains how to run comprehensive benchmarks on all combinations of planning algorithms.

## Overview

The benchmark system tests all 16 combinations of:
- **4 Global Planning Algorithms**: grid_bfs, grid_dfs, astar, dijkstra
- **4 Local Planning Algorithms**: reactive_bfs, reactive_dfs, potential_field, greedy

Across:
- **7 maps** (1.png to 7.png with corresponding 1.txt to 7.txt for start/goal positions)
- **4 obstacle counts**: 0, 50, 100, 200 dynamic obstacles

**Total: 448 test configurations** (4 × 4 × 7 × 4)

## Evaluation Metrics

Each test measures:
1. **Path Length**: Total number of cells in the path
2. **Direction Changes**: Number of times the agent changes direction
3. **Planning Time**: Total execution time (seconds)
4. **Success Rate**: Whether the agent reached the goal

## Files

- `benchmark.py` - Main benchmark script that runs all tests
- `analyze_results.py` - Analysis and visualization script
- `benchmark_results.csv` - Output CSV with all test results
- `BENCHMARK_GUIDE.md` - This file

## Quick Start

### 1. Run Full Benchmark (All 448 tests)

```bash
python benchmark.py
```

This will:
- Test all 16 algorithm combinations
- On all 7 maps
- With 0, 50, 100, and 200 obstacles
- Save results to `benchmark_results.csv`
- Print a summary

**Note**: This will take a significant amount of time (estimated 30-60 minutes depending on your hardware).

### 2. Run Partial Benchmark

Test specific maps:
```bash
python benchmark.py -m 1 2 3
```

Test specific obstacle counts:
```bash
python benchmark.py -o 0 50
```

Combine both:
```bash
python benchmark.py -m 1 2 -o 0 100
```

### 3. Customize Settings

```bash
python benchmark.py --max-steps 3000 --output my_results.csv
```

Options:
- `-m, --maps`: Map numbers to test (default: 1-7)
- `-o, --obstacles`: Obstacle counts (default: 0 50 100 200)
- `-s, --max-steps`: Maximum steps per test (default: 2000)
- `--output`: Output CSV filename (default: benchmark_results.csv)

### 4. Analyze Results

After running the benchmark:

```bash
python analyze_results.py
```

This generates:
- `success_rates.png` - Bar chart of success rates by algorithm combination
- `metrics_comparison.png` - Comparison of path length, direction changes, and time
- `obstacle_impact.png` - How obstacle count affects performance
- `algorithm_heatmap.png` - Heatmap showing algorithm performance
- `benchmark_report.txt` - Detailed text report

With custom input:
```bash
python analyze_results.py --input my_results.csv --output-dir ./results
```

## Understanding Results

### CSV Output Format

The `benchmark_results.csv` contains the following columns:

| Column | Description |
|--------|-------------|
| map | Map number (1-7) |
| global_alg | Global planning algorithm |
| local_alg | Local planning algorithm |
| obstacles | Number of dynamic obstacles |
| success | True if goal was reached |
| path_length | Total path length |
| direction_changes | Number of direction changes |
| global_time | Global planning time (seconds) |
| total_time | Total execution time (seconds) |
| steps | Number of execution steps |
| error | Error message if failed |

### Interpreting Metrics

**Path Length (lower is better)**
- Shorter paths are more efficient
- Compare different algorithms to find the most direct routes

**Direction Changes (lower is better)**
- Fewer changes = smoother paths
- Important for real robots (reduces wear, energy consumption)
- Lower values indicate more stable planning

**Planning Time (lower is faster)**
- Measures computational efficiency
- Critical for real-time applications
- Balance with path quality

**Success Rate (higher is better)**
- Percentage of tests where the goal was reached
- Most important metric for reliability
- Should be considered alongside other metrics

## Example Workflow

1. **Quick test on one map**:
   ```bash
   python benchmark.py -m 1 -o 0
   ```

2. **Test without obstacles**:
   ```bash
   python benchmark.py -o 0
   ```

3. **Full benchmark**:
   ```bash
   python benchmark.py
   ```

4. **Analyze results**:
   ```bash
   python analyze_results.py
   ```

5. **Review**:
   - Check `benchmark_report.txt` for summary
   - View PNG files for visual analysis
   - Open CSV in Excel/Pandas for detailed analysis

## Tips

1. **Start Small**: Test on 1-2 maps first to verify everything works
2. **Monitor Progress**: The benchmark prints progress updates
3. **Check Failures**: Review the 'error' column in CSV for failed tests
4. **Compare Algorithms**: Use visualizations to identify best performers
5. **Consider Trade-offs**: Fast algorithms might have longer paths; find the right balance

## Troubleshooting

**Issue**: Benchmark runs very slowly
- **Solution**: Reduce number of maps or obstacle counts, or increase max-steps

**Issue**: Many tests failing
- **Solution**: Check that map files (PNG and TXT) exist in ./map directory
- **Solution**: Verify map dimensions match between PNG and TXT

**Issue**: Out of memory
- **Solution**: Test fewer combinations at a time, then combine results

**Issue**: Analysis script fails
- **Solution**: Ensure pandas, matplotlib, and seaborn are installed
  ```bash
  pip install pandas matplotlib seaborn
  ```

## Advanced Usage

### Custom Analysis

Load and analyze results in Python:

```python
import pandas as pd

# Load results
df = pd.read_csv('benchmark_results.csv')

# Filter successful runs
df_success = df[df['success'] == True]

# Best algorithm by path length
best_by_path = df_success.groupby(['global_alg', 'local_alg'])['path_length'].mean().sort_values()
print(best_by_path.head())

# Success rate by obstacles
success_by_obs = df.groupby('obstacles')['success'].mean()
print(success_by_obs)
```

### Batch Processing

Run multiple benchmarks with different settings:

```bash
# Test each map separately
for i in {1..7}; do
    python benchmark.py -m $i --output map${i}_results.csv
done
```

## Requirements

Make sure you have these packages installed:

```bash
pip install numpy matplotlib pandas seaborn
```

The project structure should be:
```
Code/
├── benchmark.py
├── analyze_results.py
├── main.py
├── map/
│   ├── 1.png, 1.txt
│   ├── 2.png, 2.txt
│   ├── ...
│   └── 7.png, 7.txt
├── planning/
│   ├── global_planning.py
│   └── local_planning.py
└── utils/
    └── (utility modules)
```

## Expected Output

After running the full benchmark and analysis, you should have:

1. **benchmark_results.csv** - 448 rows of test results
2. **benchmark_report.txt** - Text summary with statistics
3. **success_rates.png** - Visual comparison of all 16 combinations
4. **metrics_comparison.png** - Bar charts for all metrics
5. **obstacle_impact.png** - Line plots showing obstacle effects
6. **algorithm_heatmap.png** - Heatmaps for easy comparison

The console will also display:
- Progress updates during testing
- Final summary with success rates
- Top performing combinations
- Statistics by map and obstacle count
