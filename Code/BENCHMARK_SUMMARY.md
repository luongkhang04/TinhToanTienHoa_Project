# Benchmark System for Path Planning Algorithms

## Summary

This comprehensive benchmarking system tests **16 combinations** of global and local planning algorithms across **7 maps** with **4 different obstacle configurations** (0, 50, 100, 200 obstacles), resulting in **448 total test configurations**.

## What Was Created

### 1. Main Benchmark Script (`benchmark.py`)
- Runs all algorithm combinations automatically
- Tests on all maps with different obstacle counts
- Measures path length, direction changes, and planning time
- Saves results to CSV for analysis
- Prints progress and summary statistics

### 2. Analysis Script (`analyze_results.py`)
- Loads benchmark results from CSV
- Generates 4 visualization plots:
  - Success rates by algorithm combination
  - Metrics comparison (path length, direction changes, time)
  - Impact of obstacle count on performance
  - Heatmaps for algorithm comparison
- Creates detailed text report

### 3. Setup Verification (`verify_setup.py`)
- Checks all dependencies are installed
- Verifies map files exist (1.png to 7.png, 1.txt to 7.txt)
- Tests project modules load correctly
- Runs a quick test to ensure everything works
- Estimates benchmark runtime

### 4. Documentation (`BENCHMARK_GUIDE.md`)
- Complete usage instructions
- Explanation of all metrics
- Examples and troubleshooting
- Advanced usage tips

## Algorithms Tested

**Global Planning (4 algorithms):**
1. Grid BFS - Breadth-First Search
2. Grid DFS - Depth-First Search  
3. A* - A-star with heuristic
4. Dijkstra - Dijkstra's algorithm

**Local Planning (4 algorithms):**
1. Reactive BFS - Real-time BFS
2. Reactive DFS - Real-time DFS
3. Potential Field - Artificial potential fields
4. Greedy - Greedy local planner

**Total: 16 combinations** (4 × 4)

## Evaluation Metrics

1. **Path Length** - Total cells in path (lower is better)
2. **Direction Changes** - Number of turns (lower is better, indicates smoother paths)
3. **Planning Time** - Execution time in seconds (lower is faster)
4. **Success Rate** - Percentage of successful goal reaches (higher is better)

## Quick Start

### Step 1: Verify Setup
```bash
python verify_setup.py
```

This checks:
- ✓ Dependencies installed (numpy, matplotlib, pandas, seaborn)
- ✓ Map files exist (./map/1.png through 7.png with .txt files)
- ✓ Project modules load correctly
- ✓ Quick test runs successfully

### Step 2: Run Benchmark

**Option A: Full Benchmark (448 tests, ~30-60 minutes)**
```bash
python benchmark.py
```

**Option B: Quick Test (16 tests, ~1-2 minutes)**
```bash
python benchmark.py -m 1 -o 0
```

**Option C: Custom Selection**
```bash
python benchmark.py -m 1 2 3 -o 0 50
```

### Step 3: Analyze Results
```bash
python analyze_results.py
```

This creates:
- `success_rates.png`
- `metrics_comparison.png`
- `obstacle_impact.png`
- `algorithm_heatmap.png`
- `benchmark_report.txt`

## Expected Results

After running the full benchmark, you'll have:

1. **benchmark_results.csv** - Complete data with 448 rows:
   ```
   map,global_alg,local_alg,obstacles,success,path_length,direction_changes,global_time,total_time,steps,error
   1,grid_bfs,reactive_bfs,0,True,142,23,0.0234,0.891,135,
   1,grid_bfs,reactive_dfs,0,True,148,27,0.0234,0.923,141,
   ...
   ```

2. **Visual Analysis** - 4 PNG plots showing:
   - Which algorithm combinations work best
   - How metrics compare across algorithms
   - Impact of obstacles on performance
   - Heatmap for quick comparison

3. **Text Report** - Detailed statistics including:
   - Overall success rates
   - Rankings of algorithm combinations
   - Performance by obstacle count
   - Performance by map
   - Best combinations for each metric

## Understanding Output

### Success Rates
Shows which algorithm combinations successfully reach the goal most often. Look for combinations with >90% success rate.

### Path Length
Shorter paths are more efficient. Compare successful runs to find optimal algorithms.

### Direction Changes
Fewer changes = smoother paths. Important for real robots where frequent turning is costly.

### Planning Time
Faster algorithms are better for real-time applications. Balance speed with path quality.

## Command Reference

```bash
# Verify setup before starting
python verify_setup.py

# Run full benchmark (all 448 tests)
python benchmark.py

# Test specific maps
python benchmark.py -m 1 2 3

# Test specific obstacle counts
python benchmark.py -o 0 50

# Combine options
python benchmark.py -m 1 -o 0 100 --max-steps 3000

# Custom output file
python benchmark.py --output my_results.csv

# Analyze results
python analyze_results.py

# Analyze custom results
python analyze_results.py --input my_results.csv --output-dir ./plots

# Get help
python benchmark.py --help
python analyze_results.py --help
```

## Files Structure

```
Code/
├── benchmark.py              # Main benchmark script
├── analyze_results.py        # Analysis and visualization
├── verify_setup.py           # Setup verification
├── BENCHMARK_GUIDE.md        # Detailed documentation
├── BENCHMARK_SUMMARY.md      # This file
├── requirements.txt          # Python dependencies (updated)
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

## Output Files

After running benchmark and analysis:

```
Code/
├── benchmark_results.csv      # All test results
├── benchmark_report.txt       # Detailed text summary
├── success_rates.png          # Success rate comparison
├── metrics_comparison.png     # Metrics bar charts
├── obstacle_impact.png        # Obstacle effect plots
└── algorithm_heatmap.png      # Performance heatmaps
```

## Tips for Best Results

1. **Start with verification**: Always run `verify_setup.py` first
2. **Test incrementally**: Try `-m 1 -o 0` before full benchmark
3. **Monitor progress**: Watch console output during benchmark
4. **Review failures**: Check CSV 'error' column for failed tests
5. **Compare carefully**: Use visualizations to identify trade-offs
6. **Consider context**: Best algorithm depends on your priorities (speed vs. path quality)

## Troubleshooting

**Problem**: Dependencies missing  
**Solution**: `pip install -r requirements.txt`

**Problem**: Map files not found  
**Solution**: Ensure map/ directory has 1.png through 7.png and corresponding .txt files

**Problem**: Benchmark too slow  
**Solution**: Start with fewer tests using `-m 1 2 -o 0`

**Problem**: Analysis fails  
**Solution**: Make sure benchmark completed and CSV file exists

## Next Steps

1. Run `verify_setup.py` to check everything is ready
2. Start with a quick test: `python benchmark.py -m 1 -o 0`
3. If successful, run full benchmark: `python benchmark.py`
4. Analyze results: `python analyze_results.py`
5. Review plots and report to identify best algorithms

## Example Workflow

```bash
# 1. Verify everything is set up
python verify_setup.py

# 2. Quick test on Map 1 with no obstacles (16 tests, ~1 min)
python benchmark.py -m 1 -o 0

# 3. If successful, run partial benchmark (112 tests, ~10 min)
python benchmark.py -m 1 2 -o 0 50 100 200

# 4. Analyze partial results
python analyze_results.py

# 5. If satisfied, run full benchmark (448 tests, ~30-60 min)
python benchmark.py

# 6. Final analysis
python analyze_results.py

# 7. Review results
# - Open PNG files to see visualizations
# - Read benchmark_report.txt for summary
# - Load CSV in Excel/Python for detailed analysis
```

## Credits

This benchmarking system tests path planning algorithms from the Evolutionary Computation course project. It provides comprehensive evaluation across multiple scenarios to identify optimal algorithm combinations for different use cases.
