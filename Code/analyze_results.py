"""
Script to analyze and visualize benchmark results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create results folder if it doesn't exist
RESULTS_DIR = 'results'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def load_results(filename='benchmark_results.csv'):
    """Load benchmark results from CSV"""
    try:
        filepath = os.path.join(RESULTS_DIR, filename)
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} results from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


def plot_success_rates(df, save_path='success_rates.png'):
    """Plot success rates by algorithm combination"""
    # Calculate success rate for each combination
    success_by_combo = df.groupby(['global_alg', 'local_alg']).agg({
        'success': ['sum', 'count']
    }).reset_index()
    success_by_combo.columns = ['global_alg', 'local_alg', 'successes', 'total']
    success_by_combo['success_rate'] = success_by_combo['successes'] / success_by_combo['total'] * 100
    success_by_combo['combo'] = success_by_combo['global_alg'] + ' + ' + success_by_combo['local_alg']
    
    # Sort by success rate
    success_by_combo = success_by_combo.sort_values('success_rate', ascending=False)
    
    # Plot
    plt.figure(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(success_by_combo)))
    bars = plt.bar(range(len(success_by_combo)), success_by_combo['success_rate'], color=colors)
    plt.xlabel('Algorithm Combination', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('Success Rate by Algorithm Combination', fontsize=14, fontweight='bold')
    plt.xticks(range(len(success_by_combo)), success_by_combo['combo'], rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, save_path)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved success rates plot to {filepath}")
    plt.close()


def plot_metrics_comparison(df, save_path='metrics_comparison.png'):
    """Plot comparison of path length, direction changes, and time"""
    # Filter only successful runs
    df_success = df[df['success'] == True].copy()
    
    if len(df_success) == 0:
        print("No successful runs to plot")
        return
    
    # Create combo column
    df_success['combo'] = df_success['global_alg'] + ' + ' + df_success['local_alg']
    
    # Calculate averages
    metrics = df_success.groupby('combo').agg({
        'path_length': 'mean',
        'direction_changes': 'mean',
        'total_time': 'mean'
    }).reset_index()
    
    # Sort by path length
    metrics = metrics.sort_values('path_length')
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Path length
    axes[0].barh(range(len(metrics)), metrics['path_length'], color='steelblue')
    axes[0].set_yticks(range(len(metrics)))
    axes[0].set_yticklabels(metrics['combo'])
    axes[0].set_xlabel('Average Path Length', fontsize=11)
    axes[0].set_title('Path Length (lower is better)', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Direction changes
    metrics_sorted = metrics.sort_values('direction_changes')
    axes[1].barh(range(len(metrics_sorted)), metrics_sorted['direction_changes'], color='coral')
    axes[1].set_yticks(range(len(metrics_sorted)))
    axes[1].set_yticklabels(metrics_sorted['combo'])
    axes[1].set_xlabel('Average Direction Changes', fontsize=11)
    axes[1].set_title('Direction Changes (lower is better)', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Time
    metrics_sorted = metrics.sort_values('total_time')
    axes[2].barh(range(len(metrics_sorted)), metrics_sorted['total_time'], color='mediumseagreen')
    axes[2].set_yticks(range(len(metrics_sorted)))
    axes[2].set_yticklabels(metrics_sorted['combo'])
    axes[2].set_xlabel('Average Time (seconds)', fontsize=11)
    axes[2].set_title('Planning Time (lower is better)', fontsize=12, fontweight='bold')
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, save_path)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved metrics comparison to {filepath}")
    plt.close()


def plot_obstacle_impact(df, save_path='obstacle_impact.png'):
    """Plot how obstacle count affects performance"""
    df_success = df[df['success'] == True].copy()
    
    if len(df_success) == 0:
        print("No successful runs to plot")
        return
    
    df_success['combo'] = df_success['global_alg'] + ' + ' + df_success['local_alg']
    
    # Calculate metrics by obstacle count
    metrics_by_obs = df_success.groupby(['obstacles', 'combo']).agg({
        'path_length': 'mean',
        'direction_changes': 'mean',
        'total_time': 'mean'
    }).reset_index()
    
    # Get top 5 combinations by success rate
    success_rates = df.groupby('combo')['success'].mean().sort_values(ascending=False).head(5)
    top_combos = success_rates.index.tolist()
    
    # Filter to top combos
    metrics_by_obs = metrics_by_obs[metrics_by_obs['combo'].isin(top_combos)]
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Path length vs obstacles
    for combo in top_combos:
        data = metrics_by_obs[metrics_by_obs['combo'] == combo]
        axes[0].plot(data['obstacles'], data['path_length'], marker='o', label=combo, linewidth=2)
    axes[0].set_xlabel('Number of Obstacles', fontsize=11)
    axes[0].set_ylabel('Average Path Length', fontsize=11)
    axes[0].set_title('Path Length vs Obstacles', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=8, loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Direction changes vs obstacles
    for combo in top_combos:
        data = metrics_by_obs[metrics_by_obs['combo'] == combo]
        axes[1].plot(data['obstacles'], data['direction_changes'], marker='o', label=combo, linewidth=2)
    axes[1].set_xlabel('Number of Obstacles', fontsize=11)
    axes[1].set_ylabel('Average Direction Changes', fontsize=11)
    axes[1].set_title('Direction Changes vs Obstacles', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=8, loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # Time vs obstacles
    for combo in top_combos:
        data = metrics_by_obs[metrics_by_obs['combo'] == combo]
        axes[2].plot(data['obstacles'], data['total_time'], marker='o', label=combo, linewidth=2)
    axes[2].set_xlabel('Number of Obstacles', fontsize=11)
    axes[2].set_ylabel('Average Time (seconds)', fontsize=11)
    axes[2].set_title('Planning Time vs Obstacles', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=8, loc='best')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, save_path)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved obstacle impact plot to {filepath}")
    plt.close()


def plot_heatmap(df, save_path='algorithm_heatmap.png'):
    """Create heatmap of algorithm performance"""
    df_success = df[df['success'] == True].copy()
    
    if len(df_success) == 0:
        print("No successful runs to plot")
        return
    
    # Calculate average metrics by global and local algorithm
    pivot_path = df_success.pivot_table(
        values='path_length',
        index='local_alg',
        columns='global_alg',
        aggfunc='mean'
    )
    
    pivot_changes = df_success.pivot_table(
        values='direction_changes',
        index='local_alg',
        columns='global_alg',
        aggfunc='mean'
    )
    
    pivot_time = df_success.pivot_table(
        values='total_time',
        index='local_alg',
        columns='global_alg',
        aggfunc='mean'
    )
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Path length heatmap
    sns.heatmap(pivot_path, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=axes[0], cbar_kws={'label': 'Path Length'})
    axes[0].set_title('Average Path Length', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Global Algorithm', fontsize=11)
    axes[0].set_ylabel('Local Algorithm', fontsize=11)
    
    # Direction changes heatmap
    sns.heatmap(pivot_changes, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=axes[1], cbar_kws={'label': 'Direction Changes'})
    axes[1].set_title('Average Direction Changes', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Global Algorithm', fontsize=11)
    axes[1].set_ylabel('Local Algorithm', fontsize=11)
    
    # Time heatmap
    sns.heatmap(pivot_time, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=axes[2], cbar_kws={'label': 'Time (s)'})
    axes[2].set_title('Average Planning Time', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Global Algorithm', fontsize=11)
    axes[2].set_ylabel('Local Algorithm', fontsize=11)
    
    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, save_path)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved algorithm heatmap to {filepath}")
    plt.close()


def generate_report(df, output_file='benchmark_report.txt'):
    """Generate detailed text report"""
    filepath = os.path.join(RESULTS_DIR, output_file)
    with open(filepath, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BENCHMARK RESULTS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        total = len(df)
        successful = df['success'].sum()
        f.write(f"Total Tests: {total}\n")
        f.write(f"Successful: {successful} ({100*successful/total:.1f}%)\n")
        f.write(f"Failed: {total-successful} ({100*(total-successful)/total:.1f}%)\n\n")
        
        # Success statistics
        df_success = df[df['success'] == True].copy()
        if len(df_success) > 0:
            f.write("-"*80 + "\n")
            f.write("SUCCESSFUL RUNS STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Average Path Length: {df_success['path_length'].mean():.2f} ± {df_success['path_length'].std():.2f}\n")
            f.write(f"Average Direction Changes: {df_success['direction_changes'].mean():.2f} ± {df_success['direction_changes'].std():.2f}\n")
            f.write(f"Average Total Time: {df_success['total_time'].mean():.4f}s ± {df_success['total_time'].std():.4f}s\n")
            f.write(f"Average Global Planning Time: {df_success['global_time'].mean():.4f}s ± {df_success['global_time'].std():.4f}s\n\n")
        
        # Best combinations
        df['combo'] = df['global_alg'] + ' + ' + df['local_alg']
        success_rates = df.groupby('combo').agg({
            'success': ['sum', 'count']
        })
        success_rates.columns = ['successes', 'total']
        success_rates['rate'] = success_rates['successes'] / success_rates['total']
        success_rates = success_rates.sort_values('rate', ascending=False)
        
        f.write("-"*80 + "\n")
        f.write("ALGORITHM COMBINATIONS RANKED BY SUCCESS RATE\n")
        f.write("-"*80 + "\n")
        for i, (combo, row) in enumerate(success_rates.iterrows(), 1):
            f.write(f"{i:2}. {combo:35} {row['successes']:3}/{row['total']:3} ({100*row['rate']:5.1f}%)\n")
        f.write("\n")
        
        # Performance by obstacle count
        f.write("-"*80 + "\n")
        f.write("PERFORMANCE BY OBSTACLE COUNT\n")
        f.write("-"*80 + "\n")
        for obs_count in sorted(df['obstacles'].unique()):
            df_obs = df[df['obstacles'] == obs_count]
            success_rate = df_obs['success'].sum() / len(df_obs)
            f.write(f"\n{obs_count} Obstacles: {df_obs['success'].sum()}/{len(df_obs)} successful ({100*success_rate:.1f}%)\n")
            
            df_obs_success = df_obs[df_obs['success'] == True]
            if len(df_obs_success) > 0:
                f.write(f"  Avg Path Length: {df_obs_success['path_length'].mean():.2f}\n")
                f.write(f"  Avg Direction Changes: {df_obs_success['direction_changes'].mean():.2f}\n")
                f.write(f"  Avg Time: {df_obs_success['total_time'].mean():.4f}s\n")
        f.write("\n")
        
        # Performance by map
        f.write("-"*80 + "\n")
        f.write("PERFORMANCE BY MAP\n")
        f.write("-"*80 + "\n")
        for map_num in sorted(df['map'].unique()):
            df_map = df[df['map'] == map_num]
            success_rate = df_map['success'].sum() / len(df_map)
            f.write(f"Map {map_num}: {df_map['success'].sum():2}/{len(df_map):2} successful ({100*success_rate:5.1f}%)\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"Saved detailed report to {filepath}")


def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('--input', default='benchmark_results.csv',
                       help='Input CSV file in results/ folder (default: benchmark_results.csv)')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for plots (default: results)')
    
    args = parser.parse_args()
    
    # Load results
    df = load_results(args.input)
    if df is None:
        return
    
    # Add combo column
    df['combo'] = df['global_alg'] + ' + ' + df['local_alg']
    
    print("\nGenerating visualizations...")
    
    # Generate plots
    plot_success_rates(df, 'success_rates.png')
    plot_metrics_comparison(df, 'metrics_comparison.png')
    plot_obstacle_impact(df, 'obstacle_impact.png')
    plot_heatmap(df, 'algorithm_heatmap.png')
    
    # Generate report
    generate_report(df, 'benchmark_report.txt')
    
    print("\nAnalysis complete!")
    print(f"Generated files in {RESULTS_DIR}:")
    print("  - success_rates.png")
    print("  - metrics_comparison.png")
    print("  - obstacle_impact.png")
    print("  - algorithm_heatmap.png")
    print("  - benchmark_report.txt")


if __name__ == '__main__':
    main()
