"""
Quick test to verify benchmark setup before running full tests.
Tests a few combinations to ensure everything works correctly.
"""

import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    required = ['numpy', 'matplotlib', 'pandas', 'seaborn']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("All dependencies installed!\n")
    return True


def check_map_files():
    """Check if map files exist"""
    print("Checking map files...")
    map_dir = 'map'
    
    if not os.path.exists(map_dir):
        print(f"  ✗ Directory '{map_dir}' not found")
        return False
    
    missing = []
    for i in range(1, 8):
        png_file = f"{map_dir}/{i}.png"
        txt_file = f"{map_dir}/{i}.txt"
        
        if os.path.exists(png_file) and os.path.exists(txt_file):
            print(f"  ✓ Map {i} ({png_file}, {txt_file})")
        else:
            if not os.path.exists(png_file):
                print(f"  ✗ {png_file} - MISSING")
                missing.append(png_file)
            if not os.path.exists(txt_file):
                print(f"  ✗ {txt_file} - MISSING")
                missing.append(txt_file)
    
    if missing:
        print(f"\nMissing files: {len(missing)}")
        return False
    
    print("All map files found!\n")
    return True


def check_modules():
    """Check if planning modules are available"""
    print("Checking project modules...")
    
    try:
        from planning.global_planning import GridBFSPlanner, GridDFSPlanner, GridAStarPlanner, SimpleDijkstraPlanner
        print("  ✓ Global planning algorithms")
    except ImportError as e:
        print(f"  ✗ Global planning algorithms - {e}")
        return False
    
    try:
        from planning.local_planning import ReactiveBFSPlanner, ReactiveDFSPlanner, PotentialFieldPlanner, GreedyLocalPlanner
        print("  ✓ Local planning algorithms")
    except ImportError as e:
        print(f"  ✗ Local planning algorithms - {e}")
        return False
    
    try:
        from utils.Global import Global
        from utils import move
        print("  ✓ Utility modules")
    except ImportError as e:
        print(f"  ✗ Utility modules - {e}")
        return False
    
    print("All modules loaded successfully!\n")
    return True


def run_quick_test():
    """Run a quick test with one configuration"""
    print("Running quick test (1 configuration)...")
    print("  Map: 1, Obstacles: 0, Algorithms: grid_bfs + reactive_bfs\n")
    
    try:
        from benchmark import BenchmarkRunner
        
        # Run single test
        benchmark = BenchmarkRunner(maps=[1], obstacle_counts=[0], max_steps=500)
        result = benchmark.run_single_test(1, 'grid_bfs', 'reactive_bfs', 0)
        
        print("\nTest Result:")
        print(f"  Success: {result['success']}")
        print(f"  Path Length: {result['path_length']}")
        print(f"  Direction Changes: {result['direction_changes']}")
        print(f"  Total Time: {result['total_time']:.3f}s")
        
        if result['success']:
            print("\n✓ Quick test PASSED!")
            return True
        else:
            print(f"\n✗ Quick test FAILED: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n✗ Quick test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def estimate_runtime():
    """Estimate full benchmark runtime"""
    print("\n" + "="*60)
    print("BENCHMARK RUNTIME ESTIMATE")
    print("="*60)
    
    num_global = 4  # grid_bfs, grid_dfs, astar, dijkstra
    num_local = 4   # reactive_bfs, reactive_dfs, potential_field, greedy
    num_maps = 7
    num_obstacles = 4  # 0, 50, 100, 200
    
    total_tests = num_global * num_local * num_maps * num_obstacles
    
    print(f"Total tests: {total_tests}")
    print(f"  {num_global} global algorithms × {num_local} local algorithms × {num_maps} maps × {num_obstacles} obstacle counts")
    print(f"\nEstimated time per test: 3-10 seconds")
    print(f"Estimated total time: {total_tests * 3 / 60:.1f} - {total_tests * 10 / 60:.1f} minutes")
    print(f"\nNote: Actual time may vary based on:")
    print(f"  - Map complexity")
    print(f"  - Algorithm performance")
    print(f"  - Hardware specifications")
    print("="*60)


def main():
    """Main verification function"""
    print("\n" + "="*60)
    print("BENCHMARK SETUP VERIFICATION")
    print("="*60 + "\n")
    
    all_checks_passed = True
    
    # Check dependencies
    if not check_dependencies():
        all_checks_passed = False
    
    # Check map files
    if not check_map_files():
        all_checks_passed = False
    
    # Check modules
    if not check_modules():
        all_checks_passed = False
    
    if not all_checks_passed:
        print("\n" + "="*60)
        print("SETUP INCOMPLETE")
        print("="*60)
        print("Please fix the issues above before running the benchmark.")
        return False
    
    # Run quick test
    print("="*60)
    if not run_quick_test():
        print("\n" + "="*60)
        print("QUICK TEST FAILED")
        print("="*60)
        print("Please check the error messages and fix before running full benchmark.")
        return False
    
    # Show runtime estimate
    estimate_runtime()
    
    print("\n" + "="*60)
    print("SETUP VERIFICATION COMPLETE")
    print("="*60)
    print("\nYou're ready to run the full benchmark!")
    print("\nTo start benchmarking:")
    print("  Full benchmark:    python benchmark.py")
    print("  Partial test:      python benchmark.py -m 1 2 -o 0 50")
    print("  Help:              python benchmark.py --help")
    print("\nAfter completion:")
    print("  Analyze results:   python analyze_results.py")
    print("="*60 + "\n")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
