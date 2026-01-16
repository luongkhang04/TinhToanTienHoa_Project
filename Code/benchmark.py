"""
Benchmark script to evaluate all combinations of global and local planning algorithms.
Tests on multiple maps with varying numbers of dynamic obstacles.
"""

import sys
import time
import csv
import os
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import threading
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Create results folder if it doesn't exist
RESULTS_DIR = 'results'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

from utils import move
from utils.Global import Global
from planning.global_planning import GridBFSPlanner, GridDFSPlanner, GridAStarPlanner, SimpleDijkstraPlanner
from planning.local_planning import (
    ReactiveBFSPlanner,
    ReactiveDFSPlanner,
    PotentialFieldPlanner,
    GreedyLocalPlanner,
    DynamicWindowPlanner,
)
from utils.agent_graph import agent_graph


class BenchmarkRunner:
    """Run comprehensive benchmarks on all algorithm combinations"""
    
    # Define all algorithms
    GLOBAL_ALGORITHMS = {
        'grid_bfs': GridBFSPlanner,
        'grid_dfs': GridDFSPlanner,
        'astar': GridAStarPlanner,
        'dijkstra': SimpleDijkstraPlanner,
    }
    
    LOCAL_ALGORITHMS = {
        'reactive_bfs': ReactiveBFSPlanner,
        'reactive_dfs': ReactiveDFSPlanner,
        'potential_field': PotentialFieldPlanner,
        'greedy': GreedyLocalPlanner,
        'dwa': DynamicWindowPlanner,
    }
    
    def __init__(self, maps=None, obstacle_counts=None, max_steps=2000, timeout=100):
        """
        Args:
            maps: List of map numbers (default: 1-7)
            obstacle_counts: List of obstacle counts (default: [0, 50, 100, 200])
            max_steps: Maximum steps per planning attempt
            timeout: Maximum time (seconds) per test (default: 100)
        """
        self.maps = maps if maps is not None else list(range(1, 8))
        self.obstacle_counts = obstacle_counts if obstacle_counts is not None else [0, 50, 100, 200]
        self.max_steps = max_steps
        self.timeout = timeout
        self.results = []
        self._abort_test = False  # Flag to abort current test
        
    def load_start_goal(self, map_num):
        """Load start and goal positions from map txt file"""
        filename = f'map/{map_num}.txt'
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                start = tuple(map(int, lines[0].strip().split(',')))
                goal = tuple(map(int, lines[1].strip().split(',')))
                return start, goal
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            # Return default positions
            return (8, 8), (119, 8)
    
    def load_map(self, map_num):
        """Load map image"""
        filename = f'map/{map_num}.png'
        try:
            image = plt.imread(filename)
            return image
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    
    def count_direction_changes(self, path):
        """Count number of times the direction changes in the path"""
        if len(path) < 3:
            return 0
        
        changes = 0
        prev_direction = None
        
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            
            # Normalize direction
            if dx != 0:
                dx = dx // abs(dx)
            if dy != 0:
                dy = dy // abs(dy)
            
            direction = (dx, dy)
            
            if prev_direction is not None and direction != prev_direction and direction != (0, 0):
                changes += 1
            
            if direction != (0, 0):
                prev_direction = direction
        
        return changes
    
    def run_single_test(self, map_num, global_alg_name, local_alg_name, num_obstacles):
        """
        Run a single test configuration
        
        Returns:
            dict with results: {
                'map': map number,
                'global_alg': algorithm name,
                'local_alg': algorithm name,
                'obstacles': count,
                'success': True/False,
                'path_length': length,
                'direction_changes': count,
                'global_time': seconds,
                'total_time': seconds,
                'steps': number of execution steps
            }
        """
        print(f"\nTesting Map {map_num}: {global_alg_name} + {local_alg_name}, {num_obstacles} obstacles")
        
        result = {
            'map': map_num,
            'global_alg': global_alg_name,
            'local_alg': local_alg_name,
            'obstacles': num_obstacles,
            'success': False,
            'path_length': 0,
            'direction_changes': 0,
            'global_time': 0.0,
            'total_time': 0.0,
            'steps': 0,
            'error': None
        }
        
        try:
            # Load map and positions
            image = self.load_map(map_num)
            if image is None:
                result['error'] = 'Map not found'
                return result
            
            start_pos, goal_pos = self.load_start_goal(map_num)
            map_width, map_height = image.shape
            map_dimensions = (map_width, map_height)
            
            # Initialize environment
            g = Global(map_dimensions)
            
            # Load static obstacles
            for x in range(map_width):
                for y in range(map_height):
                    if image[y, x] == 0:
                        g.createObstacle(move.NONE, (x, y))
            
            # Create global planner and plan
            global_planner_class = self.GLOBAL_ALGORITHMS[global_alg_name]
            global_planner = global_planner_class(image, map_dimensions)
            
            global_start = time.time()
            global_path, _ = global_planner.plan(start_pos, goal_pos)
            global_time = time.time() - global_start
            result['global_time'] = global_time
            
            if not global_path or len(global_path) == 0:
                result['error'] = 'Global planning failed'
                return result
            
            # Create agent
            agent = g.createAgent(start_pos)
            agent.range = 2
            
            # Create moving obstacles
            moving_obstacles = []
            if num_obstacles > 0:
                x_positions = np.random.randint(0, map_width, num_obstacles)
                y_positions = np.random.randint(0, map_height, num_obstacles)
                positions = np.column_stack([x_positions, y_positions])
                
                for element in positions:
                    obs = g.createObstacle(move.RANDOM, tuple(element))
                    moving_obstacles.append(obs)
            
            # Create local planner
            obs = g.observe()
            local_planner_class = self.LOCAL_ALGORITHMS[local_alg_name]
            local_planner = local_planner_class(agent, obs, map_width, map_height, margin=1)
            
            # Execute local planning
            execution_start = time.time()
            idx_goal = 0
            current_state = tuple(int(x) for x in agent.location)
            previous_state = current_state
            complete_path = [current_state]
            
            steps = 0
            stuck_counter = 0
            total_stuck_count = 0  # Total stuck instances (for early abort)
            stuck_threshold = 15  # Trigger replanning after 15 steps without progress
            max_total_stuck = 50  # Abort test if stuck too many times overall
            goal_pos_int = tuple(int(x) for x in goal_pos)
            final_goal_tolerance = 3  # Tolerance for reaching final goal
            waypoint_tolerance = 2    # Tolerance for intermediate waypoints
            
            while steps < self.max_steps:
                # Check timeout
                elapsed = time.time() - execution_start
                if elapsed > self.timeout:
                    result['error'] = f'Timeout after {elapsed:.1f}s'
                    break
                
                # Check if algorithm is hopelessly stuck
                if total_stuck_count > max_total_stuck:
                    result['error'] = f'Algorithm stuck {total_stuck_count} times, aborting'
                    break
                # Check if we've reached the final goal
                dist_to_final = abs(current_state[0] - goal_pos_int[0]) + abs(current_state[1] - goal_pos_int[1])
                if dist_to_final <= final_goal_tolerance:
                    result['success'] = True
                    break
                
                # Get current waypoint goal
                if idx_goal >= len(global_path):
                    # All waypoints processed but not at final goal - target final goal directly
                    goal = goal_pos_int
                else:
                    goal = tuple(int(x) for x in global_path[idx_goal])
                
                projected_goal = local_planner.graph.project_to_surroundings(goal)
                
                # Get local plan
                action = local_planner.plan_to_goal(current_state, projected_goal)
                
                # If no action, try random movement or skip
                action = tuple(int(x) for x in action) if action is not None else (0, 0)
                
                if action == (0, 0):
                    # Try to escape stuck position with random valid move
                    from utils.agent_graph import agent_graph as get_graph
                    obs = g.observe()
                    temp_graph = get_graph(agent, obs, map_width, map_height, inflation=1)
                    valid_neighbors = temp_graph.get_adjacent_states(current_state)
                    
                    if valid_neighbors:
                        action = (valid_neighbors[0][0] - current_state[0], 
                                 valid_neighbors[0][1] - current_state[1])
                    else:
                        # Completely stuck, try moving to next waypoint (but not past goal)
                        if idx_goal < len(global_path):
                            idx_goal += 1
                        stuck_counter = 0
                        continue
                
                # Execute action
                agent.action = action
                g.next()
                
                # Update state
                previous_state = current_state
                current_state = tuple(int(x) for x in agent.location)
                complete_path.append(current_state)
                
                # Detect stuck: not moving OR not making progress
                if current_state == previous_state:
                    stuck_counter += 1
                else:
                    # Check if we're making progress toward goal
                    old_dist = abs(previous_state[0] - goal[0]) + abs(previous_state[1] - goal[1])
                    new_dist = abs(current_state[0] - goal[0]) + abs(current_state[1] - goal[1])
                    if new_dist >= old_dist:
                        stuck_counter += 1
                    else:
                        stuck_counter = 0
                
                if stuck_counter > stuck_threshold:
                    # Replan globally from current position to final goal
                    total_stuck_count += stuck_counter
                    global_path, replan_time = global_planner.plan(current_state, goal_pos)
                    result['global_time'] += replan_time
                    idx_goal = 0
                    stuck_counter = 0
                    
                    if not global_path or len(global_path) == 0:
                        result['error'] = 'Replanning failed'
                        break
                    continue
                
                # Update observation
                obs = g.observe()
                local_planner.update_observation(agent, obs)
                
                # Check waypoint reached (with tolerance)
                dist_to_waypoint = abs(current_state[0] - goal[0]) + abs(current_state[1] - goal[1])
                if dist_to_waypoint <= waypoint_tolerance and idx_goal < len(global_path):
                    idx_goal += 1
                    stuck_counter = 0
                
                steps += 1
            
            execution_time = time.time() - execution_start
            result['total_time'] = execution_time
            result['steps'] = steps
            
            # Final success check based on distance to goal
            dist_to_final = abs(current_state[0] - goal_pos_int[0]) + abs(current_state[1] - goal_pos_int[1])
            if dist_to_final <= final_goal_tolerance:
                result['success'] = True
                result['path_length'] = len(complete_path)
                result['direction_changes'] = self.count_direction_changes(complete_path)
            else:
                result['error'] = f'Failed to reach goal (reached {idx_goal}/{len(global_path)} waypoints, dist={dist_to_final})'
            
            print(f"  Result: {'SUCCESS' if result['success'] else 'FAILED'} - "
                  f"Path: {result['path_length']}, Changes: {result['direction_changes']}, "
                  f"Time: {result['total_time']:.3f}s")
            
        except Exception as e:
            result['error'] = str(e)
            print(f"  ERROR: {e}")
        
        return result
    
    def run_all_tests(self):
        """Run all combinations of tests"""
        total_tests = (len(self.GLOBAL_ALGORITHMS) * len(self.LOCAL_ALGORITHMS) * 
                       len(self.maps) * len(self.obstacle_counts))
        
        print(f"\n{'='*80}")
        print(f"Starting Benchmark: {total_tests} total tests")
        print(f"Maps: {self.maps}")
        print(f"Obstacle counts: {self.obstacle_counts}")
        print(f"Global algorithms: {list(self.GLOBAL_ALGORITHMS.keys())}")
        print(f"Local algorithms: {list(self.LOCAL_ALGORITHMS.keys())}")
        print(f"Max steps: {self.max_steps}, Timeout: {self.timeout}s per test")
        print(f"{'='*80}\n")
        
        test_num = 0
        
        # Iterate through all combinations
        for map_num in self.maps:
            for num_obstacles in self.obstacle_counts:
                for global_alg in self.GLOBAL_ALGORITHMS.keys():
                    for local_alg in self.LOCAL_ALGORITHMS.keys():
                        test_num += 1
                        print(f"\n[Test {test_num}/{total_tests}]")
                        
                        result = self.run_single_test(map_num, global_alg, local_alg, num_obstacles)
                        self.results.append(result)
        
        print(f"\n{'='*80}")
        print(f"Benchmark Complete: {test_num} tests finished")
        print(f"{'='*80}\n")
    
    def save_results(self, filename='benchmark_results.csv'):
        """Save results to CSV file in results folder"""
        if not self.results:
            print("No results to save")
            return
        
        filepath = os.path.join(RESULTS_DIR, filename)
        fieldnames = ['map', 'global_alg', 'local_alg', 'obstacles', 'success', 
                     'path_length', 'direction_changes', 'global_time', 'total_time', 
                     'steps', 'error']
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"\nResults saved to {filepath}")
    
    def print_summary(self):
        """Print summary statistics"""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Overall statistics
        total = len(self.results)
        successful = sum(1 for r in self.results if r['success'])
        failed = total - successful
        
        print(f"\nOverall: {successful}/{total} successful ({100*successful/total:.1f}%)")
        print(f"Failed: {failed} ({100*failed/total:.1f}%)")
        
        # Statistics by algorithm combination
        print("\n" + "-"*80)
        print("Success Rate by Algorithm Combination:")
        print("-"*80)
        
        for global_alg in self.GLOBAL_ALGORITHMS.keys():
            for local_alg in self.LOCAL_ALGORITHMS.keys():
                combo_results = [r for r in self.results 
                               if r['global_alg'] == global_alg and r['local_alg'] == local_alg]
                combo_success = sum(1 for r in combo_results if r['success'])
                combo_total = len(combo_results)
                
                if combo_total > 0:
                    avg_path = np.mean([r['path_length'] for r in combo_results if r['success']]) if combo_success > 0 else 0
                    avg_changes = np.mean([r['direction_changes'] for r in combo_results if r['success']]) if combo_success > 0 else 0
                    avg_time = np.mean([r['total_time'] for r in combo_results if r['success']]) if combo_success > 0 else 0
                    
                    print(f"{global_alg:12} + {local_alg:15}: "
                          f"{combo_success:2}/{combo_total:2} ({100*combo_success/combo_total:5.1f}%) | "
                          f"Avg Path: {avg_path:6.1f} | "
                          f"Avg Changes: {avg_changes:5.1f} | "
                          f"Avg Time: {avg_time:6.3f}s")
        
        # Statistics by obstacle count
        print("\n" + "-"*80)
        print("Success Rate by Obstacle Count:")
        print("-"*80)
        
        for obs_count in self.obstacle_counts:
            obs_results = [r for r in self.results if r['obstacles'] == obs_count]
            obs_success = sum(1 for r in obs_results if r['success'])
            obs_total = len(obs_results)
            
            if obs_total > 0:
                print(f"{obs_count:3} obstacles: {obs_success:3}/{obs_total:3} "
                      f"({100*obs_success/obs_total:5.1f}%)")
        
        # Statistics by map
        print("\n" + "-"*80)
        print("Success Rate by Map:")
        print("-"*80)
        
        for map_num in self.maps:
            map_results = [r for r in self.results if r['map'] == map_num]
            map_success = sum(1 for r in map_results if r['success'])
            map_total = len(map_results)
            
            if map_total > 0:
                print(f"Map {map_num}: {map_success:2}/{map_total:2} "
                      f"({100*map_success/map_total:5.1f}%)")
        
        # Best performing combinations
        print("\n" + "-"*80)
        print("Top 5 Algorithm Combinations (by success rate):")
        print("-"*80)
        
        combo_stats = {}
        for global_alg in self.GLOBAL_ALGORITHMS.keys():
            for local_alg in self.LOCAL_ALGORITHMS.keys():
                combo_results = [r for r in self.results 
                               if r['global_alg'] == global_alg and r['local_alg'] == local_alg]
                combo_success = sum(1 for r in combo_results if r['success'])
                combo_total = len(combo_results)
                
                if combo_total > 0:
                    success_rate = combo_success / combo_total
                    combo_stats[f"{global_alg} + {local_alg}"] = {
                        'rate': success_rate,
                        'success': combo_success,
                        'total': combo_total
                    }
        
        sorted_combos = sorted(combo_stats.items(), key=lambda x: x[1]['rate'], reverse=True)
        for i, (combo, stats) in enumerate(sorted_combos[:5], 1):
            print(f"{i}. {combo:30}: {stats['success']:2}/{stats['total']:2} "
                  f"({100*stats['rate']:5.1f}%)")
        
        print("\n" + "="*80)


def main():
    """Main benchmark execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark all planning algorithm combinations')
    parser.add_argument('-m', '--maps', nargs='+', type=int, default=None,
                       help='Map numbers to test (default: 1-7)')
    parser.add_argument('-o', '--obstacles', nargs='+', type=int, default=None,
                       help='Obstacle counts to test (default: 0 50 100 200)')
    parser.add_argument('-s', '--max-steps', type=int, default=2000,
                       help='Maximum steps per test (default: 2000)')
    parser.add_argument('-t', '--timeout', type=int, default=100,
                       help='Maximum time per test in seconds (default: 100)')
    parser.add_argument('--output', default='benchmark_results.csv',
                       help='Output CSV filename in results/ folder (default: benchmark_results.csv)')
    
    args = parser.parse_args()
    
    # Create and run benchmark
    benchmark = BenchmarkRunner(
        maps=args.maps,
        obstacle_counts=args.obstacles,
        max_steps=args.max_steps,
        timeout=args.timeout
    )
    
    benchmark.run_all_tests()
    benchmark.save_results(args.output)
    benchmark.print_summary()


if __name__ == '__main__':
    main()
