"""
Main path planning execution module.
Coordinates global and local planning for dynamic environments.
"""

import sys
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import csv

from utils import move
from utils.Global import Global
from planning.global_planning import GridBFSPlanner, GridDFSPlanner, GridAStarPlanner, SimpleDijkstraPlanner
from planning.local_planning import ReactiveBFSPlanner, ReactiveDFSPlanner, PotentialFieldPlanner, GreedyLocalPlanner


class PathPlanner:
    """
    Integrated path planner combining global and local planning.
    """
    
    def __init__(self, map_filename, global_planner_name='grid_bfs', 
                 local_planner_name='reactive_bfs', num_obstacles=200):
        """
        Args:
            map_filename: Name of map PNG file in 'map/' directory
            global_planner_name: Type of global planner
            local_planner_name: Type of local planner
            num_obstacles: Number of moving obstacles
        """
        self.map_filename = map_filename
        self.global_planner_name = global_planner_name
        self.local_planner_name = local_planner_name
        self.num_obstacles = num_obstacles
        
        # Load map
        self.image = plt.imread('map/' + map_filename)
        self.map_width, self.map_height = self.image.shape
        self.map_dimensions = (self.map_width, self.map_height)
        
        print(f"Map Width: {self.map_width}, Height: {self.map_height}")
        
        # Initialize environment
        self.g = Global(self.map_dimensions)
        self._load_obstacles()
        
        # Create planners
        self.global_planner = self._create_global_planner(global_planner_name)
        self.local_planner = None
        
        # Configuration
        self.start_pos = (8, 8)
        self.goal_pos = (119, 8)
        
        # Statistics
        self.times = [0]  # time steps
        self.complete_path = []
        self.moving_obstacles = []
        self.moving_obstacles_positions = {}
        
    def _load_obstacles(self):
        """Load static obstacles from map image"""
        for x in range(self.map_width):
            for y in range(self.map_height):
                if self.image[y, x] == 0:
                    self.g.createObstacle(move.NONE, (x, y))
    
    def _create_global_planner(self, planner_name):
        """Factory method to create global planner"""
        planners = {
            'grid_bfs': GridBFSPlanner,
            'grid_dfs': GridDFSPlanner,
            'astar': GridAStarPlanner,
            'dijkstra': SimpleDijkstraPlanner,
        }
        
        planner_class = planners.get(planner_name, GridBFSPlanner)
        return planner_class(self.image, self.map_dimensions)
    
    def _create_local_planner(self, planner_name):
        """Factory method to create local planner"""
        planners = {
            'reactive_bfs': ReactiveBFSPlanner,
            'reactive_dfs': ReactiveDFSPlanner,
            'potential_field': PotentialFieldPlanner,
            'greedy': GreedyLocalPlanner,
        }
        
        planner_class = planners.get(planner_name, ReactiveBFSPlanner)
        
        agent = self.g.agents[0]
        obs = self.g.observe()
        
        # Use smaller inflation margin to avoid blocking narrow corridors
        return planner_class(agent, obs, self.map_width, self.map_height, margin=1)
    
    def _create_moving_obstacles(self):
        """Create random moving obstacles"""
        x_positions = np.random.randint(0, self.map_width, self.num_obstacles)
        y_positions = np.random.randint(0, self.map_height, self.num_obstacles)
        positions = np.column_stack([x_positions, y_positions])
        
        for element in positions:
            obs = self.g.createObstacle(move.RANDOM, tuple(element))
            self.moving_obstacles.append(obs)
            self.moving_obstacles_positions[obs] = [tuple(element)]
    
    def plan(self):
        """Execute path planning with global and local planners"""
        print(f"\n{'='*60}")
        print(f"Path Planning with {self.global_planner_name} + {self.local_planner_name}")
        print(f"{'='*60}\n")
        
        # Global planning
        print("Computing global path...")
        global_path, global_time = self.global_planner.plan(self.start_pos, self.goal_pos)
        print(f"Global path computed in {global_time:.4f}s")
        print(f"Path waypoints: {len(global_path)}")
        
        self.times.append(global_time)
        
        # Create agent and moving obstacles
        agent = self.g.createAgent(self.start_pos)
        agent.range = 2
        
        self._create_moving_obstacles()
        
        print(f"Created agent and {len(self.moving_obstacles)} moving obstacles\n")
        
        # Local planning execution
        print("Executing local planning...")
        self.local_planner = self._create_local_planner(self.local_planner_name)
        
        idx_goal = 0
        current_state = tuple(agent.location)
        previous_state = current_state
        self.complete_path = [current_state]
        
        max_steps = 1000  # Prevent infinite loops
        steps = 0
        stuck_counter = 0
        stuck_threshold = 15
        
        while idx_goal < len(global_path) and steps < max_steps:
            goal = tuple(global_path[idx_goal])
            projected_goal = self.local_planner.graph.project_to_surroundings(goal)
            
            # Get local plan
            action = self.local_planner.plan_to_goal(current_state, projected_goal)
            
            # Execute action
            agent.action = action
            self.g.next()
            self.times[0] += 1
            
            # Update state
            previous_state = current_state
            current_state = tuple(agent.location)
            self.complete_path.append(current_state)
            
            # Detect stuck: not moving
            if current_state == previous_state:
                stuck_counter += 1
                if stuck_counter > stuck_threshold:
                    # Replan globally from current position to final goal
                    print(f"  Stuck detected at {current_state}, replanning global path")
                    global_path, global_time = self.global_planner.plan(current_state, self.goal_pos)
                    self.times.append(global_time)
                    idx_goal = 0
                    stuck_counter = 0
                    continue
            else:
                stuck_counter = 0
            
            # Update observation
            obs = self.g.observe()
            self.local_planner.update_observation(agent, obs)
            
            # Track obstacles
            for obstacle in self.moving_obstacles:
                self.moving_obstacles_positions[obstacle].append(
                    tuple(obstacle.location))
            
            # Check goal reached
            if current_state == goal:
                idx_goal += 1
                stuck_counter = 0
            
            steps += 1
            
            if steps % 50 == 0:
                print(f"  Step {steps}: at {current_state}, goal {idx_goal}/{len(global_path)}")
        
        print(f"Planning completed in {self.times[0]} steps")
        print(f"Total path length: {len(self.complete_path)}")
        
        return self.times, self.complete_path
    
    def visualize(self):
        """Create animation of the path"""
        print("\nCreating visualization...")
        
        self.g.complete_path = self.complete_path
        
        figsize = 4
        fig, ax = plt.subplots(1, 1, figsize=(self.map_width/self.map_height*figsize, figsize))
        plt.title(f"Path Planning: {self.global_planner_name} + {self.local_planner_name}")
        ax.set_xlim([-0.5, self.map_width-0.5])
        ax.set_ylim([-0.5, self.map_height-0.5])
        ax.set_aspect('equal')
        ax.invert_yaxis()
        plt.tight_layout()
        
        # Plot static obstacles
        for o in self.g.obstacles:
            if self.moving_obstacles_positions.get(o) is None:
                x, y = o.location
                rx = [x-0.5, x+0.5, x+0.5, x-0.5]
                ry = [y-0.5, y-0.5, y+0.5, y+0.5]
                plt.fill(rx, ry, color='red', zorder=-2, alpha=0.8)
        
        # Plot global path
        global_path = self.global_planner.plan(self.start_pos, self.goal_pos)[0]
        for state in global_path:
            x, y = state
            rx = [x-0.5, x+0.5, x+0.5, x-0.5]
            ry = [y-0.5, y-0.5, y+0.5, y+0.5]
            plt.fill(rx, ry, color='blue', zorder=-2, alpha=0.3)
        
        moving_obstacles_artists = []
        observation_artists = []
        
        def plot_frame(i):
            nonlocal moving_obstacles_artists, observation_artists
            
            plt.title(f"Step {i}: {self.global_planner_name} + {self.local_planner_name}")
            
            # Remove previous obstacles
            for artist_list in moving_obstacles_artists:
                for artist in artist_list:
                    artist.remove()
            moving_obstacles_artists = []
            
            # Plot agent and observation
            agent = self.g.agents[0]
            x, y = self.complete_path[i]
            rx = [x-0.5, x+0.5, x+0.5, x-0.5]
            ry = [y-0.5, y-0.5, y+0.5, y+0.5]
            plt.fill(rx, ry, color='skyblue', zorder=-2, alpha=0.8)
            
            # Remove previous observation
            for artist_list in observation_artists:
                for artist in artist_list:
                    artist.remove()
            observation_artists = []
            
            ax_min = max(0, x-agent.range)
            ax_max = min(self.map_width-1, x+agent.range)
            ay_min = max(0, y-agent.range)
            ay_max = min(self.map_height-1, y+agent.range)
            rx = [ax_min-0.5, ax_max+0.5, ax_max+0.5, ax_min-0.5]
            ry = [ay_min-0.5, ay_min-0.5, ay_max+0.5, ay_max+0.5]
            obs_artist = plt.fill(rx, ry, color='lightgreen', zorder=-2, alpha=0.2)
            observation_artists.append(obs_artist)
            
            # Plot moving obstacles
            for o in self.moving_obstacles:
                pos_list = self.moving_obstacles_positions[o]
                idx = min(i, len(pos_list) - 1)
                x, y = pos_list[idx]
                rx = [x-0.5, x+0.5, x+0.5, x-0.5]
                ry = [y-0.5, y-0.5, y+0.5, y+0.5]
                artists = plt.fill(rx, ry, color='darkred', zorder=-2, alpha=0.8)
                moving_obstacles_artists.append(artists)
        
        ani = FuncAnimation(fig, plot_frame, frames=len(self.complete_path), 
                          interval=100, repeat=False)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Hierarchical Path Planning with Algorithm Comparison')
    parser.add_argument('-m', '--map', help='Map file name', default='1.png')
    parser.add_argument('-g', '--global', dest='global_planner', 
                       choices=['grid_bfs', 'grid_dfs', 'astar', 'dijkstra'],
                       default='grid_bfs', help='Global planning algorithm')
    parser.add_argument('-l', '--local', dest='local_planner',
                       choices=['reactive_bfs', 'reactive_dfs', 'potential_field', 'greedy'],
                       default='reactive_bfs', help='Local planning algorithm')
    parser.add_argument('-o', '--obstacles', type=int, default=100, help='Number of moving obstacles')
    parser.add_argument('-v', '--visualize', action='store_true', help='Show visualization')
    
    args = parser.parse_args()
    
    planner = PathPlanner(args.map, args.global_planner, args.local_planner, args.obstacles)
    times, path = planner.plan()
    
    if args.visualize:
        planner.visualize()


if __name__ == '__main__':
    main()
