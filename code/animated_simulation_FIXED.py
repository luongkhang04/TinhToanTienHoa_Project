# Mobile Robot Path Planning - WITH REAL-TIME ANIMATION
# Pure Pursuit + Dynamic Avoidance + Live Visualization
# FIXED: Show animation window properly

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
import time
import heapq

# ==================== ENVIRONMENT ====================

class GridEnvironment:
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.obstacles = []
        self.dynamic_obstacles = []
        
    def add_obstacle(self, x, y, width, height):
        self.obstacles.append((x, y, width, height))
    
    def add_dynamic_obstacle(self, x, y, vx, vy, radius=0.5):
        self.dynamic_obstacles.append({
            'x': x, 'y': y, 'vx': vx, 'vy': vy, 'radius': radius,
            'x_init': x, 'y_init': y
        })
    
    def update_dynamic_obstacles(self, dt=0.1):
        for obs in self.dynamic_obstacles:
            obs['x'] += obs['vx'] * dt
            obs['y'] += obs['vy'] * dt
            
            if obs['x'] <= obs['radius'] or obs['x'] >= self.width - obs['radius']:
                obs['vx'] *= -1
            if obs['y'] <= obs['radius'] or obs['y'] >= self.height - obs['radius']:
                obs['vy'] *= -1
    
    def reset_dynamic_obstacles(self):
        for obs in self.dynamic_obstacles:
            obs['x'] = obs['x_init']
            obs['y'] = obs['y_init']
    
    def is_collision(self, x, y, robot_radius=0.3):
        if x < robot_radius or x >= self.width - robot_radius:
            return True
        if y < robot_radius or y >= self.height - robot_radius:
            return True
        
        for ox, oy, w, h in self.obstacles:
            if (ox - robot_radius <= x <= ox + w + robot_radius and
                oy - robot_radius <= y <= oy + h + robot_radius):
                return True
        return False
    
    def is_collision_dynamic(self, x, y, robot_radius=0.3):
        for obs in self.dynamic_obstacles:
            dist = np.sqrt((x - obs['x'])**2 + (y - obs['y'])**2)
            if dist < obs['radius'] + robot_radius:
                return True
        return False
    
    def get_repulsive_force_from_dynamic_obstacles(self, x, y, safety_dist=1.5):
        """Calculate repulsive force from dynamic obstacles"""
        force_x, force_y = 0.0, 0.0
        
        for obs in self.dynamic_obstacles:
            dx = x - obs['x']
            dy = y - obs['y']
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < safety_dist and dist > 0.01:
                magnitude = (safety_dist - dist) / (dist**2)
                force_x += magnitude * dx / dist
                force_y += magnitude * dy / dist
        
        return force_x, force_y
    
    def is_line_collision_free(self, p1, p2, robot_radius=0.3, num_checks=10):
        for i in range(num_checks + 1):
            t = i / num_checks
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if self.is_collision(x, y, robot_radius):
                return False
        return True

# ==================== IMPROVED A* ====================

class ImprovedAStar:
    def __init__(self, env):
        self.env = env
        self.step_small = 1.0
        self.step_large = 1.5
        
    def heuristic(self, pos, goal):
        return np.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)
    
    def get_obstacle_density(self, pos, radius=2.0):
        count = 0
        for ox, oy, w, h in self.env.obstacles:
            dist = np.sqrt((pos[0] - ox - w/2)**2 + (pos[1] - oy - h/2)**2)
            if dist < radius:
                count += 1
        return count
    
    def adaptive_step_size(self, pos):
        return self.step_small if self.get_obstacle_density(pos) >= 2 else self.step_large
    
    def get_neighbors(self, pos):
        step = self.adaptive_step_size(pos)
        neighbors = []
        directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
        
        for dx, dy in directions:
            new_x, new_y = pos[0] + dx * step, pos[1] + dy * step
            
            if not self.env.is_collision(new_x, new_y):
                if step > 1.0:
                    if self.env.is_line_collision_free(pos, (new_x, new_y)):
                        cost = step if dx * dy == 0 else step * np.sqrt(2)
                        neighbors.append(((round(new_x, 1), round(new_y, 1)), cost))
                else:
                    cost = step if dx * dy == 0 else step * np.sqrt(2)
                    neighbors.append(((round(new_x, 1), round(new_y, 1)), cost))
        
        return neighbors
    
    def plan(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {start: 0}
        
        explored_nodes = 0
        start_time = time.time()
        
        while open_list:
            current = heapq.heappop(open_list)[1]
            explored_nodes += 1
            
            if self.heuristic(current, goal) < 1.5:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path, explored_nodes, time.time() - start_time
            
            for neighbor, move_cost in self.get_neighbors(current):
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score, neighbor))
        
        return None, explored_nodes, time.time() - start_time

# ==================== PURE PURSUIT WITH AVOIDANCE ====================

class PurePursuitWithAvoidance:
    def __init__(self, env):
        self.env = env
        self.base_speed = 0.5
        self.max_w = 20 * np.pi / 180
        self.lookahead_distance = 2.0
        self.robot_radius = 0.3
        
        self.avoidance_gain = 0.4
        self.path_following_gain = 0.6
        
    def find_lookahead_point(self, state, path, current_idx):
        x, y = state[0], state[1]
        
        for i in range(current_idx, len(path)):
            px, py = path[i]
            dist = np.sqrt((px - x)**2 + (py - y)**2)
            
            if dist >= self.lookahead_distance:
                return path[i], i
        
        return path[-1], len(path) - 1
    
    def compute_velocity_with_avoidance(self, state, target_point):
        x, y, theta = state
        
        dx_path = target_point[0] - x
        dy_path = target_point[1] - y
        target_theta_path = np.arctan2(dy_path, dx_path)
        
        fx, fy = self.env.get_repulsive_force_from_dynamic_obstacles(x, y)
        
        if abs(fx) > 0.01 or abs(fy) > 0.01:
            avoidance_theta = np.arctan2(fy, fx)
            
            combined_dx = (self.path_following_gain * np.cos(target_theta_path) + 
                          self.avoidance_gain * np.cos(avoidance_theta))
            combined_dy = (self.path_following_gain * np.sin(target_theta_path) + 
                          self.avoidance_gain * np.sin(avoidance_theta))
            
            target_theta = np.arctan2(combined_dy, combined_dx)
        else:
            target_theta = target_theta_path
        
        angle_error = target_theta - theta
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        
        w = np.clip(angle_error * 1.5, -self.max_w, self.max_w)
        
        if abs(fx) > 0.01 or abs(fy) > 0.01:
            v = 0.25
        elif abs(angle_error) > 0.5:
            v = 0.2
        else:
            v = self.base_speed * (1 - abs(angle_error) / np.pi)
        
        return max(v, 0.1), w
    
    def motion(self, state, v, w, dt=0.1):
        x, y, theta = state
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += w * dt
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        return [x, y, theta]

# ==================== ANIMATED PLANNER ====================

class AnimatedHybridPlanner:
    def __init__(self, env):
        self.env = env
        self.global_planner = ImprovedAStar(env)
        self.local_controller = PurePursuitWithAvoidance(env)
        
        # Animation data
        self.states = []
        self.dynamic_states = []
        
    def plan_global_path(self, start, goal):
        return self.global_planner.plan(start, goal)
    
    def simulate_and_record(self, global_path, start, goal, max_steps=800):
        """Simulate and record all states for animation"""
        if global_path is None:
            return None
        
        state = [start[0], start[1], 0.0]
        current_path_idx = 0
        
        print(f"      Simulating trajectory...")
        
        for step in range(max_steps):
            self.env.update_dynamic_obstacles(dt=0.1)
            
            # Record state
            self.states.append(state[:])
            self.dynamic_states.append([
                (obs['x'], obs['y'], obs['radius']) 
                for obs in self.env.dynamic_obstacles
            ])
            
            dist_to_goal = np.sqrt((state[0] - goal[0])**2 + (state[1] - goal[1])**2)
            
            if dist_to_goal < 0.8:
                print(f"      ✓ Goal reached in {step} steps")
                return True
            
            for i in range(current_path_idx, len(global_path)):
                dist_to_waypoint = np.sqrt(
                    (state[0] - global_path[i][0])**2 +
                    (state[1] - global_path[i][1])**2
                )
                if dist_to_waypoint < 0.8:
                    current_path_idx = i
            
            target_point, _ = self.local_controller.find_lookahead_point(
                state, global_path, current_path_idx
            )
            
            v, w = self.local_controller.compute_velocity_with_avoidance(
                state, target_point
            )
            
            state = self.local_controller.motion(state, v, w)
        
        print(f"      Max steps reached")
        return False

# ==================== REAL-TIME ANIMATION ====================

def animate_simulation(env, global_path, start, goal, states, dynamic_states, save_gif=False):
    """Create real-time animation"""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Static obstacles
    for ox, oy, w, h in env.obstacles:
        rect = Rectangle((ox, oy), w, h, facecolor='dimgray', 
                        edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
    
    # Global path
    path_x = [p[0] for p in global_path]
    path_y = [p[1] for p in global_path]
    ax.plot(path_x, path_y, 'orange', linewidth=2.5, alpha=0.5, 
           label='Global Path', linestyle='--')
    
    # Start/Goal
    ax.plot(start[0], start[1], 'go', markersize=18, label='Start', zorder=5)
    ax.plot(goal[0], goal[1], 'r*', markersize=26, label='Goal', zorder=5)
    
    # Dynamic elements (will be updated)
    robot_circle = Circle((start[0], start[1]), 0.3, 
                         facecolor='blue', edgecolor='darkblue', 
                         linewidth=2, alpha=0.8, zorder=6)
    ax.add_patch(robot_circle)
    
    # Dynamic obstacles
    dyn_circles = []
    for obs in env.dynamic_obstacles:
        circle = Circle((obs['x_init'], obs['y_init']), obs['radius'],
                       facecolor='red', edgecolor='darkred',
                       linewidth=1.5, alpha=0.6, zorder=4)
        ax.add_patch(circle)
        dyn_circles.append(circle)
    
    # Trajectory trail
    trajectory_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7, 
                               label='Actual Trajectory')
    
    # Info text
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=11, verticalalignment='top',
                       family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax.set_xlim(-0.5, env.width + 0.5)
    ax.set_ylim(-0.5, env.height + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_title('Real-Time Simulation: Hybrid A* + Dynamic Avoidance', 
                fontweight='bold', fontsize=14)
    
    def update(frame):
        if frame >= len(states):
            return robot_circle, trajectory_line, info_text, *dyn_circles
        
        # Update robot position
        robot_x, robot_y = states[frame][0], states[frame][1]
        robot_circle.center = (robot_x, robot_y)
        
        # Update dynamic obstacles
        for i, circle in enumerate(dyn_circles):
            if i < len(dynamic_states[frame]):
                obs_x, obs_y, obs_r = dynamic_states[frame][i]
                circle.center = (obs_x, obs_y)
        
        # Update trajectory trail
        trail_x = [s[0] for s in states[:frame+1]]
        trail_y = [s[1] for s in states[:frame+1]]
        trajectory_line.set_data(trail_x, trail_y)
        
        # Update info
        dist_to_goal = np.sqrt((robot_x - goal[0])**2 + (robot_y - goal[1])**2)
        info_text.set_text(
            f"Step: {frame}\n"
            f"Position: ({robot_x:.2f}, {robot_y:.2f})\n"
            f"Distance to goal: {dist_to_goal:.2f} m"
        )
        
        return robot_circle, trajectory_line, info_text, *dyn_circles
    
    # Create animation (update every 50ms for smooth playback)
    anim = FuncAnimation(fig, update, frames=len(states),
                        interval=50, blit=True, repeat=True)
    
    plt.tight_layout()
    
    # Save as GIF (optional)
    if save_gif:
        print(f"\n      Saving animation...")
        try:
            anim.save('robot_simulation.gif', writer='pillow', fps=20, dpi=100)
            print(f"      ✓ Animation saved: 'robot_simulation.gif'")
        except Exception as e:
            print(f"      ✗ Could not save GIF: {e}")
    
    # Show animation window
    print(f"\n      >>> Animation window opening... <<<")
    print(f"      >>> Close window to exit <<<")
    plt.show()
    
    return anim

# ==================== SETUP ====================

def create_environment_with_dynamic():
    env = GridEnvironment(width=20, height=18)
    
    obstacles = [
        (8, 2, 1, 3),
        (4, 8, 2, 1),
        (12, 9, 1, 2),
    ]
    
    for ox, oy, w, h in obstacles:
        env.add_obstacle(ox, oy, w, h)
    
    env.add_dynamic_obstacle(6, 6, 0.08, 0.06, radius=0.3)
    env.add_dynamic_obstacle(14, 8, -0.1, 0.08, radius=0.3)
    
    return env

def run_animated_experiment(save_gif=False):
    """
    Run animated simulation
    
    Args:
        save_gif: If True, save animation as GIF (slower)
                 If False, only show real-time animation (faster)
    """
    print("=" * 70)
    print("REAL-TIME ANIMATED SIMULATION")
    print("Hybrid A* + Pure Pursuit + Dynamic Avoidance")
    print("=" * 70)
    
    env = create_environment_with_dynamic()
    start = (1.0, 1.0)
    goal = (17.0, 15.0)
    
    print(f"\n[Step 1] Global planning...")
    planner = AnimatedHybridPlanner(env)
    global_path, nodes, time_s = planner.plan_global_path(start, goal)
    
    if global_path:
        print(f"      ✓ Path found: {len(global_path)} waypoints")
    else:
        print("      ✗ No path!")
        return None
    
    print(f"\n[Step 2] Simulating trajectory...")
    env.reset_dynamic_obstacles()
    
    success = planner.simulate_and_record(global_path, start, goal, max_steps=800)
    
    if not success:
        print("      ✗ Simulation did not reach goal")
    
    print(f"\n[Step 3] Creating animation...")
    print(f"      Total frames: {len(planner.states)}")
    
    if save_gif:
        print(f"      Mode: Show + Save GIF")
    else:
        print(f"      Mode: Show only (faster)")
    
    animate_simulation(env, global_path, start, goal, 
                      planner.states, planner.dynamic_states, save_gif=save_gif)
    
    print("\n" + "=" * 70)
    print("✓ ANIMATION COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    try:
        # Set save_gif=True to save animation as GIF
        # Set save_gif=False for faster playback (no save)
        run_animated_experiment(save_gif=False)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
