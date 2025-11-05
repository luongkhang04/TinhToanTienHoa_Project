# Reinforcement Learning for Mobile Robot Path Planning
# Updated to use Gymnasium (new OpenAI standard)
# Using PPO (Proximal Policy Optimization) with Stable-Baselines3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import gymnasium as gym
from gymnasium import spaces
import heapq
import time

# ==================== ENVIRONMENT ====================

class GridEnvironment:
    """Grid-based environment with static and dynamic obstacles"""
    
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

# ==================== IMPROVED A* ====================

class ImprovedAStar:
    """A* algorithm with adaptive step size"""
    
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
                cost = step if dx * dy == 0 else step * np.sqrt(2)
                neighbors.append(((round(new_x, 1), round(new_y, 1)), cost))
        
        return neighbors
    
    def plan(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {start: 0}
        
        while open_list:
            current = heapq.heappop(open_list)[1]
            
            if self.heuristic(current, goal) < 1.5:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            for neighbor, move_cost in self.get_neighbors(current):
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score, neighbor))
        
        return None

# ==================== GYMNASIUM RL ENVIRONMENT ====================

class RobotNavigationEnv(gym.Env):
    """
    Gymnasium environment for robot navigation with RL
    
    Combines global path (A*) with local navigation (RL)
    
    Action Space: Box(2,) - continuous [v, w]
    Observation Space: Box(19,) - state + lidar + goal info
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 20}
    
    def __init__(self, grid_env, start, goal, render_mode=None):
        super(RobotNavigationEnv, self).__init__()
        
        self.grid_env = grid_env
        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.robot_radius = 0.3
        self.dt = 0.1
        self.render_mode = render_mode
        
        # Get global path
        planner = ImprovedAStar(grid_env)
        self.global_path = planner.plan(start, goal)
        
        # Action space: continuous (v, w)
        # v: linear velocity [0, 0.8]
        # w: angular velocity [-0.5, 0.5] rad/s
        self.action_space = spaces.Box(
            low=np.array([0.0, -0.5], dtype=np.float32),
            high=np.array([0.8, 0.5], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: 19 dimensions
        # [x, y, sin(θ), cos(θ), v, w, dx_goal, dy_goal, dist_goal, 
        #  dx_local, dy_local, dist_local, lidar_0, ..., lidar_7]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(19,),
            dtype=np.float32
        )
        
        self.max_steps = 500
        self.current_step = 0
        self.state = None
        self.current_path_idx = 0
        self.prev_dist_to_goal = None
        
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state
        
        Returns:
            observation, info (Gymnasium format)
        """
        super().reset(seed=seed)
        
        self.grid_env.reset_dynamic_obstacles()
        self.current_step = 0
        self.current_path_idx = 0
        self.prev_dist_to_goal = None
        
        # Initial state: [x, y, theta, v, w]
        self.state = [self.start[0], self.start[1], 0.0, 0.0, 0.0]
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def _get_observation(self):
        """Get current observation as 19-dim vector"""
        x, y, theta, v, w = self.state
        
        # Distance and direction to goal
        dx_goal = self.goal[0] - x
        dy_goal = self.goal[1] - y
        dist_goal = np.sqrt(dx_goal**2 + dy_goal**2)
        
        # Local goal (next waypoint on global path)
        if self.current_path_idx < len(self.global_path):
            local_goal = self.global_path[self.current_path_idx]
            dx_local = float(local_goal[0]) - x
            dy_local = float(local_goal[1]) - y
        else:
            dx_local = self.goal[0] - x
            dy_local = self.goal[1] - y
        
        dist_local = np.sqrt(dx_local**2 + dy_local**2)
        
        # Lidar scan (8 directions)
        lidar = self._get_lidar_scan(x, y, theta)
        
        # Observation vector
        obs = np.array([
            x / 20.0,           # Normalize position
            y / 20.0,
            np.sin(theta),      # Orientation (sin, cos is better than angle)
            np.cos(theta),
            v / 0.8,            # Normalize velocities
            w / 0.5,
            dx_goal / 20.0,     # Direction to goal
            dy_goal / 20.0,
            dist_goal / 30.0,   # Distance to goal
            dx_local / 20.0,    # Direction to local waypoint
            dy_local / 20.0,
            dist_local / 30.0,  # Distance to local waypoint
            *lidar              # 8 lidar readings
        ], dtype=np.float32)
        
        return obs
    
    def _get_lidar_scan(self, x, y, theta, max_range=3.0, num_rays=8):
        """Simulate 8-ray lidar scan"""
        ranges = []
        
        for i in range(num_rays):
            angle = theta + (2 * np.pi * i / num_rays)
            
            # Ray casting
            for r in np.linspace(0, max_range, 30):
                scan_x = x + r * np.cos(angle)
                scan_y = y + r * np.sin(angle)
                
                # Check static collision
                if self.grid_env.is_collision(scan_x, scan_y, 0.1):
                    ranges.append(r / max_range)
                    break
                
                # Check dynamic collision
                if self.grid_env.is_collision_dynamic(scan_x, scan_y, 0.1):
                    ranges.append(r / max_range)
                    break
            else:
                ranges.append(1.0)  # No obstacle detected
        
        return np.array(ranges, dtype=np.float32)
    
    def step(self, action):
        """
        Execute action and return next state, reward, done, truncated, info
        
        Gymnasium format returns: obs, reward, terminated, truncated, info
        """
        v, w = float(action[0]), float(action[1])
        x, y, theta, _, _ = self.state
        
        # Update robot state (simple kinematic model)
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        theta += w * self.dt
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        
        self.state = [x, y, theta, v, w]
        
        # Update dynamic obstacles
        self.grid_env.update_dynamic_obstacles(self.dt)
        
        # Calculate reward and check termination
        reward, terminated, info = self._calculate_reward()
        
        self.current_step += 1
        truncated = False
        
        # Max steps reached
        if self.current_step >= self.max_steps:
            truncated = True
            info['timeout'] = True
        
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_reward(self):
        """
        Calculate reward based on current state
        
        Returns: reward, terminated, info
        """
        x, y, theta, v, w = self.state
        reward = 0.0
        terminated = False
        info = {}
        
        # Distance to goal
        dist_to_goal = np.sqrt((x - self.goal[0])**2 + (y - self.goal[1])**2)
        
        # 1. Goal reaching reward
        if dist_to_goal < 0.8:
            reward += 1000.0
            terminated = True
            info['success'] = True
            return reward, terminated, info
        
        # 2. Collision with static obstacle
        if self.grid_env.is_collision(x, y, self.robot_radius):
            reward -= 500.0
            terminated = True
            info['collision_static'] = True
            return reward, terminated, info
        
        # 3. Collision with dynamic obstacle
        if self.grid_env.is_collision_dynamic(x, y, self.robot_radius):
            reward -= 500.0
            terminated = True
            info['collision_dynamic'] = True
            return reward, terminated, info
        
        # 4. Progress towards goal
        if self.prev_dist_to_goal is None:
            self.prev_dist_to_goal = dist_to_goal
        
        progress = self.prev_dist_to_goal - dist_to_goal
        reward += 10.0 * progress
        self.prev_dist_to_goal = dist_to_goal
        
        # 5. Close to obstacle penalty
        min_obs_dist = float('inf')
        
        # Static obstacles
        for ox, oy, w, h in self.grid_env.obstacles:
            dist = np.sqrt((x - ox - w/2)**2 + (y - oy - h/2)**2)
            min_obs_dist = min(min_obs_dist, dist)
        
        # Dynamic obstacles
        for obs in self.grid_env.dynamic_obstacles:
            dist = np.sqrt((x - obs['x'])**2 + (y - obs['y'])**2)
            min_obs_dist = min(min_obs_dist, dist - obs['radius'])
        
        if min_obs_dist < 1.0 and min_obs_dist > 0:
            reward -= 5.0 * (1.0 - min_obs_dist)
        
        # 6. Smooth motion reward (penalize sharp turns)
        reward -= 0.1 * abs(w)
        
        # 7. Time penalty (encourage efficiency)
        reward -= 1.0
        
        # 8. Local goal reaching reward
        if self.current_path_idx < len(self.global_path):
            local_goal = self.global_path[self.current_path_idx]
            dist_to_local = np.sqrt((x - float(local_goal[0]))**2 + (y - float(local_goal[1]))**2)
            
            if dist_to_local < 0.8:
                reward += 50.0  # Waypoint reached
                self.current_path_idx += 1
                info['waypoint_reached'] = True
        
        return reward, terminated, info
    
    def render(self):
        """Render environment (optional)"""
        pass
    
    def close(self):
        """Clean up"""
        pass

# ==================== TRAINING WITH STABLE-BASELINES3 ====================

def train_ppo_agent(total_timesteps=500_000):
    """
    Train PPO agent for robot navigation
    
    Requirements:
        pip install stable-baselines3[extra]
        pip install gymnasium
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    print("=" * 70)
    print("TRAINING PPO AGENT WITH GYMNASIUM")
    print("=" * 70)
    
    # Create environment
    print("\n[1/4] Creating environment...")
    grid_env = GridEnvironment(width=20, height=18)
    
    # Add obstacles
    obstacles = [(8, 2, 1, 3), (4, 8, 2, 1), (12, 9, 1, 2)]
    for ox, oy, w, h in obstacles:
        grid_env.add_obstacle(ox, oy, w, h)
    
    # Add dynamic obstacles
    grid_env.add_dynamic_obstacle(6, 6, 0.05, 0.04, radius=0.3)
    grid_env.add_dynamic_obstacle(14, 8, -0.06, 0.05, radius=0.3)
    
    start = (1.0, 1.0)
    goal = (18.0, 18.0)
    
    env = RobotNavigationEnv(grid_env, start, goal)
    
    # Check environment
    print("[2/4] Checking environment...")
    try:
        check_env(env)
        print("✓ Environment is valid!")
    except Exception as e:
        print(f"✗ Environment check failed: {e}")
        print(f"\nDebug info:")
        print(f"  Observation space shape: {env.observation_space.shape}")
        obs, info = env.reset()
        print(f"  Actual observation shape: {obs.shape}")
        print(f"  Observation: {obs}")
        return None, None
    
    # Create PPO agent
    print("\n[3/4] Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./ppo_robot_tensorboard/"
    )
    
    # Train
    print("\n[4/4] Training PPO agent...")
    print(f"Training for {total_timesteps:,} timesteps...")
    print("This will take 30-60 minutes on CPU\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        log_interval=10
    )
    
    # Save model
    model_path = "ppo_robot_navigation_gymnasium"
    model.save(model_path)
    print(f"\n✓ Model saved: {model_path}.zip")
    
    return model, env

# ==================== TESTING TRAINED AGENT ====================

def test_trained_agent(num_episodes=5):
    """Test trained PPO agent"""
    from stable_baselines3 import PPO
    
    print("\n" + "=" * 70)
    print("TESTING TRAINED PPO AGENT")
    print("=" * 70)
    
    # Create environment
    grid_env = GridEnvironment(width=20, height=18)
    obstacles = [(8, 2, 1, 3), (4, 8, 2, 1), (12, 9, 1, 2)]
    for ox, oy, w, h in obstacles:
        grid_env.add_obstacle(ox, oy, w, h)
    
    grid_env.add_dynamic_obstacle(6, 6, 0.05, 0.04, radius=0.3)
    grid_env.add_dynamic_obstacle(14, 8, -0.06, 0.05, radius=0.3)
    
    start = (1.0, 1.0)
    goal = (18.0, 18.0)
    
    env = RobotNavigationEnv(grid_env, start, goal)
    
    # Load trained model
    print("\n[1/3] Loading trained model...")
    model_path = "ppo_robot_navigation_gymnasium"
    try:
        model = PPO.load(model_path)
        print(f"✓ Model loaded: {model_path}.zip")
    except FileNotFoundError:
        print(f"✗ Model not found: {model_path}.zip")
        print("Please train model first: train_ppo_agent()")
        return None
    
    # Run test episodes
    print(f"\n[2/3] Running {num_episodes} test episodes...")
    
    all_trajectories = []
    results = {'success': 0, 'collision': 0, 'timeout': 0}
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        trajectory = [env.state[:3]]
        episode_reward = 0
        
        for step in range(500):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append(env.state[:3])
            episode_reward += reward
            
            if terminated or truncated:
                if info.get('success'):
                    results['success'] += 1
                    status = "✓ SUCCESS"
                elif info.get('collision_static') or info.get('collision_dynamic'):
                    results['collision'] += 1
                    status = "✗ COLLISION"
                elif info.get('timeout'):
                    results['timeout'] += 1
                    status = "⏱ TIMEOUT"
                
                print(f"  Episode {episode+1}: {status} (reward={episode_reward:.1f})")
                break
        
        all_trajectories.append(trajectory)
    
    # Print statistics
    print(f"\n[3/3] Test Results:")
    print(f"  Success:   {results['success']}/{num_episodes}")
    print(f"  Collision: {results['collision']}/{num_episodes}")
    print(f"  Timeout:   {results['timeout']}/{num_episodes}")
    print(f"  Success Rate: {results['success']/num_episodes*100:.1f}%")
    
    return all_trajectories, env

# ==================== VISUALIZATION ====================

def visualize_rl_trajectories(trajectories, env, title="RL Agent Navigation"):
    """Visualize RL agent trajectories"""
    num_episodes = len(trajectories)
    cols = 3
    rows = (num_episodes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if num_episodes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for ep_idx, trajectory in enumerate(trajectories):
        ax = axes[ep_idx]
        
        # Obstacles
        for ox, oy, w, h in env.grid_env.obstacles:
            rect = Rectangle((ox, oy), w, h, facecolor='dimgray', 
                            edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
        
        # Dynamic obstacles
        for obs in env.grid_env.dynamic_obstacles:
            circle = Circle((obs['x_init'], obs['y_init']), obs['radius'],
                           facecolor='red', edgecolor='darkred',
                           linewidth=1.5, alpha=0.6)
            ax.add_patch(circle)
        
        # Global path
        if env.global_path:
            gpath_x = [p[0] for p in env.global_path]
            gpath_y = [p[1] for p in env.global_path]
            ax.plot(gpath_x, gpath_y, 'orange', linewidth=2, alpha=0.5,
                   label='Global Path', linestyle='--')
        
        # Trajectory
        traj_x = [s[0] for s in trajectory]
        traj_y = [s[1] for s in trajectory]
        ax.plot(traj_x, traj_y, 'b-', linewidth=2.5, label='RL Trajectory', alpha=0.8)
        
        # Start/Goal
        ax.plot(env.start[0], env.start[1], 'go', markersize=12, label='Start', zorder=5)
        ax.plot(env.goal[0], env.goal[1], 'r*', markersize=18, label='Goal', zorder=5)
        
        ax.set_xlim(-0.5, env.grid_env.width + 0.5)
        ax.set_ylim(-0.5, env.grid_env.height + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(f'Episode {ep_idx+1}', fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    
    # Hide unused subplots
    for idx in range(num_episodes, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('rl_trajectories.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: rl_trajectories.png")
    plt.show()

# ==================== MAIN ====================

if __name__ == "__main__":
    print("=" * 70)
    print("REINFORCEMENT LEARNING FOR ROBOT NAVIGATION")
    print("Updated to use Gymnasium (OpenAI Standard)")
    print("=" * 70)
    
    print("\nInstallation:")
    print("  pip install stable-baselines3[extra]")
    print("  pip install gymnasium")
    
    print("\n" + "=" * 70)
    print("OPTIONS")
    print("=" * 70)
    
    print("\n[Option 1] Train new PPO agent (first time)")
    print("  model, env = train_ppo_agent(total_timesteps=500_000)")
    print("  # Training time: ~30-60 minutes on CPU")
    
    print("\n[Option 2] Test trained agent")
    print("  trajectories, env = test_trained_agent(num_episodes=5)")
    print("  visualize_rl_trajectories(trajectories, env)")
    
    print("\n" + "=" * 70)
    print("To run uncomment the code at the end of this script")
    print("=" * 70)
    
    # Uncomment to train:
    model, env = train_ppo_agent(total_timesteps=500_000)
    
    # Uncomment to test:
    # trajectories, env = test_trained_agent(num_episodes=5)
    # visualize_rl_trajectories(trajectories, env)