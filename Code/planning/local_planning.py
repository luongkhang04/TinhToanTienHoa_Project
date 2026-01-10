"""
Local path planning algorithms for reactive navigation in agent's observation.
Handles moving obstacles and real-time replanning.
"""

import numpy as np
from utils.agent_graph import agent_graph
from utils import move


class LocalPlanner:
    """Base class for local planners"""
    
    def __init__(self, agent, observation, xlim, ylim, margin=2):
        """
        Args:
            agent: Agent object with location and range
            observation: Current obstacle observation (0 free, >0 obstacle)
            xlim, ylim: Environment dimensions
            margin: Safety margin (cells) to keep from obstacles
        """
        self.agent = agent
        self.observation = observation
        self.xlim = xlim
        self.ylim = ylim
        self.margin = max(0, int(margin))
        self.graph = agent_graph(agent, observation, xlim, ylim, inflation=self.margin)
        self.stuck_count = 0
        self.stuck_threshold = 10
        self._clearance_map = None
        
    def plan_to_goal(self, current_state, projected_goal):
        """
        Compute action to reach goal from current state.
        Returns: action (movement vector)
        """
        raise NotImplementedError
    
    def update_observation(self, agent, observation):
        """Update local graph with new observation"""
        self.agent = agent
        self.observation = np.copy(observation)
        self.graph = agent_graph(agent, observation, self.xlim, self.ylim, inflation=self.margin)
        self.stuck_count = 0
        self._clearance_map = None
    
    def handle_stuck(self, idx_goal):
        """
        Handle stuck situation. Called when agent doesn't move for threshold steps.
        Returns: updated goal index
        """
        if self.stuck_count > self.stuck_threshold and idx_goal >= 1:
            idx_goal -= 1
            self.stuck_count = 0
        return idx_goal

    # ---- Clearance helpers ----
    def _ensure_clearance_map(self):
        if self._clearance_map is None:
            self._clearance_map = self._compute_clearance_map()
        return self._clearance_map

    def _compute_clearance_map(self):
        """Distance (Chebyshev) to nearest obstacle within observation; out-of-bounds treated as obstacle.

        Note: Global.observe() returns grid slices in (x, y) order. We keep that order
        here to stay consistent with agent_graph, so we index obs[x, y].
        """
        obs = self.observation  # shape: (W, H) with x-first indexing
        W, H = obs.shape
        dist = np.full((W, H), fill_value=np.iinfo(np.int16).max, dtype=np.int16)
        from collections import deque
        q = deque()
        # obstacles are encoded as 1; ignore agent cells (value 2)
        for x in range(W):
            for y in range(H):
                if obs[x, y] == 1:
                    dist[x, y] = 0
                    q.append((x, y))
        if not q:
            return dist
        dirs = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
        while q:
            x, y = q.popleft()
            nd = dist[x, y] + 1
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and nd < dist[nx, ny]:
                    dist[nx, ny] = nd
                    q.append((nx, ny))
        return dist

    def _cell_has_margin(self, pos, margin=None):
        """Check clearance - pos is in GLOBAL coordinates, converts to local"""
        gx, gy = int(pos[0]), int(pos[1])
        m = self.margin if margin is None else margin
        
        # Convert to local observation coordinates
        agent_loc = self.agent.location
        agent_range = self.agent.range
        
        obs_x_min = max(0, int(agent_loc[0]) - agent_range)
        obs_y_min = max(0, int(agent_loc[1]) - agent_range)
        
        lx = gx - obs_x_min
        ly = gy - obs_y_min
        
        W, H = self.observation.shape
        if not (0 <= lx < W and 0 <= ly < H):
            return False
        cm = self._ensure_clearance_map()
        return cm[lx, ly] > m


class ReactiveBFSPlanner(LocalPlanner):
    """Local BFS planner - reactive navigation with observation"""
    
    def plan_to_goal(self, current_state, projected_goal):
        """Find path to goal using BFS in local observation"""
        # If current state and goal are same, don't move
        if current_state == projected_goal:
            return move.NONE
            
        inter_path = self.graph.get_optimal_path(current_state, projected_goal)
        
        if len(inter_path) < 2:
            # No path found - try to move towards goal anyway
            goal_direction = np.array(projected_goal) - np.array(current_state)
            if np.linalg.norm(goal_direction) > 0:
                # Normalize and try to move one step
                action = np.sign(goal_direction).astype(int)
                return action
            return move.NONE
        
        next_state = inter_path[1]
        action = np.array(next_state) - np.array(current_state)
        
        return action


class ReactiveDFSPlanner(LocalPlanner):
    """Local DFS planner - depth-first search in observation"""
    
    def plan_to_goal(self, current_state, projected_goal):
        """Find path to goal using DFS in local observation"""
        inter_path = self.graph.get_path_DFS(current_state, projected_goal, [], set())
        
        if len(inter_path) < 2:
            return move.NONE
        
        next_state = inter_path[1]
        action = np.array(next_state) - np.array(current_state)
        
        # Track stuck status
        if np.array_equal(action, move.NONE):
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        
        return action


class PotentialFieldPlanner(LocalPlanner):
    """
    Local planner using artificial potential fields.
    Attractive force towards goal, repulsive force from obstacles.
    """
    
    def __init__(self, agent, observation, xlim, ylim, 
                 attraction_weight=1.0, repulsion_weight=2.0, repulsion_range=3, margin=2):
        super().__init__(agent, observation, xlim, ylim, margin=margin)
        self.attraction_weight = attraction_weight
        self.repulsion_weight = repulsion_weight
        self.repulsion_range = repulsion_range
    
    def plan_to_goal(self, current_state, projected_goal):
        """Compute action using potential field"""
        current = np.array(current_state, dtype=float)
        goal = np.array(projected_goal, dtype=float)
        
        # Attractive force towards goal
        direction_to_goal = goal - current
        distance_to_goal = np.linalg.norm(direction_to_goal)
        
        if distance_to_goal < 1e-6:
            return move.NONE
        
        attractive_force = (direction_to_goal / distance_to_goal) * self.attraction_weight
        
        # Repulsive force from obstacles
        repulsive_force = np.array([0.0, 0.0])
        
        for dx in range(-self.repulsion_range, self.repulsion_range + 1):
            for dy in range(-self.repulsion_range, self.repulsion_range + 1):
                obstacle_pos = current + np.array([dx, dy])
                
                # Check if obstacle position is within observation bounds and has obstacle
                if self._has_obstacle(obstacle_pos):
                    direction_away = current - obstacle_pos
                    distance = np.linalg.norm(direction_away)
                    
                    if distance < 1e-6:
                        distance = 1e-3
                    
                    repulsive_force += (direction_away / distance) * self.repulsion_weight
        
        # Combine forces
        total_force = attractive_force + repulsive_force
        
        # Convert to grid action
        action = self._force_to_action(total_force)
        
        # Track stuck status
        if np.array_equal(action, move.NONE):
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        
        return action
    
    def _has_obstacle(self, pos):
        """Check if position violates margin (treat low-clearance as obstacle)."""
        x, y = int(pos[0]), int(pos[1])
        # Out of world or out of observation => obstacle
        obs_shape = self.observation.shape
        if not (0 <= x < obs_shape[1] and 0 <= y < obs_shape[0]):
            return True
        return not self._cell_has_margin((x, y), margin=self.margin)
    
    def _force_to_action(self, force):
        """Convert continuous force vector to discrete grid action"""
        if np.linalg.norm(force) < 1e-6:
            return move.NONE
        
        # Find direction of maximum force component
        abs_force = np.abs(force)
        
        if abs_force[0] > abs_force[1]:
            # Horizontal dominates
            if force[0] > 0:
                return move.RIGHT
            else:
                return move.LEFT
        elif abs_force[1] > 0:
            # Vertical dominates
            if force[1] > 0:
                return move.UP
            else:
                return move.DOWN
        else:
            return move.NONE


class GreedyLocalPlanner(LocalPlanner):
    """
    Greedy planner - always move closer to goal if possible,
    avoiding obstacles. Good for simple environments.
    """
    
    def plan_to_goal(self, current_state, projected_goal):
        """Greedily move towards goal"""
        current = np.array(current_state)
        goal = np.array(projected_goal)
        
        # Calculate direction to goal
        direction = goal - current
        
        # Try moving in primary direction
        for move_vector in [move.RIGHT, move.LEFT, move.UP, move.DOWN, 
                            move.NE, move.NW, move.SE, move.SW]:
            next_pos = current + move_vector
            
            # Check if valid and reduces distance to goal
            if self._is_valid_move(next_pos):
                new_distance = np.linalg.norm(goal - next_pos)
                old_distance = np.linalg.norm(direction)
                
                if new_distance < old_distance:
                    self.stuck_count = 0
                    return move_vector
        
        # No improving move found, try any valid move
        for move_vector in [move.RIGHT, move.LEFT, move.UP, move.DOWN, 
                            move.NE, move.NW, move.SE, move.SW]:
            next_pos = current + move_vector
            if self._is_valid_move(next_pos):
                self.stuck_count += 1
                return move_vector
        
        self.stuck_count += 1
        return move.NONE
    
    def _is_valid_move(self, pos):
        """Check if move to position is valid (respects margin, in bounds)
        
        pos is in global coordinates, need to convert to local observation coords
        """
        gx, gy = int(pos[0]), int(pos[1])
        
        # Check global bounds first
        if not (0 <= gx < self.xlim and 0 <= gy < self.ylim):
            return False
        
        # Convert to local observation coordinates
        agent_loc = self.agent.location
        agent_range = self.agent.range
        
        # Local observation bounds
        obs_x_min = max(0, int(agent_loc[0]) - agent_range)
        obs_y_min = max(0, int(agent_loc[1]) - agent_range)
        
        # Local coordinates within observation
        lx = gx - obs_x_min
        ly = gy - obs_y_min
        
        W, H = self.observation.shape
        if not (0 <= lx < W and 0 <= ly < H):
            return False
        
        # Check if cell is free (not obstacle)
        if self.observation[lx, ly] == 1:
            return False
        
        # Check margin using local coordinates
        return self._cell_has_margin_local(lx, ly, margin=self.margin)
    
    def _cell_has_margin_local(self, lx, ly, margin=None):
        """Check clearance using local observation coordinates"""
        m = self.margin if margin is None else margin
        W, H = self.observation.shape
        if not (0 <= lx < W and 0 <= ly < H):
            return False
        cm = self._ensure_clearance_map()
        return cm[lx, ly] > m
