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


class DynamicWindowPlanner(LocalPlanner):
    """
    Dynamic Window Approach (DWA) adapted for grid-based local planning.
    Samples velocity commands within a dynamic window and scores them by
    goal heading, clearance, and speed.
    """

    def __init__(self, agent, observation, xlim, ylim,
                 max_speed=1, max_accel=1, prediction_steps=3, dt=1.0,
                 heading_weight=1.0, clearance_weight=1.0, speed_weight=0.2,
                 clearance_cap=5, margin=2):
        super().__init__(agent, observation, xlim, ylim, margin=margin)
        self.max_speed = max(1, int(max_speed))
        self.max_accel = max(0, int(max_accel))
        self.prediction_steps = max(1, int(prediction_steps))
        self.dt = float(dt)
        self.heading_weight = float(heading_weight)
        self.clearance_weight = float(clearance_weight)
        self.speed_weight = float(speed_weight)
        self.clearance_cap = max(1, int(clearance_cap))
        self.last_action = np.array([0, 0], dtype=int)

    def plan_to_goal(self, current_state, projected_goal):
        """Select an action using DWA scoring."""
        if current_state == projected_goal:
            return move.NONE

        current = np.array(current_state, dtype=int)
        goal = np.array(projected_goal, dtype=float)

        best_score = None
        best_action = move.NONE

        for velocity in self._generate_velocity_candidates():
            score = self._score_velocity(current, goal, velocity)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_action = velocity

        if best_score is None:
            self.stuck_count += 1
            return move.NONE

        if np.array_equal(best_action, move.NONE):
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        self.last_action = np.array(best_action, dtype=int)
        return np.array(best_action, dtype=int)

    def _generate_velocity_candidates(self):
        last_vx, last_vy = int(self.last_action[0]), int(self.last_action[1])
        vx_min = max(-self.max_speed, last_vx - self.max_accel)
        vx_max = min(self.max_speed, last_vx + self.max_accel)
        vy_min = max(-self.max_speed, last_vy - self.max_accel)
        vy_max = min(self.max_speed, last_vy + self.max_accel)

        candidates = []
        for vx in range(vx_min, vx_max + 1):
            for vy in range(vy_min, vy_max + 1):
                if max(abs(vx), abs(vy)) > self.max_speed:
                    continue
                candidates.append(np.array([vx, vy], dtype=int))
        return candidates

    def _score_velocity(self, current, goal, velocity):
        trajectory = self._simulate_trajectory(current, velocity)
        if trajectory is None:
            return None

        final_pos = trajectory[-1]
        dist_to_goal = np.linalg.norm(goal - final_pos)
        heading_score = 1.0 / (dist_to_goal + 1.0)

        min_clearance = min(self._clearance_value(pos) for pos in trajectory)
        if min_clearance <= self.margin:
            return None
        clearance_score = min(min_clearance, self.clearance_cap) / float(self.clearance_cap)

        speed = np.linalg.norm(velocity)
        speed_score = speed / float(max(1, self.max_speed))

        return (self.heading_weight * heading_score +
                self.clearance_weight * clearance_score +
                self.speed_weight * speed_score)

    def _simulate_trajectory(self, current, velocity):
        if np.all(velocity == 0):
            if not self._is_valid_pos(current):
                return None
            return [current]

        trajectory = []
        for step in range(1, self.prediction_steps + 1):
            pos = current + (velocity * step)
            if not self._is_inside_global(pos):
                return None
            if not self._is_inside_observation(pos):
                break
            if not self._is_valid_pos(pos):
                return None
            trajectory.append(pos)
        if not trajectory:
            return None
        return trajectory

    def _is_valid_pos(self, pos):
        x, y = int(pos[0]), int(pos[1])
        return self._cell_has_margin((x, y), margin=self.margin)

    def _is_inside_global(self, pos):
        x, y = int(pos[0]), int(pos[1])
        return 0 <= x < self.xlim and 0 <= y < self.ylim

    def _is_inside_observation(self, pos):
        gx, gy = int(pos[0]), int(pos[1])
        agent_loc = self.agent.location
        agent_range = self.agent.range

        obs_x_min = max(0, int(agent_loc[0]) - agent_range)
        obs_y_min = max(0, int(agent_loc[1]) - agent_range)
        obs_x_max = min(self.xlim - 1, int(agent_loc[0]) + agent_range)
        obs_y_max = min(self.ylim - 1, int(agent_loc[1]) + agent_range)

        return obs_x_min <= gx <= obs_x_max and obs_y_min <= gy <= obs_y_max

    def _clearance_value(self, pos):
        gx, gy = int(pos[0]), int(pos[1])
        agent_loc = self.agent.location
        agent_range = self.agent.range

        obs_x_min = max(0, int(agent_loc[0]) - agent_range)
        obs_y_min = max(0, int(agent_loc[1]) - agent_range)

        lx = gx - obs_x_min
        ly = gy - obs_y_min

        W, H = self.observation.shape
        if not (0 <= lx < W and 0 <= ly < H):
            return -1
        cm = self._ensure_clearance_map()
        return int(cm[lx, ly])


class EvolutionaryLocalPlanner(LocalPlanner):
    """
    Rolling-horizon evolutionary planner.
    Optimizes a short action sequence and executes the first action.
    """

    def __init__(self, agent, observation, xlim, ylim,
                 horizon=6, population_size=40, generations=8,
                 mutation_rate=0.15, elite_frac=0.2, goal_bias=0.6,
                 goal_weight=2.0, clearance_weight=0.5, turn_weight=0.2,
                 stall_weight=0.6, unknown_weight=0.4, step_weight=0.2,
                 increase_weight=1.0, first_step_weight=1.0, seed_bfs=True,
                 margin=2):
        super().__init__(agent, observation, xlim, ylim, margin=margin)
        self.horizon = max(1, int(horizon))
        self.population_size = max(2, int(population_size))
        self.generations = max(1, int(generations))
        self.mutation_rate = float(mutation_rate)
        self.elite_frac = max(0.0, min(1.0, float(elite_frac)))
        self.goal_bias = max(0.0, min(1.0, float(goal_bias)))
        self.goal_weight = float(goal_weight)
        self.clearance_weight = float(clearance_weight)
        self.turn_weight = float(turn_weight)
        self.stall_weight = float(stall_weight)
        self.unknown_weight = float(unknown_weight)
        self.step_weight = float(step_weight)
        self.increase_weight = float(increase_weight)
        self.first_step_weight = float(first_step_weight)
        self.seed_bfs = bool(seed_bfs)
        self.action_set = [
            move.NONE, move.LEFT, move.RIGHT, move.UP, move.DOWN,
            move.NE, move.NW, move.SE, move.SW,
        ]
        self._none_action_idx = 0
        self._action_index = {
            (int(action[0]), int(action[1])): idx
            for idx, action in enumerate(self.action_set)
        }

    def plan_to_goal(self, current_state, projected_goal):
        if current_state == projected_goal:
            return move.NONE

        population = self._init_population(current_state, projected_goal)
        best_seq = None
        best_score = None

        for _ in range(self.generations):
            scored = []
            for seq in population:
                score = self._evaluate_sequence(seq, current_state, projected_goal)
                if score is None:
                    continue
                scored.append((score, seq))
                if best_score is None or score < best_score:
                    best_score = score
                    best_seq = seq

            if not scored:
                population = self._init_population(current_state, projected_goal)
                continue

            scored.sort(key=lambda item: item[0])
            elite_count = max(1, int(self.elite_frac * self.population_size))
            elite_count = min(elite_count, len(scored))
            elites = [seq for _, seq in scored[:elite_count]]

            new_population = list(elites)
            while len(new_population) < self.population_size:
                parent_a = self._tournament_select(scored)
                parent_b = self._tournament_select(scored)
                child = self._crossover(parent_a, parent_b)
                child = self._mutate(child, current_state, projected_goal)
                new_population.append(child)
            population = new_population

        if best_seq is None:
            self.stuck_count += 1
            return move.NONE

        action = self.action_set[int(best_seq[0])]
        if np.array_equal(action, move.NONE):
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        return np.array(action, dtype=int)

    def _init_population(self, current_state, goal):
        current = np.array(current_state, dtype=int)
        goal = np.array(goal, dtype=int)
        population = []
        if self.seed_bfs:
            seed = self._seed_from_bfs(current_state, goal)
            if seed is not None:
                population.append(seed)
                if len(population) < self.population_size:
                    population.append(self._mutate(seed, current_state, goal))

        remaining = self.population_size - len(population)
        for _ in range(remaining):
            seq = np.zeros(self.horizon, dtype=int)
            pos = current.copy()
            for step in range(self.horizon):
                action_idx = self._sample_action_idx(pos, goal)
                seq[step] = action_idx
                pos = pos + self.action_set[action_idx]
            population.append(seq)
        return population

    def _seed_from_bfs(self, current_state, goal):
        if self.graph is None:
            return None
        goal_state = (int(goal[0]), int(goal[1]))
        path = self.graph.get_optimal_path(current_state, goal_state)
        if not path or len(path) < 2 or path[0] == path[1]:
            return None
        seq = np.zeros(self.horizon, dtype=int)
        for i in range(self.horizon):
            if i + 1 < len(path):
                delta = np.array(path[i + 1]) - np.array(path[i])
                seq[i] = self._action_to_index(delta)
            else:
                seq[i] = self._none_action_idx
        return seq

    def _action_to_index(self, action):
        key = (int(action[0]), int(action[1]))
        return self._action_index.get(key, self._none_action_idx)

    def _sample_action_idx(self, pos, goal):
        valid = self._valid_action_indices(pos)
        if len(valid) == 1 or np.random.random() > self.goal_bias:
            return int(np.random.choice(valid))

        best_dist = None
        best_actions = []
        for idx in valid:
            next_pos = pos + self.action_set[idx]
            dist = abs(goal[0] - next_pos[0]) + abs(goal[1] - next_pos[1])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_actions = [idx]
            elif dist == best_dist:
                best_actions.append(idx)
        return int(np.random.choice(best_actions))

    def _valid_action_indices(self, pos):
        valid = []
        for idx, action in enumerate(self.action_set):
            next_pos = pos + action
            if not self._is_inside_global(next_pos):
                continue
            if self._is_inside_observation(next_pos):
                if not self._cell_has_margin(next_pos, margin=self.margin):
                    continue
            valid.append(idx)
        return valid if valid else [self._none_action_idx]

    def _evaluate_sequence(self, seq, current_state, goal):
        pos = np.array(current_state, dtype=int)
        goal = np.array(goal, dtype=int)
        turns = 0
        stalls = 0
        unknown = 0
        clearance_cost = 0.0
        prev_action = None
        min_dist = None
        dist_sum = 0
        increase_penalty = 0.0
        first_step_penalty = 0.0
        prev_dist = abs(goal[0] - pos[0]) + abs(goal[1] - pos[1])

        for step_idx, idx in enumerate(seq):
            action = self.action_set[int(idx)]
            if np.array_equal(action, move.NONE):
                stalls += 1

            next_pos = pos + action
            if not self._is_inside_global(next_pos):
                return None

            if self._is_inside_observation(next_pos):
                if not self._cell_has_margin(next_pos, margin=self.margin):
                    return None
                clearance = self._clearance_value(next_pos)
                if clearance > 0:
                    clearance_cost += 1.0 / (clearance + 1.0)
            else:
                unknown += 1

            if prev_action is not None:
                if (not np.array_equal(prev_action, move.NONE) and
                        not np.array_equal(action, move.NONE) and
                        not np.array_equal(prev_action, action)):
                    turns += 1

            pos = next_pos
            dist = abs(goal[0] - pos[0]) + abs(goal[1] - pos[1])
            dist_sum += dist
            if dist > prev_dist:
                increase_penalty += (dist - prev_dist)
            if step_idx == 0 and dist >= prev_dist:
                first_step_penalty = (dist - prev_dist) + 1.0
            min_dist = dist if min_dist is None else min(min_dist, dist)
            prev_action = action
            prev_dist = dist

        final_dist = abs(goal[0] - pos[0]) + abs(goal[1] - pos[1])
        min_dist = final_dist if min_dist is None else min_dist
        dist_score = 0.7 * final_dist + 0.3 * min_dist
        avg_dist = dist_sum / float(len(seq)) if len(seq) > 0 else 0.0

        return (self.goal_weight * dist_score +
                self.step_weight * avg_dist +
                self.increase_weight * increase_penalty +
                self.first_step_weight * first_step_penalty +
                self.clearance_weight * clearance_cost +
                self.turn_weight * turns +
                self.stall_weight * stalls +
                self.unknown_weight * unknown)

    def _tournament_select(self, scored, k=3):
        k = min(k, len(scored))
        indices = np.random.choice(len(scored), size=k, replace=False)
        best_seq = None
        best_score = None
        for idx in indices:
            score, seq = scored[idx]
            if best_score is None or score < best_score:
                best_score = score
                best_seq = seq
        return best_seq

    def _crossover(self, parent_a, parent_b):
        if self.horizon == 1:
            return parent_a.copy()
        point = np.random.randint(1, self.horizon)
        return np.concatenate([parent_a[:point], parent_b[point:]])

    def _mutate(self, seq, current_state, goal):
        if self.mutation_rate <= 0:
            return seq
        pos = np.array(current_state, dtype=int)
        goal = np.array(goal, dtype=int)
        mutated = seq.copy()
        for step in range(self.horizon):
            if np.random.random() < self.mutation_rate:
                mutated[step] = self._sample_action_idx(pos, goal)
            pos = pos + self.action_set[int(mutated[step])]
        return mutated

    def _is_inside_global(self, pos):
        x, y = int(pos[0]), int(pos[1])
        return 0 <= x < self.xlim and 0 <= y < self.ylim

    def _is_inside_observation(self, pos):
        gx, gy = int(pos[0]), int(pos[1])
        agent_loc = self.agent.location
        agent_range = self.agent.range

        obs_x_min = max(0, int(agent_loc[0]) - agent_range)
        obs_y_min = max(0, int(agent_loc[1]) - agent_range)
        obs_x_max = min(self.xlim - 1, int(agent_loc[0]) + agent_range)
        obs_y_max = min(self.ylim - 1, int(agent_loc[1]) + agent_range)

        return obs_x_min <= gx <= obs_x_max and obs_y_min <= gy <= obs_y_max

    def _clearance_value(self, pos):
        gx, gy = int(pos[0]), int(pos[1])
        agent_loc = self.agent.location
        agent_range = self.agent.range

        obs_x_min = max(0, int(agent_loc[0]) - agent_range)
        obs_y_min = max(0, int(agent_loc[1]) - agent_range)

        lx = gx - obs_x_min
        ly = gy - obs_y_min

        W, H = self.observation.shape
        if not (0 <= lx < W and 0 <= ly < H):
            return -1
        cm = self._ensure_clearance_map()
        return int(cm[lx, ly])
