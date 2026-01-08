"""
Global path planning algorithms for computing waypoints through the environment.
Algorithms: Grid BFS, Grid DFS, A*, Dijkstra
"""

import time
import math
import numpy as np
from collections import deque
import heapq


class GlobalPlanner:
    """Base class for global path planners"""
    
    def __init__(self, map_image, map_dimensions):
        """
        Args:
            map_image: 2D numpy array where 0 = obstacle, 1 = free
            map_dimensions: tuple (width, height)
        """
        self.map_image = map_image
        self.map_width, self.map_height = map_dimensions
        self.planning_time = 0
        self.safety_margin = 2  # cells of clearance from obstacles
        self._clearance_map = None
        
    def plan(self, start, goal):
        """
        Compute global path from start to goal.
        Returns: list of waypoints, planning time
        """
        raise NotImplementedError

    # ---- Helpers to make grid-based paths sparse (A*, Dijkstra) ----
    def _is_free_cell(self, x, y):
        return 0 <= x < self.map_width and 0 <= y < self.map_height and self.map_image[y, x] > 0

    def _bresenham_line(self, x0, y0, x1, y1):
        """Yield integer grid cells on the line from (x0,y0) to (x1,y1)."""
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            yield (x, y)
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def _has_line_of_sight(self, a, b):
        """Return True if straight line a->b crosses only free cells."""
        (x0, y0) = a
        (x1, y1) = b
        for (x, y) in self._bresenham_line(x0, y0, x1, y1):
            if not self._is_free_cell(x, y):
                return False
        return True

    def _sparsify_by_los(self, path, min_skip=1):
        """
        Compress a dense cell-by-cell path by keeping only breakpoints where
        line-of-sight breaks. Optionally skip near-adjacent points via min_skip.
        """
        if not path:
            return path
        sparse = [path[0]]
        anchor_idx = 0
        i = 1
        while i < len(path):
            # Extend as far as LOS holds from current anchor
            j = i
            while j + 1 < len(path) and self._has_line_of_sight(path[anchor_idx], path[j + 1]):
                j += 1
            # Keep the furthest visible point as next waypoint
            sparse.append(path[j])
            anchor_idx = j
            i = j + max(1, min_skip)
        # Ensure goal present
        if sparse[-1] != path[-1]:
            sparse.append(path[-1])
        return sparse

    # ---- Clearance map and margin-enforcement helpers ----
    def _ensure_clearance_map(self):
        if self._clearance_map is None:
            self._clearance_map = self._compute_clearance_map()
        return self._clearance_map

    def _compute_clearance_map(self):
        """Multi-source BFS distance (in 8-connectivity steps) to nearest obstacle."""
        from collections import deque
        H, W = self.map_height, self.map_width
        dist = np.full((H, W), fill_value=np.iinfo(np.int32).max, dtype=np.int32)
        q = deque()
        # Initialize with obstacle cells (distance 0)
        for y in range(H):
            for x in range(W):
                if self.map_image[y, x] == 0:
                    dist[y, x] = 0
                    q.append((x, y))
        # If no obstacles, return large distances
        if not q:
            return dist
        # 8-directional BFS
        dirs = [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)]
        while q:
            x, y = q.popleft()
            d = dist[y, x]
            nd = d + 1
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    q.append((nx, ny))
        return dist

    def _cell_has_margin(self, x, y, margin=None):
        cm = self._ensure_clearance_map()
        m = self.safety_margin if margin is None else margin
        # Need strictly greater than margin to keep a buffer (0=obstacle, 1=adjacent)
        return cm[y, x] > m

    def _push_away_from_walls(self, path, margin=None, search_radius_factor=2):
        """Adjust waypoints so each sits at least `margin` cells from obstacles."""
        if not path:
            return path
        m = self.safety_margin if margin is None else margin
        cm = self._ensure_clearance_map()
        adjusted = []
        for (x, y) in path:
            x = int(x); y = int(y)
            if 0 <= x < self.map_width and 0 <= y < self.map_height and cm[y, x] > m:
                adjusted.append((x, y))
                continue
            # Search a local neighborhood for best-clearance cell near original
            best = None
            best_score = None
            rmax = max(1, m * search_radius_factor)
            for r in range(1, rmax + 1):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        nx, ny = x + dx, y + dy
                        if not (0 <= nx < self.map_width and 0 <= ny < self.map_height):
                            continue
                        if self.map_image[ny, nx] == 0:
                            continue
                        c = cm[ny, nx]
                        if c > m:
                            # Prefer higher clearance, then shorter deviation
                            score = (c, -abs(dx) - abs(dy))
                            if best_score is None or score > best_score:
                                best_score = score
                                best = (nx, ny)
                if best is not None:
                    break
            adjusted.append(best if best is not None else (x, y))
        return adjusted


class GridBFSPlanner(GlobalPlanner):
    """Grid-based BFS planner (replaces removed QuadTree dependency)."""

    def __init__(self, map_image, map_dimensions, safety_margin=2):
        super().__init__(map_image, map_dimensions)
        self.safety_margin = safety_margin

    def plan(self, start, goal):
        start_time = time.perf_counter()
        path = self._bfs(start, goal)
        path = self._sparsify_by_los(path, min_skip=1)
        path = self._push_away_from_walls(path)
        self.planning_time = time.perf_counter() - start_time
        return path, self.planning_time

    def _neighbors(self, node):
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in dirs:
            nx, ny = node[0] + dx, node[1] + dy
            if not self._is_free_cell(nx, ny):
                continue
            if not self._cell_has_margin(nx, ny, margin=self.safety_margin):
                continue
            yield (nx, ny)

    def _bfs(self, start, goal):
        if not self._is_free_cell(*start) or not self._is_free_cell(*goal):
            return [start]
        q = deque([start])
        parent = {start: None}
        while q:
            node = q.popleft()
            if node == goal:
                return self._reconstruct(parent, node)
            for nb in self._neighbors(node):
                if nb in parent:
                    continue
                parent[nb] = node
                q.append(nb)
        return [start]

    def _reconstruct(self, parent, node):
        path = [node]
        while parent[node] is not None:
            node = parent[node]
            path.append(node)
        path.reverse()
        return path


class GridDFSPlanner(GlobalPlanner):
    """Grid-based DFS planner (replaces removed QuadTree dependency)."""

    def __init__(self, map_image, map_dimensions, safety_margin=2):
        super().__init__(map_image, map_dimensions)
        self.safety_margin = safety_margin

    def plan(self, start, goal):
        start_time = time.perf_counter()
        path = self._dfs(start, goal)
        path = self._sparsify_by_los(path, min_skip=1)
        path = self._push_away_from_walls(path)
        self.planning_time = time.perf_counter() - start_time
        return path, self.planning_time

    def _neighbors(self, node):
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in dirs:
            nx, ny = node[0] + dx, node[1] + dy
            if not self._is_free_cell(nx, ny):
                continue
            if not self._cell_has_margin(nx, ny, margin=self.safety_margin):
                continue
            yield (nx, ny)

    def _dfs(self, start, goal):
        if not self._is_free_cell(*start) or not self._is_free_cell(*goal):
            return [start]
        stack = [start]
        parent = {start: None}
        visited = set([start])
        while stack:
            node = stack.pop()
            if node == goal:
                return self._reconstruct(parent, node)
            for nb in self._neighbors(node):
                if nb in visited:
                    continue
                visited.add(nb)
                parent[nb] = node
                stack.append(nb)
        return [start]

    def _reconstruct(self, parent, node):
        path = [node]
        while parent[node] is not None:
            node = parent[node]
            path.append(node)
        path.reverse()
        return path


class GridAStarPlanner(GlobalPlanner):
    """A* path planning on full grid (slower but simpler baseline)"""
    
    def __init__(self, map_image, map_dimensions, safety_margin=2):
        super().__init__(map_image, map_dimensions)
        self.safety_margin = safety_margin

    def plan(self, start, goal):
        """Find path using A* on full grid"""
        start_time = time.perf_counter()
        
        path = self._astar(start, goal)
        # Make path sparse via line-of-sight shortcutting to mimic QuadTree waypoints
        path = self._sparsify_by_los(path, min_skip=1)
        # Push away from walls to respect safety margin
        path = self._push_away_from_walls(path)
        
        self.planning_time = time.perf_counter() - start_time
        return path, self.planning_time
    
    def _heuristic(self, pos, goal):
        """Manhattan distance heuristic"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def _astar(self, start, goal):
        """A* algorithm on grid"""
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            # Check 8 neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    # Check bounds
                    if not (0 <= neighbor[0] < self.map_width and 
                            0 <= neighbor[1] < self.map_height):
                        continue
                    
                    # Check obstacle
                    if self.map_image[neighbor[1], neighbor[0]] == 0:
                        continue
                    # Check safety margin
                    if not self._cell_has_margin(neighbor[0], neighbor[1], self.safety_margin):
                        continue
                    
                    # Calculate cost
                    cost = 1 if dx == 0 or dy == 0 else 1.414
                    tentative_g = g_score[current] + cost
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + self._heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, neighbor))
        
        return [start, goal]  # Fallback if no path found
    
    def _reconstruct_path(self, came_from, current):
        """Reconstruct path from A*"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))


class SimpleDijkstraPlanner(GlobalPlanner):
    """Dijkstra's algorithm - simple baseline without heuristic"""
    
    def __init__(self, map_image, map_dimensions, safety_margin=2):
        super().__init__(map_image, map_dimensions)
        self.safety_margin = safety_margin

    def plan(self, start, goal):
        """Find path using Dijkstra"""
        start_time = time.perf_counter()
        
        path = self._dijkstra(start, goal)
        # Make path sparse via line-of-sight shortcutting to mimic QuadTree waypoints
        path = self._sparsify_by_los(path, min_skip=1)
        # Push away from walls to respect safety margin
        path = self._push_away_from_walls(path)
        
        self.planning_time = time.perf_counter() - start_time
        return path, self.planning_time
    
    def _dijkstra(self, start, goal):
        """Dijkstra's algorithm on grid"""
        distances = {}
        came_from = {}
        unvisited = set()
        
        # Initialize
        for x in range(self.map_width):
            for y in range(self.map_height):
                if self.map_image[y, x] > 0:
                    distances[(x, y)] = float('inf')
                    unvisited.add((x, y))
        
        distances[start] = 0
        
        while unvisited:
            current = min(unvisited, key=lambda x: distances.get(x, float('inf')))
            
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            unvisited.remove(current)
            
            # Check neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    if neighbor not in unvisited:
                        continue
                    
                    # Check safety margin for neighbor
                    if not self._cell_has_margin(neighbor[0], neighbor[1], self.safety_margin):
                        continue
                    
                    cost = 1 if dx == 0 or dy == 0 else 1.414
                    new_distance = distances[current] + cost
                    
                    if new_distance < distances.get(neighbor, float('inf')):
                        distances[neighbor] = new_distance
                        came_from[neighbor] = current
        
        return [start, goal]  # Fallback
    
    def _reconstruct_path(self, came_from, current):
        """Reconstruct path from Dijkstra"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))
