import numpy as np


def create_grid(width, height, obstacles):
    """Create a grid with obstacles. obstacles is list of (x,y) cells to set as occupied."""
    grid = np.zeros((width, height), dtype=np.uint8)
    for (x, y) in obstacles:
        if 0 <= x < width and 0 <= y < height:
            grid[x, y] = 1
    return grid


def inflate_obstacles(grid, inflation_radius=1):
    """Naive inflation: mark cells within manhattan distance <= r as obstacles."""
    w, h = grid.shape
    out = grid.copy()
    obs = np.argwhere(grid == 1)
    for (x, y) in obs:
        for dx in range(-inflation_radius, inflation_radius + 1):
            for dy in range(-inflation_radius, inflation_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    out[nx, ny] = 1
    return out


def compute_obstacle_distance_map(grid):
    """Return a float array same shape as grid with Euclidean distance to nearest obstacle.
    This is a simple O(n*m*k) implementation but fine for small maps.
    """
    w, h = grid.shape
    out = np.full((w, h), np.inf, dtype=float)
    obs = np.argwhere(grid == 1)
    if obs.size == 0:
        out[:] = np.inf
        return out
    for x in range(w):
        for y in range(h):
            if grid[x, y] == 1:
                out[x, y] = 0.0
            else:
                d = np.min(np.hypot(obs[:, 0] - x, obs[:, 1] - y))
                out[x, y] = d
    return out


def in_bounds(grid, x, y):
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]


def point_to_cell(pt):
    """Convert continuous (x,y) to integer cell (round)."""
    return (int(round(pt[0])), int(round(pt[1])))
