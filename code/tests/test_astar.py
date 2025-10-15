import numpy as np
from src.astar import a_star, smooth_path_bezier


def test_a_star_simple():
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[2, 1] = 1
    grid[2, 2] = 1
    grid[2, 3] = 1
    start = (0, 0)
    goal = (4, 4)
    path = a_star(grid, start, goal, allow_diagonal=True)
    assert path, "A* should find a path in open grid with a small wall"
    # ensure path starts and ends correctly
    assert path[0] == start
    assert path[-1] == goal


def test_smooth_path_preserves_endpoints():
    path = [(0, 0), (1, 0), (2, 0), (3, 0)]
    smooth = smooth_path_bezier(path, samples_per_segment=10)
    assert abs(smooth[0][0] - path[0][0]) < 1e-6
    assert abs(smooth[-1][0] - path[-1][0]) < 1e-6
