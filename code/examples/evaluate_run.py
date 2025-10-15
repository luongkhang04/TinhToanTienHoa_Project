import time
import math
import os
import sys
import numpy as np

# ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.astar import a_star, smooth_path_bezier
from src.utils import create_grid, inflate_obstacles, compute_obstacle_distance_map
from src.dwa import DWAConfig, dwa_control


def run_sim(verbose=False):
    w, h = 60, 40
    obstacles = []
    for i in range(10, 50):
        obstacles.append((i, 20))
    for j in range(5, 15):
        obstacles.append((30, j))
    grid = create_grid(w, h, obstacles)
    grid_infl = inflate_obstacles(grid, inflation_radius=1)
    odmap = compute_obstacle_distance_map(grid_infl)

    start = (5, 5)
    goal = (55, 35)

    path = a_star(grid_infl, start, goal, allow_diagonal=True)
    if not path:
        raise RuntimeError('No path found')
    smooth = smooth_path_bezier(path, samples_per_segment=12)

    x = np.array([start[0], start[1], 0.0, 0.0])
    config = DWAConfig()

    traj = []
    best_trajs = []

    start_time = time.time()
    it = 0
    goal_reached = False
    max_iters = 2000
    while not goal_reached and it < max_iters:
        it += 1
        dists = [math.hypot(x[0] - p[0], x[1] - p[1]) for p in smooth]
        ind = int(np.argmin(dists))
        look_ahead = min(ind + 6, len(smooth) - 1)
        local_goal = smooth[look_ahead]

        u, best_traj = dwa_control(x, config, local_goal, grid_infl, obstacle_distance_map=odmap)
        vx = u[0]
        yawrate = u[1]
        x[0] += vx * math.cos(x[2]) * config.dt
        x[1] += vx * math.sin(x[2]) * config.dt
        x[2] += yawrate * config.dt
        x[3] = vx

        traj.append((x[0], x[1], x[2], x[3]))
        best_trajs.append(best_traj)

        if math.hypot(x[0] - goal[0], x[1] - goal[1]) <= 1.0:
            goal_reached = True

    elapsed = time.time() - start_time

    # compute metrics
    iterations = it
    total_distance = 0.0
    for i in range(1, len(traj)):
        dx = traj[i][0] - traj[i-1][0]
        dy = traj[i][1] - traj[i-1][1]
        total_distance += math.hypot(dx, dy)
    avg_speed = sum(abs(p[3]) for p in traj) / max(1, len(traj))

    # min clearance from obstacle distance map at robot locations
    min_clearance = float('inf')
    for p in traj:
        cx, cy = int(round(p[0])), int(round(p[1]))
        if 0 <= cx < grid_infl.shape[0] and 0 <= cy < grid_infl.shape[1]:
            d = odmap[cx, cy]
            if d < min_clearance:
                min_clearance = d

    result = {
        'iterations': iterations,
        'elapsed_s': elapsed,
        'total_distance': total_distance,
        'avg_speed': avg_speed,
        'min_clearance': min_clearance,
        'goal_reached': goal_reached,
    }

    if verbose:
        print(result)
    return result


if __name__ == '__main__':
    res = run_sim(verbose=True)
