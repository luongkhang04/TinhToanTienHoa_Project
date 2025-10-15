import time
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on sys.path so `src` package can be imported when running this script
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.astar import a_star, smooth_path_bezier
from src.utils import create_grid, inflate_obstacles, compute_obstacle_distance_map
from src.dwa import DWAConfig, dwa_control


def main():
    # build map
    w, h = 60, 40
    obstacles = []
    # add some random and structured obstacles
    for i in range(10, 50):
        obstacles.append((i, 20))
    for j in range(5, 15):
        obstacles.append((30, j))
    grid = create_grid(w, h, obstacles)
    # inflate obstacles to account for robot size
    grid_infl = inflate_obstacles(grid, inflation_radius=1)
    odmap = compute_obstacle_distance_map(grid_infl)

    start = (5, 5)
    goal = (55, 35)

    path = a_star(grid_infl, start, goal, allow_diagonal=True)
    if not path:
        print('No path found')
        return
    smooth = smooth_path_bezier(path, samples_per_segment=12)

    # robot state x, y, yaw, v
    x = np.array([start[0], start[1], 0.0, 0.0])
    config = DWAConfig()

    traj_x = [x[0]]
    traj_y = [x[1]]

    fig, ax = plt.subplots()

    ax.imshow(grid.T, origin='lower', cmap='Greys', interpolation='nearest')
    px = [p[0] for p in path]
    py = [p[1] for p in path]
    ax.plot(px, py, '-b', label='A* path')
    sx = [p[0] for p in smooth]
    sy = [p[1] for p in smooth]
    ax.plot(sx, sy, '-c', label='Smoothed')
    ax.plot(start[0], start[1], 'go')
    ax.plot(goal[0], goal[1], 'ro')

    # create robot plot with sequences (required for set_data calls)
    robot_plot, = ax.plot([x[0]], [x[1]], 'ro', markersize=6)

    plt.legend()
    plt.ion()
    plt.show()

    goal_reached = False
    max_iters = 1000
    it = 0
    while not goal_reached and it < max_iters:
        it += 1
        # choose local goal as a point on the smoothed path ahead
        # find nearest point on smoothed path
        dists = [math.hypot(x[0] - p[0], x[1] - p[1]) for p in smooth]
        ind = int(np.argmin(dists))
        look_ahead = min(ind + 6, len(smooth) - 1)
        local_goal = smooth[look_ahead]

        u, best_traj = dwa_control(x, config, local_goal, grid_infl, obstacle_distance_map=odmap)
        # apply control for dt
        vx = u[0]
        yawrate = u[1]
        # motion update
        x[0] += vx * math.cos(x[2]) * config.dt
        x[1] += vx * math.sin(x[2]) * config.dt
        x[2] += yawrate * config.dt
        x[3] = vx

        traj_x.append(x[0])
        traj_y.append(x[1])

        robot_plot.set_data([x[0]], [x[1]])
        ax.plot([p[0] for p in best_traj], [p[1] for p in best_traj], "-g", alpha=0.2)
        plt.pause(0.001)

        if math.hypot(x[0] - goal[0], x[1] - goal[1]) <= 1.0:
            goal_reached = True

    print('Finished after', it, 'iterations. Goal reached=', goal_reached)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
