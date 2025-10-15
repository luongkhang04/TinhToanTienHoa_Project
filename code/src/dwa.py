import math
import numpy as np
from src.utils import point_to_cell


class DWAConfig:
    def __init__(self):
        # robot limits
        self.max_speed = 1.2  # m/s
        # allow stronger reverse to enable backing away when stuck
        self.min_speed = -0.5
        self.max_yawrate = 40.0 * math.pi / 180.0
        self.max_accel = 0.4
        self.max_delta_yawrate = 40.0 * math.pi / 180.0
        # simulation
        self.dt = 0.1
        self.predict_time = 2.0
        # objective weights
        self.to_goal_cost_gain = 1.0
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        # robot radius
        self.robot_radius = 0.5


def motion(x, u, dt):
    # x = [x, y, yaw, v]
    x_new = x.copy()
    x_new[0] += u[0] * math.cos(x[2]) * dt
    x_new[1] += u[0] * math.sin(x[2]) * dt
    x_new[2] += u[1] * dt
    x_new[3] = u[0]
    return x_new


def calc_dynamic_window(x, config):
    # v_min, v_max, yawrate_min, yawrate_max
    vs = [config.min_speed, config.max_speed, -config.max_yawrate, config.max_yawrate]
    # based on current state x[3] is current v
    v = x[3]
    dw = [v - config.max_accel * config.dt, v + config.max_accel * config.dt,
          -config.max_delta_yawrate * config.dt, config.max_delta_yawrate * config.dt]
    # intersect
    dw0 = [max(vs[0], dw[0]), min(vs[1], dw[1]), max(vs[2], dw[2]), min(vs[3], dw[3])]
    return dw0


def predict_trajectory(x_init, v, yrate, config):
    x = np.array(x_init, dtype=float)
    traj = [x[:3].copy()]
    time = 0.0
    while time <= config.predict_time:
        u = [v, yrate]
        x = motion(x, u, config.dt)
        traj.append(x[:3].copy())
        time += config.dt
    return np.array(traj)


def calc_to_goal_cost(traj, goal):
    dx = goal[0] - traj[-1, 0]
    dy = goal[1] - traj[-1, 1]
    error_angle = math.atan2(dy, dx)
    traj_yaw = traj[-1, 2]
    cost = abs(math.atan2(math.sin(error_angle - traj_yaw), math.cos(error_angle - traj_yaw)))
    return cost


def calc_obstacle_cost(traj, grid, config, obstacle_distance_map=None):
    # if obstacle_distance_map provided, use it; otherwise convert grid
    min_r = float('inf')
    w, h = grid.shape
    for (x, y, _) in traj:
        cx, cy = point_to_cell((x, y))
        if 0 <= cx < w and 0 <= cy < h:
            if grid[cx, cy] == 1:
                return float('inf')
            if obstacle_distance_map is not None:
                d = obstacle_distance_map[cx, cy]
            else:
                # fallback: distance to nearest obstacle cell by checking neighbors up to some radius
                d = 0.0
            if d < min_r:
                min_r = d
        else:
            return float('inf')
    if min_r == float('inf'):
        return 0.0
    # penalize trajectories that bring robot close to obstacles
    if min_r <= 0.0:
        return float('inf')
    return 1.0 / min_r


def dwa_control(x, config, goal, grid, obstacle_distance_map=None):
    dw = calc_dynamic_window(x, config)
    best_u = [0.0, 0.0]
    best_score = -float('inf')
    best_traj = np.array([x[:3]])

    v_min, v_max, y_min, y_max = dw
    # sample space: linear sampling
    v_samples = np.linspace(v_min, v_max, num=5)
    y_samples = np.linspace(y_min, y_max, num=7)

    for v in v_samples:
        for y in y_samples:
            traj = predict_trajectory(x, v, y, config)
            to_goal_cost = config.to_goal_cost_gain * (1.0 - calc_to_goal_cost(traj, goal) / math.pi)
            speed_cost = config.speed_cost_gain * (v / config.max_speed)
            ob_cost_raw = calc_obstacle_cost(traj, grid, config, obstacle_distance_map)
            if ob_cost_raw == float('inf'):
                continue
            ob_cost = config.obstacle_cost_gain * (1.0 / (1.0 + ob_cost_raw))
            score = to_goal_cost + speed_cost + ob_cost
            if score > best_score:
                best_score = score
                best_u = [v, y]
                best_traj = traj
    # If no feasible candidate found (e.g., surrounded or inside obstacle),
    # return a safe backup command so the robot can try to retreat.
    if best_score == -float('inf'):
        # choose a small reverse velocity and zero yawrate as a simple retreat
        backup_v = config.min_speed
        backup_y = 0.0
        backup_traj = predict_trajectory(x, backup_v, backup_y, config)
        return [backup_v, backup_y], backup_traj

    return best_u, best_traj
