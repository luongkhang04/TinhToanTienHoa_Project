import numpy as np
from src.dwa import predict_trajectory, DWAConfig


def test_predict_trajectory_forward():
    config = DWAConfig()
    x_init = [0.0, 0.0, 0.0, 0.0]
    traj = predict_trajectory(x_init, v=0.5, yrate=0.0, config=config)
    # ensure trajectory length > 1 and robot moved forward in x
    assert traj.shape[0] > 1
    assert traj[-1, 0] > 0.0


def test_predict_trajectory_turn():
    config = DWAConfig()
    x_init = [0.0, 0.0, 0.0, 0.0]
    traj = predict_trajectory(x_init, v=0.2, yrate=0.5, config=config)
    # trajectory yaw should change
    assert abs(traj[-1, 2] - traj[0, 2]) > 0.0
