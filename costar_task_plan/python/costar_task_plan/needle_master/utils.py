import numpy as np
import copy


def RotateTrajectory(traj, angle):
    traj2 = copy.copy(traj)
    R = np.array(
        [[np.cos(angle), -1 * np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    traj2[:, :2] = traj[:, :2].dot(R)
    traj2[:, 3] += angle
