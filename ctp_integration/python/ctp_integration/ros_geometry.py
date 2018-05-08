from __future__ import print_function
import numpy as np

def pose_to_vec_quat_pair(c_pose):
    c_xyz = np.array([c_pose.transform.translation.x,
                c_pose.transform.translation.y,
                c_pose.transform.translation.z,])
    c_quat = np.array([c_pose.transform.rotation.x,
                c_pose.transform.rotation.y,
                c_pose.transform.rotation.z,
                c_pose.transform.rotation.w,])
    return c_xyz, c_quat

def pose_to_vec_quat_list(c_pose):
    """
    """
    c_xyz, c_quat = pose_to_vec_quat_pair(c_pose)
    return np.concatenate([c_xyz, c_quat])