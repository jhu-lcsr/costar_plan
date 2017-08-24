import numpy as np
import pybullet as pb

from collections import namedtuple

# Don't change ImageData lightly,
# consider only adding to it because
# some code such as in features.py
# depends on the order of
# this tuple's elements.
ImageData = namedtuple(
    'ImageData', ['name', 'rgb', 'depth', 'mask', 'camera_view_matrix', 'camera_projection_matrix'], verbose=False)


class Camera(object):

    '''
    Wrapper for a PyBullet camera.

    Params:
    -------
    pos: position of the camera
    target: where the camera is looking
    up: camera "up" vector (defaults to z axis)
    image_height: height of image to capture
    image_width: width of image to capture
    '''

    def __init__(self, name, target,
                 distance,
                 roll, pitch, yaw,
                 up_idx=2,
                 image_width=1024 / 8,
                 image_height=768 / 8,
                 fov=45,
                 near_plane=0.1,
                 far_plane=10):
        '''
        Create camera matrix for a particular position in the simulation. Task
        definitions should produce these and
        '''
        self.name = name
        self.matrix = np.array(pb.computeViewMatrixFromYawPitchRoll(
            target, distance, yaw=yaw, pitch=pitch, roll=roll, upAxisIndex=up_idx))
        self.image_height = image_height
        self.image_width = image_width
        self.aspect_ratio = self.image_width / self.image_height
        self.fov = fov
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.projection_matrix = np.array(pb.computeProjectionMatrixFOV(
            self.fov,
            self.aspect_ratio,
            self.near_plane,
            self.far_plane))

    def capture(self):
        _, _, rgb, depth, mask = pb.getCameraImage(
            self.image_width, self.image_height,
            viewMatrix=self.matrix,
            projectionMatrix=self.projection_matrix)
        # TODO(ahundt) remove division in rgb / 255.
        # TODO(ahundt)
        return ImageData(self.name, rgb / 255., depth, mask, self.matrix, self.projection_matrix)
