
import pybullet as pb

from collections import namedtuple

ImageData = namedtuple(
    'ImageData', ['name', 'rgb', 'depth', 'mask'], verbose=False)


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
                 image_width=1024 / 4,
                 image_height=768 / 4,
                 fov=45,
                 near_plane=0.1,
                 far_plane=1000):
        '''
        Create camera matrix for a particular position in the simulation. Task
        definitions should produce these and
        '''
        self.name = name
        self.matrix = pb.computeViewMatrixFromYawPitchRoll(
            target, distance, yaw, pitch, roll, up_idx)
        self.image_height = image_height
        self.image_width = image_width
        self.aspect_ratio = self.image_width / self.image_height
        self.fov = fov
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.projection_matrix = pb.computeProjectionMatrixFOV(
            self.fov,
            self.aspect_ratio,
            self.near_plane,
            self.far_plane)

    def capture(self):
        _, _, rgb, depth, mask = pb.getCameraImage(
            self.image_width, self.image_height, viewMatrix=self.matrix, projectionMatrix=self.projection_matrix)
        return ImageData(self.name, rgb, depth, mask)
