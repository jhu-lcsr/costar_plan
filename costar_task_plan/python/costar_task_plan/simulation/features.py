from costar_task_plan.abstract.features import AbstractFeatures

import numpy as np


def GetAvailableFeatures():
    return ['empty', 'null', 'depth', 'rgb', 'joint_state', 'multi', 'grasp_segmentation']


def GetFeatures(features):
    '''
    Returns a particular task definition in the simulation.
    '''
    try:
        return {
            '': EmptyFeatures(),
            'null': EmptyFeatures(),
            'empty': EmptyFeatures(),
            'depth': DepthImageFeatures(),
            'joint_state': JointStateFeatures(),
            'rgb': RgbImageFeatures(),
            'multi': ImagePlusFeatures(),
            'grasp_segmentation': GraspSegmentationFeatures(),
        }[features]
    except KeyError, e:
        raise NotImplementedError(
            'Feature function not implemented: %s', str(e))


class EmptyFeatures(AbstractFeatures):

    '''
    This is a very simple set of features. It does, well, nothing at all. It is
    super fast, though, which makes it good for running execution tests.
    '''

    def compute(self, world, state):
        return np.array([0])

    def updateBounds(self, world):
        pass

    def getBounds(self):
        return np.array([0]), np.array([0])


class DepthImageFeatures(AbstractFeatures):

    '''
    The only features we return are the depths associated with each camera pixel.
    So we get 2.5D data here.
    '''

    def compute(self, world, state):
        return world.cameras[0].capture().depth

    def updateBounds(self, world):
        raise Exception('feature.updateBounds not yet implemented!')

    def getBounds(self):
        raise Exception('feature.getBounds not yet implemented!')


class JointStateFeatures(AbstractFeatures):

    def compute(self, world, state):
        return np.append(state.arm, state.gripper)

    def updateBounds(self, world):
        raise Exception('feature.updateBounds not yet implemented!')

    def getBounds(self):
        raise Exception('feature.getBounds not yet implemented!')


class RgbImageFeatures(AbstractFeatures):

    '''
    The only feature data we return will be a single RGB image from the first
    camera placed in the world, where ever that may be.
    '''

    def compute(self, world, state):
        return world.cameras[0].capture().rgb


class ImagePlusFeatures(AbstractFeatures):

    '''
    Include arm, state, and gripper features.
    '''

    def compute(self, world, state):
        img = world.cameras[0].capture().rgb
        return [img[:, :, :3], state.arm, state.gripper]

    @property
    def description(self):
        return ["features", "arm", "gripper"]


class GraspSegmentationFeatures(AbstractFeatures):

    '''
    This set of features includes data helpful for training segmentation.
    object_translation_rotation, state.arm, state.gripper, image_data, object_surface_points

    object_surface_points is where a ray cast from the camera to the object struck the first
    surface. If the ray never hit any surface

    This also represents all objects in the world as a single vector. This
    means that we need to have a constant size world, where we always have the
    same objects in the same order.
    '''

    def compute(self, world, state):
        import pybullet as pb
        object_translation_rotation = []
        image_data = world.cameras[0].capture()
        # 'camera_view_matrix' namedtuple index is 4
        # TODO(ahundt) ensure camera matrix translation component is from world origin to camera origin
        # camera ray is from the origin of the camera
        camera_ray_from = [image_data[4][0:2, 3]] * len(world.id_by_object.items())
        # camera_ray_to is the center of each object
        camera_ray_to = []
        for i, (name, oid) in enumerate(world.id_by_object.items()):
            obj = world.actors[oid][0:2, 3]
            object_translation_rotation += [obj.T.p, obj.T.M.GetQuaternion()]
            camera_ray_to += obj.T.p

        object_surface_points = []
        raylist = pb.rayTestBatch(camera_ray_from, camera_ray_to)
        for i, (uid, linkidx, hitfrac, hitpos, hitnormal) in range(raylist):
            if uid is -1:
                # if the object wasn't hit, use its origin
                object_surface_points += [world.id_by_object.items()[i].T.p]
            else:
                object_surface_points += hitpos
        return [object_translation_rotation, state.arm, state.gripper, image_data, object_surface_points]

    @property
    def description(self):
        return ["object_translation_rotation", "arm", "gripper", "camera"]

