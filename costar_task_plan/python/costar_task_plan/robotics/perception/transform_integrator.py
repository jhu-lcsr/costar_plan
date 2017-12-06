from collections import deque

# Import a ton of ROS messages

import numpy as np
import PyKDL as kdl
import rospy
import tf
import tf_conversions.posemath as pm

class TransformIntegator(object):

    def __init__(self, name, root,
            transforms={},
            listener=None,
            broadcaster=None,
            history_length=0,
            offset=None):
        self.name = name
        self.transforms = transforms
        self.root = root
        self.history = deque()
        self.history_length = history_length
        self.offset = offset


        if listener is not None:
            self.listener = listener
        else:
            self.listener = tf.TransformListener()
        if broadcaster is not None:
            self.broadcaster = broadcaster
        else:
            self.broadcaster = tf.TransformBroadcaster()

    def addTransform(self, name, pose):
        self.transforms[name] = pose

    def tick(self):
        '''
        Look for all transforms in the list
        '''

        if not self.listener.frameExists(self.root):
            return

        count = 0
        avg_p = np.zeros(3)
        avg_q = np.zeros(4)
        for name, pose in self.transforms.items():
            if self.listener.frameExists(name):
                t = self.listener.getLatestCommonTime(self.root, name)
                p, q = self.listener.lookupTransform(self.root, name, t)
                F = pm.fromTf((p, q))
                F = F * pose
                p, q = pm.toTf(F)
                avg_p += np.array(p)
                avg_q += np.array(q)
                count += 1
        if count > 0:
            avg_p /= count
            avg_q /= count
            avg_q /= np.linalg.norm(avg_q)

            if self.history_length > 0:
                # use history to smooth predictions
                if len(self.history) >= self.history_length:
                    self.history.pop()
                self.history.appendleft((avg_p, avg_q))
                avg_p = np.zeros(3)
                avg_q = np.zeros(4)
                for p, q in self.history:
                    avg_p += p
                    avg_q += q

                avg_p /= len(self.history)
                avg_q /= len(self.history)
                avg_q /= np.linalg.norm(avg_q)

            if self.offset is not None:
                # apply some offset after we act
                F = pm.fromTf((avg_p, avg_q))
                F = F * self.offset
                avg_p, avg_q = pm.toTf(F)

            self.broadcaster.sendTransform(avg_p, avg_q, rospy.Time.now(), self.name, self.root)

