# Copyright Philipp Jund, Andrew Hundt, Kieran Wynn 2017. All Rights Reserved.
#
# https://github.com/PhilJd/tf-quaternion
# https://github.com/KieranWynn/pyquaternion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

This small library implements quaternion operations with tensorflow.
All operations are derivable.

"""
import numpy as np
import tensorflow as tf
import math


def scope_wrapper(func, *args, **kwargs):
    def scoped_func(*args, **kwargs):
        with tf.name_scope("quat_{}".format(func.__name__)):
            return func(*args, **kwargs)
    return scoped_func


@scope_wrapper
def point_to_quaternion():
    raise NotImplementedError()


@scope_wrapper
def from_rotation_matrix():
    raise NotImplementedError()


@scope_wrapper
def multiply(a, b):
    if not isinstance(a, Quaternion) and not isinstance(b, Quaternion):
        msg = "Multiplication is currently only implemented " \
              "for quaternion * quaternion"
        raise NotImplementedError(msg)
    w1, x1, y1, z1 = tf.unstack(a.value())
    w2, x2, y2, z2 = tf.unstack(b.value())
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return Quaternion(tf.stack((w, x, y, z)))


@scope_wrapper
def divide(a, b):
    if not isinstance(a, Quaternion) and not isinstance(b, Quaternion):
        msg = "Division is currently only implemented " \
              "for quaternion \ quaternion"
        raise NotImplementedError(msg)
    w1, x1, y1, z1 = tf.unstack(a.value())
    w2, x2, y2, z2 = tf.unstack(b.value())
    bnorm = b._norm()
    w = (w1*w2 + x1*x2 + y1*y2 + z1*z2) / bnorm,
    x = (-w1*x2 + x1*w2 - y1*z2 + z1*y2) / bnorm,
    y = (-w1*y2 + x1*z2 + y1*w2 - z1*x2) / bnorm,
    z = (-w1*z2 - x1*y2 + y1*x2 + z1*w2) / bnorm
    return Quaternion(tf.stack((w, x, y, z)))


class Quaternion():

    def __init__(self, initial_wxyz=(1.0, 0.0, 0.0, 0.0), dtype=tf.float32):
        """
        Args:
            initial_wxyz: The values for w, x, y, z. Must have shape=[4].
                - `tf.Tensor` or `tf.Variable` of type float16/float32/float64
                - list/tuple/np.array
                - Quaternion
                Defaults to (1.0, 0.0, 0.0, 0.0)
            dtype: The type to create the value tensor. Only considered if
                initial_wxyz is passed as list, tuple or np.array. Otherwise
                the type of the `Tensor` is used. This is to prevent silent
                changes to the gradient type resulting in a different precision
                Allowed types are float16, float32, float64.

        Returns:
            A Quaternion.

        Raises:
            ValueError, if the shape of initial_wxyz is not [4].
            TypeError, either if the `Tensor` initial_wxyz's type is not float
                or if initial_wxyz is not a Tensor/list/tuple etc.
        """
        if isinstance(initial_wxyz, (tf.Tensor, tf.Variable)):
            self._validate_shape(initial_wxyz)
            self._validate_type(initial_wxyz)
            self._q = (initial_wxyz if initial_wxyz.dtype == dtype
                       else tf.cast(initial_wxyz, dtype))
        elif isinstance(initial_wxyz, (np.ndarray, list, tuple)):
            self._validate_shape(initial_wxyz)
            self._q = tf.constant(initial_wxyz, dtype=dtype)
        elif isinstance(initial_wxyz, Quaternion):
            tensor = initial_wxyz.value()
            self._q = (tensor if tensor.dtype == dtype
                       else tf.cast(tensor, dtype))
        else:
            raise TypeError("Can not convert object of type {} to Quaternion"
                            "".format(type(initial_wxyz)))

    def value(self):
        """ Returns a `Tensor` which holds the value of the quaternion. Note
            that this does not return a reference, so you can not alter the
            quaternion through this.
        """
        return self._q

    @ staticmethod
    def _quaternions_to_tensors(quats):
        return [q.value() if isinstance(q, Quaternion) else q for q in quats]

    def __add__(a, b):
        val_a, val_b = Quaternion._quaternions_to_tensors((a, b))
        return Quaternion(val_a + val_b)

    def __sub__(a, b):
        val_a, val_b = Quaternion._quaternions_to_tensors((a, b))
        return Quaternion(val_a - val_b)

    def __mul__(a, b):
        return multiply(a, b)

    def __imul__(self, other):
        if isinstance(other, Quaternion):
            return multiply(self, other)
        #elif isinstance(other, tf.Variable) or isinstance(other, tf.Tensor):
        #    self._validate_shape(other)
        #    return multiply(self, Quaternion(other))
        else:
            msg = "Quaternion Multiplication not implemented for this type."
            raise NotImplementedError(msg)

    def __div__(a, b):
        return divide(a, b)

    def __idiv__(self, other):
        if isinstance(other, Quaternion):
            return divide(self, other)
        else:
            msg = "Quaternion Multiplication not implemented for this type."
            raise NotImplementedError(msg)

    def __repr__(self):
        return "<tfq.Quaternion ({})>".format(self._q.__repr__()[1:-1])

    @scope_wrapper
    def coeffs(self):
        return self._q

    @scope_wrapper
    def inverse(self):
        w, x, y, z = tf.unpack(tf.divide(self._q, self._norm()))
        return Quaternion((w, -x, -y, -z))

    @scope_wrapper
    def normalized(self):
        return Quaternion(tf.divide(self._q, self._abs()))

    @scope_wrapper
    def as_rotation_matrix(self):
        """ Calculates the rotation matrix. See
        [http://www.euclideanspace.com/maths/geometry/rotations/
         conversions/quaternionToMatrix/]

        Returns:
            A 3x3 `Tensor`, the rotation matrix

        """
        # helper functions
        def diag(a, b):  # computes the diagonal entries,  1 - 2*a**2 - 2*b**2
            return 1 - 2 * tf.pow(a, 2) - 2 * tf.pow(b, 2)

        def tr_add(a, b, c, d):  # computes triangle entries with addition
            return 2 * a * b + 2 * c * d

        def tr_sub(a, b, c, d):  # computes triangle entries with subtraction
            return 2 * a * b - 2 * c * d

        w, x, y, z = tf.unstack(self.normalized().value())
        return [[diag(y, z), tr_sub(x, y, z, w), tr_add(x, z, y, w)],
                [tr_add(x, y, z, w), diag(x, z), tr_sub(y, z, x, w)],
                [tr_sub(x, z, y, w), tr_add(y, z, x, w), diag(x, y)]]

    @scope_wrapper
    @staticmethod
    def random():
        """Generate a random unit quaternion.
        Uniformly distributed across the rotation space
        As per: http://planning.cs.uiuc.edu/node198.html
        https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
        """
        r1, r2, r3 = tf.random.random(3)

        q1 = tf.sqrt(1.0 - r1) * (tf.sin(2 * math.pi * r2))
        q2 = tf.sqrt(1.0 - r1) * (tf.cos(2 * math.pi * r2))
        q3 = tf.sqrt(r1)       * (tf.sin(2 * math.pi * r3))
        q4 = tf.sqrt(r1)       * (tf.cos(2 * math.pi * r3))

        return Quaternion((q1, q2, q3, q4))

    @staticmethod
    def _validate_shape(x):
        msg = "Can't create a quaternion with shape [4] from {} with shape {}."
        if isinstance(x, (list, tuple)) and np.array(x).shape != (4,):
                raise ValueError(msg.format("list/tuple", np.array(x).shape))
        elif isinstance(x, np.ndarray) and x.shape != (4,):
                raise ValueError(msg.format("np.array", x.shape))
        elif (isinstance(x, (tf.Tensor, tf.Variable))
              and x.shape.as_list() != [4]):
                raise ValueError(msg.format("tf.Tensor", x.shape.as_list()))

    @staticmethod
    def _validate_type(x):
        if not x.dtype.is_floating:
            raise TypeError("Quaternion only supports floating point numbers")

    @scope_wrapper
    def _norm(self):
        return tf.reduce_sum(tf.square(self._q))

    @scope_wrapper
    def _abs(self):
        return tf.sqrt(tf.reduce_sum(tf.square(self._q)))
