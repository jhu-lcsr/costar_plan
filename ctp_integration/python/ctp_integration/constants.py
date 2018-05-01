try:
    import rospy
except ImportError as e:
    print("constants.py could not import rospy")
    rospy = None

def GetColors():
    colors = ["red", "blue", "yellow", "green"]
    return colors

def GetHomeJointSpace():
    """ Default home joint angle pose for the ur5 robot
    """
    Q_HOME = [-0.202, -0.980, -1.800, -0.278, 1.460, 1.613]
    # Q_HOME was formerly: 
    # Q_HOME = [0.3, -1.33, -1.8, -0.27, 1.5, 1.6]
    if rospy is not None:
        try:
            Q_HOME = rospy.get_param('/costar/robot/home')
        except KeyError as e:
            rospy.logwarn("CoSTAR home position not set, using default:" + str(Q_HOME))
    else:
        print("Warning: using hardcoded home joint angles because rospy is not available.")

    # [-0.202, -0.980, -1.800, -0.278, 1.460, 1.613]
    return Q_HOME

def GetHomePose():
    """ Home Pose as a vector and quaternion
      data order: x y z qx qy qz qw

        Warning: this is currently fixed to a constant value
        and will be incorrect if the rosparam /costar/robot/home changes.
    """
    # TODO(ahundt) generate this value based on GetHomeJointSpace.
    vec_quat_home = [0.1762, -0.1558, 0.682, -0.7112, 0.1410, 0.0788, -0.6843]
    return vec_quat_home
