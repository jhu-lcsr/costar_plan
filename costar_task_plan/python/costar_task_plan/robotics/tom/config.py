
TOM_RIGHT_CONFIG = {
    'name':'right',
    'robot_description_param': "robot_description",
    'obj_frame': "torso_link",
    'base_link': "torso_link",
    'end_link': "r_gripper_base_link",
    'joint_states_topic': "/joint_states",
    'gripper_topic': '',
    'ik_solver': 'kdl',
    'gmm_k': 1,
    'dmp_k': 100,
    'dmp_d': 20,
    'dmp_basis': 10,
    'dof': 6,
    'q0': None,
    'namespace': 'tom',
    'joints': ['r_shoulder_pan_joint',
      'r_shoulder_lift_joint',
      'r_elbow_joint',
      'r_wrist_1_joint',
      'r_wrist_2_joint',
      'r_wrist_3_joint']
    }

TOM_LEFT_CONFIG = {
    'name':'left',
    'robot_description_param': "robot_description",
    'obj_frame': "torso_link",
    'base_link': "torso_link",
    'end_link': "l_gripper_link",
    'joint_states_topic': "/joint_states",
    'gripper_topic': '',
    'ik_solver': 'kdl',
    'gmm_k': 1,
    'dmp_k': 100,
    'dmp_d': 20,
    'dmp_basis': 10,
    'dof': 6,
    'q0': None,
    'namespace': 'tom',
    'joints': ['l_shoulder_pan_joint',
      'l_shoulder_lift_joint',
      'l_elbow_joint',
      'l_wrist_1_joint',
      'l_wrist_2_joint',
      'l_wrist_3_joint']
    }

