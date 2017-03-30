
TOM_RIGHT_CONFIG = {
    'name':'right',
    'robot_description_param': "robot_description",
    'ee_link': "r_ee_link",
    'base_link': "torso_link",
    'joint_states_topic': "/joint_states",
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
    'ee_link': "l_ee_link",
    'base_link': "torso_link",
    'joint_states_topic': "/joint_states",
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

