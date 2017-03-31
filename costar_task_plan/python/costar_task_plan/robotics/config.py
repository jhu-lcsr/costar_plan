
DEFAULT_MODEL_CONFIG = {
    "clusters": 10,
    "sigma": 1e-8,
    "dtw": True,
}

DEFAULT_ROBOT_CONFIG = {
    'name': 'robot',
    'robot_description_param': "robot_description",
    'ik_solver': 'kdl',
    'base_link': "base_link",
    'end_link': "ee_link",
    'joint_states_topic': "/joint_states",
    'dof': 6,
    'q0': None,
    'joints': None,
    'namespace': 'costar',
}
