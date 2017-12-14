from __future__ import print_function
from collections import named_tuple

TaskInfo = named_tuple('header','transition')
Transition = named_tuple('from','to','frequency','parameters')
TransitionParameters = named_tuple('start_time','end_time','from_parameters','to_parameters')
WrappedString = named_tuple('data')
Header = named_tuple('seq','stamp','frame_id')
