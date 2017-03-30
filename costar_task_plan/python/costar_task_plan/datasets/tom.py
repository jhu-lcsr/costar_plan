# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See License for more details

from dataset import Dataset
import os, logging

LOGGER = logging.getLogger(__name__)

try:
  import rospy, rosbag
except ImportError, e:
  LOGGER.warning("Could not load rospy/rosbag for TOM dataset." \
                 "Original error: %s" % str(e))

'''
Load a set of ROS bags
'''
class TomDataset(Dataset):

  right_arm_end_frame_topic = "/tom/arm_right/Xef_w"
  gripper_topic = "/tom/arm_right/hand/gripperState"
  arm_data_topic = "/tom/arm_right/RobotData"
  vision_topic = "/tom/head/vision/manager/visual_info"
  skin_topic = "/tom/arm_right/hand/semanticGripperCells"

  FOLDER = 0
  GOOD = 1

  # set up the folders to read from
  folders = [
              ("Ilya_Box", True),
              ("Ilya_Trash", False),
              ("Karinne_Box", True),
              ("Karinne_Trash", False),
            ]

  def __init__(self):
    super(TomDataset, self).__init__("TOM")

    # hold bag file names
    self.trash_bags = []
    self.box_bags = []

    # hold demos
    self.box = []
    self.trash = []

    self.trash_poses = []
    self.box_poses = []
    self.move_poses = []
    self.test_poses = []
    self.pickup_poses = []

    self.trash_oranges = []
    self.box_oranges = []
    self.move_oranges = []
    self.test_oranges = []
    self.pickup_oranges = []

  def download(self, *args, **kwargs):
    raise RuntimeError('downloading this dataset is not yet supported')

  def load(self, root_filepath="", *args, **kwargs):
    self.trash_bags = []
    self.box_bags = []
    for folder, is_good in self.folders:
      data_folder = os.path.join(root_filepath, folder)
      print 'Loading from folder "%s"...'%data_folder
      files = os.listdir(folder)
      for filename in files:
        terms = filename.split('.')
        if terms[-1] == "bag":
          # we found one of the rosbags
          # save it as part of the dataset
          whole_filename = os.path.join(data_folder, filename)
          print '\tLoading from file "%s"...'%whole_filename
          if is_good:
            self.box_bags.append(whole_filename)
          else:
            self.trash_bags.append(whole_filename)

    print "Extracting data..."
    print "\tExtracting data for box actions..."
    trajs, poses, oranges = self._extract_samples(self.box_bags)
    self.pickup_trajs = trajs[0]
    self.move_trajs = trajs[1]
    self.test_trajs = trajs[2]
    self.box = trajs[3]

    self.pickup_poses = poses[0]
    self.move_poses = poses[1]
    self.test_poses = poses[2]
    self.box_poses = poses[3]

    self.pickup_oranges += oranges[0]
    self.move_oranges += oranges[1]
    self.test_oranges += oranges[2]
    self.trash_oranges = oranges[3]

    print "\tExtracting data for trash actions..."
    trajs, poses, oranges = self._extract_samples(self.trash_bags)

    self.pickup_trajs += trajs[0]
    self.move_trajs += trajs[1]
    self.test_trajs += trajs[2]
    self.trash = trajs[3]

    self.pickup_oranges += oranges[0]
    self.move_oranges += oranges[1]
    self.test_oranges += oranges[2]
    self.trash_oranges = oranges[3]


  def _extract_samples(self, bag_files):
    topics = [
            self.right_arm_end_frame_topic,
            self.gripper_topic,
            self.arm_data_topic,
            self.vision_topic,
            self.skin_topic,
            ]
    trajs = []
    for filename in bag_files:
      traj = []
      bag = rosbag.Bag(filename)
      # extract the end pose of the robot arm
      pose = None
      gripper = None
      data = None
      orange = None
      for topic, msg, t in bag.read_messages(topics):
        sec = t.to_sec()
        if topic == self.gripper_topic:
            gripper = msg
        elif topic == self.arm_data_topic:
            data = msg
        elif topic == self.right_arm_end_frame_topic:
            pose = msg
        elif topic == self.vision_topic:
            orange = msg.objData[0]

        if all([pose, gripper, data]):
            gripper_open = gripper.state == 'open'
            gripper_state = gripper.stateId
            traj.append((sec, pose, data, gripper_open, gripper_state, orange))
            gripper, data, pose, orange = None, None, None, None
      trajs.append(traj)
      if len(traj) > 0:
        break

    # =========================================================================
    # Split into pickup, place, and drop trajectories
    # We know what is done after each step
    pickup_trajs = []
    move_trajs = []
    test_trajs = []
    drop_trajs = []
    lift_trajs = []
    pickup_poses = []
    move_poses = []
    test_poses = []
    drop_poses = []
    pickup_oranges = []
    move_oranges = []
    test_oranges = []
    drop_oranges = []
    for traj in trajs:
        was_open = False
        print "was actually open:", traj[0][3]
        last_stage = 0
        stage = 0
        cropped_traj = []
        cropped_orange = []
        done = False

        for t, pose, data, gopen, gstate, orange in traj:
            if was_open and not gopen:
                # pose is where we picked up an object
                if stage < 1:
                    #print "picked up an object"
                    pickup_poses.append(pose)
                    stage = 1
                elif stage < 3:
                    stage = 3
                elif stage == 3:
                    #print "done test; pickup obj"
                    test_poses.append(pose)
                    stage = 4
                else: print "other?"
            elif not was_open and gopen and stage == 1:
                #print "dropped to test"
                move_poses.append(pose)
                stage = 2
            elif not was_open and gopen and stage == 4:
                #print "dropped an object"
                drop_poses.append(pose)
                stage = 5
                done = True
            was_open = gopen

            if not last_stage == stage:
                if last_stage == 0:
                    pickup_trajs.append(cropped_traj)
                    pickup_oranges.append(cropped_orange)
                elif last_stage == 1:
                    move_trajs.append(cropped_traj)
                    move_oranges.append(cropped_orange)
                elif last_stage in [2, 3]:
                    test_trajs.append(cropped_traj)
                    test_oranges.append(cropped_orange)
                elif last_stage == 4:
                    drop_trajs.append(cropped_traj)
                    drop_oranges.append(cropped_orange)
                cropped_traj = []
            last_stage = stage
            cropped_traj.append((t, pose, data, gopen, gstate))
            cropped_orange.append(orange)

            if done: break

    return [pickup_trajs, move_trajs, test_trajs, drop_trajs], \
           [pickup_poses, move_poses, test_poses, drop_poses], \
           [pickup_oranges, move_oranges, test_oranges, drop_oranges]
