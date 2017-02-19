#include <ros/ros.h>
#include <grid/costar_planner.h>

using namespace grid;

int main(int argc, char **argv) {
  ros::init(argc,argv,"costar_plan_node");
  GridPlanner gp("robot_description","/joint_states","/planning_scene");
#if 0
  gp.SetDof(7);
  gp.SetNumBasisFunctions(5);
  gp.SetK(100);
  gp.SetD(20);
  gp.SetTau(1.0);
  gp.SetGoalThreshold(0.1);
  gp.SetVerbose(true);
#endif

#if 0
  ros::Duration(0.5).sleep();

  std::vector<double> in;

  double primitives[] = {
    // goal state
    1.4051760093677244, -1.1104719560033205, -1.9033419999549013, 1.8269932570905123, 1.2499923807229427, 0.08964526401203354, -0.8798027314692156,
    // parameters for the dmp
    0.1,0.1,0.1,0.1,0.1,
    0.1,0.1,0.1,0.1,0.1,
    0.1,0.1,0.1,0.1,0.1,
    0.1,0.1,0.1,0.1,0.1,
    0.1,0.1,0.1,0.1,0.1,
    0.1,0.1,0.1,0.1,0.1,
    0.1,0.1,0.1,0.1,0.1
  };

  in = std::vector<double>(&primitives[0],&primitives[7+35+1]);

  ros::Rate rate(100);
  while (ros::ok()) {
    ros::spinOnce();

    for (int i = 0; i < 7+35; ++i) {
      in[i] = 1000*rand();
    }
    Traj_t res = gp.TryPrimitives(in);

    rate.sleep();
  }
#endif

  return 0;
}
