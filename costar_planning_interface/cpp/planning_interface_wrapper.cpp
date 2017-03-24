
#include <costar_task_plan/costar_planner.h>
#include <costar_task_plan/planning_interface_wrapper.h>
#include <costar_task_plan/utils/python.hpp>

namespace costar {


  // Default constructor: create a planner object.
  PlanningInterfaceWrapper::PlanningInterfaceWrapper(const std::string &robotDescription,
                                                     const std::string &jointStateTopic,
                                                     const std::string &planningSceneTopic,
                                                     double padding,
                                                     unsigned int num_basis,
                                                     bool verbose) : planner(
                                                         new CostarPlanner(robotDescription,
                                                                           jointStateTopic,
                                                                           planningSceneTopic,
                                                                           padding,
                                                                           num_basis,
                                                                           verbose)) {}

  // Helper function: get current joint positions (according to the joint
  // listener, not to the planning scene).
  boost::python::list PlanningInterfaceWrapper::GetJointPositions() const {
    boost::python::list res;
    for (double x: planner->x0) {
      res.append<double>(x);
    }
    return res;
  }

  // Try a set of motion primitives; see if they work.
  // This is aimed at the python version of the code. It takes a python list
  // containing the parameters of the motion as its only argument.
  boost::python::list PlanningInterfaceWrapper::pyTryPrimitives(const boost::python::list &list) {
    std::vector<double> primitives = to_std_vector<double>(list);

    planner->scene->getCurrentStateNonConst().update(); 

    if (planner->verbose) {
      std::vector<std::string> names = planner->scene->getWorld()->getObjectIds();

      std::cout << "==========================" << std::endl;
      std::cout << "OBJECTS IN WORLD: " << std::endl;
      for (const std::string &name: names) {
        std::cout << " -- " << name << std::endl;
      }
      std::cout << "==========================" << std::endl;
    }

    Traj_t traj = planner->TryPrimitives(primitives);

    boost::python::list res;

    for (const Traj_pt_t &pt: traj.points) {
      boost::python::list p;
      boost::python::list v;

      for (unsigned int i = 0; i < pt.positions.size(); ++i) {
        p.append<double>(pt.positions[i]);
        v.append<double>(pt.velocities[i]);
      }

      res.append<boost::python::tuple>(boost::python::make_tuple(p,v));
    }

    return res;
  }

  /* try a single trajectory and see if it works.
   * this is aimed at the python version of the code. */
  bool PlanningInterfaceWrapper::pyTryTrajectory(const boost::python::list &trajectory) {
    std::vector< std::vector<double> > traj;
    std::vector< boost::python::list > tmp = to_std_vector<boost::python::list>(trajectory);

    for (boost::python::list &pt: tmp) {
      traj.push_back(to_std_vector<double>(pt));
    }

    return planner->TryTrajectory(traj);
  }

  void PlanningInterfaceWrapper::SetVerbose(bool verbose) {
    planner->SetVerbose(verbose);
  }

  void PlanningInterfaceWrapper::SetK(double k) {
    planner->SetK(k);
  } 

  void PlanningInterfaceWrapper::SetD(double d) {
    planner->SetD(d);
  }

  void PlanningInterfaceWrapper::SetTau(double tau) {
    planner->SetTau(tau);
  }

  void PlanningInterfaceWrapper::SetGoalThreshold(double threshold) {
    planner->SetGoalThreshold(threshold);
  }

  void PlanningInterfaceWrapper::SetDof(unsigned int dof) {
    planner->SetDof(dof);
  } 

  void PlanningInterfaceWrapper::SetNumBasisFunctions(unsigned int num_basis) {
    planner->SetNumBasisFunctions(num_basis);
  }

  // Are we allowed to collide?
  void PlanningInterfaceWrapper::SetCollisions(const std::string obj, bool allowed) {
    planner->SetCollisions(obj, allowed);
  } 

  // Robot object default entry
  void PlanningInterfaceWrapper::SetDefaultCollisions(const std::string link, bool ignore) {
    planner->SetDefaultCollisions(link, ignore);
  }

  void PlanningInterfaceWrapper::ResetCollisionMap() {
    planner->ResetCollisionMap();
  }

  void PlanningInterfaceWrapper::PrintInfo() const {
    planner->PrintInfo();
  }

}

#ifdef GEN_PYTHON_BINDINGS
BOOST_PYTHON_MODULE(pycostar_planner) {
  class_<costar::PlanningInterfaceWrapper>("PlanningInterfaceWrapper",init<std::string,std::string,std::string,double,unsigned int,bool>())
    .def("TryPrimitives", &costar::PlanningInterfaceWrapper::pyTryPrimitives)
    .def("TryTrajectory", &costar::PlanningInterfaceWrapper::pyTryTrajectory)
    .def("SetK", &costar::PlanningInterfaceWrapper::SetK)
    .def("SetD", &costar::PlanningInterfaceWrapper::SetD)
    .def("SetTau", &costar::PlanningInterfaceWrapper::SetTau)
    .def("SetDof", &costar::PlanningInterfaceWrapper::SetDof)
    .def("SetNumBasisFunctions", &costar::PlanningInterfaceWrapper::SetNumBasisFunctions)
    .def("SetGoalThreshold", &costar::PlanningInterfaceWrapper::SetGoalThreshold)
    .def("SetVerbose", &costar::PlanningInterfaceWrapper::SetVerbose)
    .def("PrintInfo", &costar::PlanningInterfaceWrapper::PrintInfo)
    .def("GetJointPositions", &costar::PlanningInterfaceWrapper::GetJointPositions)
    .def("SetCollisions", &costar::PlanningInterfaceWrapper::SetCollisions)
    .def("SetDefaultCollisions", &costar::PlanningInterfaceWrapper::SetDefaultCollisions)
    .def("ResetCollisionMap", &costar::PlanningInterfaceWrapper::ResetCollisionMap);
}
#endif

