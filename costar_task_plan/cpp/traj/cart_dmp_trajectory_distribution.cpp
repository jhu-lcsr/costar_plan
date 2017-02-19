#include <grid/dmp_trajectory_distribution.h>

#include <Eigen/Dense>
#include <trajectory_msgs/JointTrajectoryPoint.h>

using trajectory_msgs::JointTrajectoryPoint;
using namespace Eigen;

#define SHOW_SAMPLED_VALUES 0
#define DEFAULT_SIGMA 0.1

namespace grid {

  /**
   * Initialize a trajectory distribution with given params
   */
  CartDmpTrajectoryDistribution::DmpTrajectoryDistribution(unsigned int dim_, unsigned int nbasis_, RobotKinematicsPointer robot_)
    : dim(dim_),
    nbasis(nbasis_),
    robot(robot_),
    dist((dim_*nbasis_) + POSE_FEATURES_SIZE,1),
    verbose(false),
    dmp_list(dim_),
    dmp_goal(dim_),
    k_gain(100),
    d_gain(20),
    tau(1.0),
    goal_threshold(dim_,0.01),
    dmp_velocity_multiplier(0.1)
  {

    assert(dim == robot->getDegreesOfFreedom());

    // set up anything else?
    nvars = dist.ns[0].mu.size();

    for (unsigned int i = 0; i < dim; ++i) {
      dmp_list[i].k_gain = k_gain;
      dmp_list[i].d_gain = d_gain;
      dmp_list[i].weights.resize(nbasis);
    }
  }

  /**
   * initialize
   * set up the distribution based on a skill and an environment
   */
  void CartDmpTrajectoryDistribution::initialize(TestFeatures &features, const Skill &skill, std::vector<double> sigma) {
    //Pose p0 = features.lookup(AGENT);
    Pose p1 = features.lookup(skill.getInitializationFeature()) * skill.getInitializationFinalPose();

    unsigned int idx = 0;

    dist.ns[0].mu[POSE_FEATURE_X] = p1.p.x();
    dist.ns[0].mu[POSE_FEATURE_Y] = p1.p.y();
    dist.ns[0].mu[POSE_FEATURE_Z] = p1.p.z();

    double roll, pitch, yaw;
    p1.M.GetRPY(roll, pitch, yaw);

    dist.ns[0].mu[POSE_FEATURE_YAW] = yaw;
    dist.ns[0].mu[POSE_FEATURE_PITCH] = pitch;
    dist.ns[0].mu[POSE_FEATURE_ROLL] = roll;

    for (int j = POSE_RPY_SIZE; j < nvars; ++j) {
      dist.ns[0].mu[j] = 0;
    }

    /*for (unsigned int j = 0; j < dist.ns[0].mu.size(); ++j) {
      std::cout<<dist.ns[0].mu(j)<<"\n";
      }*/

    if (sigma.size() < nvars) {
      if (verbose) {
        std::cerr << "[GRID/TRAJECTORY DISTRIBUTION] Noise argument for trajectory search initialization was the wrong size!" << std::endl;
        std::cerr << "[GRID/TRAJECTORY DISTRIBUTION] Should be: " << dim << std::endl;
        std::cerr << "[GRID/TRAJECTORY DISTRIBUTION] Was: " << sigma.size() << std::endl;
      }
      for (int j = 0; j < nvars; ++j) {
        dist.ns[0].P(j,j) = DEFAULT_SIGMA;
      }

    } else {
      for (int j = 0; j < nvars; ++j) {
        dist.ns[0].P(j,j) = sigma[j];
      }
    }
    dist.Update();


  }

  /**
   * sample
   * Pull a random trajectory from the gmm
   * Convert it into a KDL trajectory
   * NON-CONST becuse Gmm::sample is likewise non-const
   */
  void CartDmpTrajectoryDistribution::sample(std::vector<EigenVectornd> &params,std::vector<JointTrajectory> &trajs) {

    using KDL::Vector;
    using KDL::Rotation;

    unsigned int nsamples = params.size();
    trajs.resize(nsamples);

    for (int sample = 0; sample < nsamples; ++sample) {

      EigenVectornd vec;
      vec.resize(nvars);
      dist.Sample(vec);

      params[sample] = vec;

#if SHOW_SAMPLED_VALUES
      std::cout << "Sampled: ";
      for (int j = 0; j < dim; ++j) {
        std::cout << vec[j] << " ";
      }
      std::cout << std::endl;
#endif

      // convert the first six into a pose
      Vector v1 = Vector(vec[POSE_FEATURE_X],vec[POSE_FEATURE_Y],vec[POSE_FEATURE_Z]);
      Rotation r1 = Rotation::RPY(vec[POSE_FEATURE_ROLL],vec[POSE_FEATURE_PITCH],vec[POSE_FEATURE_YAW]);
      Pose p(r1,v1);

      robot->IkPos(p,q);
      unsigned int idx = 0;
      //std::cout << "GOAL = ";
      for (; idx < dim; ++idx) {
        dmp_goal[idx] = q(idx);
        //std::cout << q(idx) << " ";
      }
      //std::cout << std::endl;

      for (unsigned int i = 0; i < dim; ++i) {
        for (unsigned int j = 0; j < nbasis; ++j) {
          dmp_list[i].weights[j] = vec[idx++];
        }
      }

      unsigned char at_goal;
      dmp::DMPTraj plan;



      dmp::generatePlan(dmp_list,robot->getJointPos(),robot->getJointVel(),0,dmp_goal,goal_threshold,-1,tau,0.1,5,plan,at_goal);

      if (verbose) {
        std::cout << "--------------------------" << std::endl;

        std::cout << "Using joints = ";
        for (const double &q: robot->getJointPos()) {
          std::cout << q << " ";
        }
        std::cout << std::endl;        std::cout << "at goal: " << (unsigned int)at_goal << std::endl;
        std::cout << "points: " << plan.points.size() << std::endl;
      }


      trajs[sample].points.resize(plan.points.size());
      for (unsigned int j = 0; j < plan.points.size(); ++j) {
        trajs[sample].points[j].positions = plan.points[j].positions;
        trajs[sample].points[j].velocities = plan.points[j].velocities;
        for (double &v: trajs[sample].points[j].velocities) {
          v *= dmp_velocity_multiplier;
        }
      }
    }
  }



  /* ==================================================================== */
  /* ==================================================================== */
  /*             BELOW THIS: SAME AS TRAJECTORY DISTRIBUTION              */
  /* ==================================================================== */
  /* ==================================================================== */


  /**
   * update
   * take a set of trajectories and samples
   * use the trajectories to reweight the distribution
   */
  void CartDmpTrajectoryDistribution::update(
      std::vector<EigenVectornd> &params,
      std::vector<double> &ps,
      double diagonal_noise)
  {
    update(params,ps,diagonal_noise,def_step_size);
  }

  /**
   * update
   * take a set of trajectories and samples
   * use the trajectories to reweight the distribution
   */
  void CartDmpTrajectoryDistribution::update(
      std::vector<EigenVectornd> &params,
      std::vector<double> &ps)
  {
    update(params,ps,diagonal_sigma,def_step_size);
  }


  /**
   * update
   * take a set of trajectories and samples
   * use the trajectories to reweight the distribution
   */
  void CartDmpTrajectoryDistribution::update(
      std::vector<EigenVectornd> &params,
      std::vector<double> &ps,
      double diagonal_noise,
      double step_size)
  {

    double psum = 0;
    for (double &d: ps) {
      psum += d;
    }

    if (dist.k == 1) {

      // one cluster only
      // compute mean

      dist.ns[0].mu *= (1 - step_size); //setZero();
      dist.ns[0].P *= (1 - step_size); //setZero();

      for (unsigned int i = 0; i < params.size(); ++i) {
        //std::cout << "mu rows = " << dist.ns[0].mu.rows() << ", vec rows = " << vec.rows() << std::endl;
        //std::cout << "mu cols = " << dist.ns[0].mu.cols() << ", vec cols = " << vec.cols() << std::endl;
        double wt = step_size * ps[i] / psum;
        dist.ns[0].mu += params[i] * wt;
      }

      for (unsigned int i = 0; i < params.size(); ++i) {
        double wt = step_size * ps[i] / psum;
        //std::cout << wt << ", " << ps[i] << ", " << psum << std::endl;
        dist.ns[0].P += wt * (params[i] - dist.ns[0].mu) * (params[i] - dist.ns[0].mu).transpose();
      }


    } else {

      // set up weighted data
      // and then fit GMM again

      std::vector<std::pair<EigenVectornd,double> > data(ps.size());
      for (unsigned int i = 0; i < ps.size(); ++i) {
        data[0].first = params[i];
        data[0].second = ps[i] / psum;
      }

      dist.Fit(data);

    }

    for (unsigned int i = 0; i < dist.k; ++i) {
      dist.ns[0].P += diagonal_noise * Matrix<double,Dynamic,Dynamic>::Identity(nvars,nvars);
    }

    dist.Update();
  }
};
