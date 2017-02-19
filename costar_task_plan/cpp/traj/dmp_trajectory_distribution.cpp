#include <grid/dmp_trajectory_distribution.h>

#include <Eigen/Dense>
#include <trajectory_msgs/JointTrajectoryPoint.h>

using namespace Eigen;

#define SHOW_SAMPLED_VALUES 0
#define DEFAULT_SIGMA 0.005
//#define DEFAULT_SIGMA 0.0000000001

namespace grid {

  /**
   * Initialize a trajectory distribution with given params
   */
  DmpTrajectoryDistribution::DmpTrajectoryDistribution(unsigned int dim_, unsigned int nbasis_, RobotKinematicsPtr robot_)
    : dim(dim_),
    checker(0),
    nbasis(nbasis_),
    robot(robot_),
    dist((dim_*nbasis_) + POSE_FEATURES_SIZE,1),
    verbose(false),
    dmp_list(dim_),
    dmp_goal(dim_),
    k_gain(100),
    d_gain(20),
    tau(2.0),
    goal_threshold(dim_,0.01),
    dmp_velocity_multiplier(0.1),
    attached(false),
    collision_detection_step(4)
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

  /** check for collisions */
  void DmpTrajectoryDistribution::setCollisionChecker(GridPlanner *c) {
    checker = c;
  }

  void DmpTrajectoryDistribution::addNoise(double d) {
    for (int j = 0; j < nvars; ++j) {
      //dist.ns[0].P(j,j) += d;
      if (j < POSE_RPY_SIZE) { 
        dist.ns[0].P(j,j) += d;
      }
      else {
        dist.ns[0].P(j,j) += 100*d;
      }
    }
    dist.Update();
  }

  void DmpTrajectoryDistribution::addNoise(std::vector<double> sigma) {
    if (sigma.size() < nvars) {
      if (verbose) {
        std::cerr << "[GRID/TRAJECTORY DISTRIBUTION] Noise argument for trajectory search initialization was the wrong size!" << std::endl;
        std::cerr << "[GRID/TRAJECTORY DISTRIBUTION] Should be: " << dim << std::endl;
        std::cerr << "[GRID/TRAJECTORY DISTRIBUTION] Was: " << sigma.size() << std::endl;
      }
      for (int j = 0; j < nvars; ++j) {
        if (j < POSE_RPY_SIZE) { 
          dist.ns[0].P(j,j) = DEFAULT_SIGMA;
        }
        else {
          dist.ns[0].P(j,j) = 100*DEFAULT_SIGMA;
        }
      }

    } else {
      for (int j = 0; j < nvars; ++j) {
        dist.ns[0].P(j,j) = sigma[j];
      }
    }
    dist.Update();
  }

  /**
   * use a model of a skill
   * compute the grasp pose from that skill and use it to plan for now
   */
  void DmpTrajectoryDistribution::attachObjectFromSkill(Skill &skill) {
    attachObjectFrame(skill.getInitializationFinalPose());
  }

  /**
   * set an attached object
   */
  void DmpTrajectoryDistribution::attachObjectFrame(const Pose &pose) {
    attachedObjectFrame = pose;
    attached = true;
  }

  /*
   * remove object
   */
  void DmpTrajectoryDistribution::detachObjectFrame() {
    attached = false;
  }

  void DmpTrajectoryDistribution::initializePose(TestFeatures &features, const Skill &skill, bool initBegin) {
    Pose p1 = features.lookup(skill.getInitializationFeature());
    //std::cout << p1 << std::endl;
    if (not initBegin) {
      p1 = p1 * skill.getInitializationFinalPose();
    } else {
      p1 = p1 * skill.getInitializationStartPose();
    }

    if (skill.hasAttachedObject()) {
      if (attached) {
        std::cout << "Attached frame loaded from DISTRIBUTION\n";
        //std::cout << attachedObjectFrame << "\n";
        p1 = p1 * attachedObjectFrame.Inverse();
      } else if (features.hasAttachedObjectFrame()) {
        std::cout << "Attached frame loaded from FEATURES\n";
        //std::cout << features.getAttachedObjectFrame() << "\n";
        p1 = p1 * features.getAttachedObjectFrame().Inverse();
      } else {
        std::cerr << __FILE__ << ":" << __LINE__ << ": This skill requires an attached object and none was provided!\n";
        assert(false);
      }
    }

    unsigned int idx = 0;

    dist.ns[0].mu[POSE_FEATURE_X] = p1.p.x();
    dist.ns[0].mu[POSE_FEATURE_Y] = p1.p.y();
    dist.ns[0].mu[POSE_FEATURE_Z] = p1.p.z();

    double roll, pitch, yaw;
    p1.M.GetRPY(roll, pitch, yaw);

    dist.ns[0].mu[POSE_FEATURE_YAW] = yaw;
    dist.ns[0].mu[POSE_FEATURE_PITCH] = pitch;
    dist.ns[0].mu[POSE_FEATURE_ROLL] = roll;
  }

  /**
   * initialize
   * set up the distribution based on a skill and an environment
   */
  void DmpTrajectoryDistribution::initialize(TestFeatures &features, const Skill &skill, bool initBegin, std::vector<double> sigma) {

    initializePose(features,skill,initBegin);

    for (int j = POSE_RPY_SIZE; j < nvars; ++j) {
      dist.ns[0].mu[j] = 0;//(double)(j % nbasis) / (double)nbasis;
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
        if (j < POSE_RPY_SIZE) { 
          dist.ns[0].P(j,j) = DEFAULT_SIGMA;
        }
        else {
          dist.ns[0].P(j,j) = 10*DEFAULT_SIGMA;
        }
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
  unsigned int DmpTrajectoryDistribution::sample(
      const std::vector<JointTrajectoryPoint> &start_pts,
      std::vector<EigenVectornd> &params,
      std::vector<JointTrajectory> &trajs, unsigned int nsamples) {

    using KDL::Vector;
    using KDL::Rotation;

    int ik_tries = 0;

    if (nsamples == 0) {
      nsamples = params.size();
    } else {
      params.resize(nsamples);
    }
    trajs.resize(nsamples);

    int sample = 0;
    while (sample < nsamples) {

      //EigenVectornd vec(nvars);
      //vec.resize(nvars);

      params[sample].resize(nvars);
      dist.Sample(params[sample]);

#if SHOW_SAMPLED_VALUES
      std::cout << "Sampled: ";
      for (int j = 0; j < dim; ++j) {
        std::cout << params[sample][j] << " ";
      }
      std::cout << std::endl;
#endif

      // convert the first six into a pose
      Vector v1 = Vector(params[sample][POSE_FEATURE_X],params[sample][POSE_FEATURE_Y],params[sample][POSE_FEATURE_Z]);
      Rotation r1 = Rotation::RPY(params[sample][POSE_FEATURE_ROLL],params[sample][POSE_FEATURE_PITCH],params[sample][POSE_FEATURE_YAW]);
      Pose p(r1,v1);

      robot->updateHint(start_pts[sample].positions);
      robot->updateVelocityHint(start_pts[sample].velocities);
      int ik_result = robot->IkPos(p,q);

      if (ik_result < 0) {
        ++ik_tries;
        if (ik_tries > 2*nsamples) {
          std::cerr << __FILE__ << ":" << __LINE__ << ": We are really having trouble with IK!\n";
          //for (unsigned int i = 0; i < POSE_RPY_SIZE; ++i) {
          //  std::cout << params[sample][i];
          //}
          std::cout << std::endl;
          std::cout << "XYZ = " << v1 << std::endl;
          std::cout << "RPY = " << params[sample][POSE_FEATURE_ROLL] << ", "
            << params[sample][POSE_FEATURE_PITCH] << ", " << params[sample][POSE_FEATURE_YAW] << "\n";
          std::cout << "Max tries reached: " << 10*nsamples << "\n";
          break;
        }
        continue;
      }

      unsigned int idx = 0;
      //std::cout << "GOAL = ";
      for (; idx < dim; ++idx) {
        dmp_goal[idx] = q(idx);
        //std::cout << q(idx) << " ";
      }
      //std::cout << std::endl;

      for (unsigned int i = 0; i < dim; ++i) {
        for (unsigned int j = 0; j < nbasis; ++j) {
          dmp_list[i].weights[j] = params[sample][idx++];
        }
      }

      unsigned char at_goal;
      dmp::DMPTraj plan;


      //std::cout << sample << "\n" << start_pts.size() << "\n";
      //std::cout << "pos " << start_pts[0].positions.size() << "\n";
      //for (const double &d: start_pts[0].positions) {
      //  std::cout << d << " ";
      //}
      //std::cout << std::endl;

      dmp::generatePlan(dmp_list,
                        start_pts[sample].positions,
                        start_pts[sample].velocities,
                        0,dmp_goal,goal_threshold,-1,tau,0.1,5,plan,at_goal);

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

      if (checker) {
        bool collision = !checker->TryTrajectory(trajs[sample],collision_detection_step);

        if (collision) {
          //std::cout << "COLLISION DETECTED!\n";
          ++ik_tries;
          if (ik_tries > 2*nsamples) {
            std::cerr << __FILE__ << ":" << __LINE__ << ": We are really having trouble with collisions!\n";
            break;
          } else {
            continue;
          }
        }
      }

      ++sample;
    }

    //std::cout << "sampled " << sample << "\n";
    return sample;
  }



  /* ==================================================================== */
  /* ==================================================================== */
  /*             BELOW THIS: SAME AS TRAJECTORY DISTRIBUTION              */
  /* ==================================================================== */
  /* ==================================================================== */

  /** is there an object attached */
  bool DmpTrajectoryDistribution::hasAttachedObject() const {
    return attached;
  }

  /** get the attached object frame */
  const Pose &DmpTrajectoryDistribution::getAttachedObjectFrame() const {
    return attachedObjectFrame;
  }

  /**
   * update
   * take a set of trajectories and samples
   * use the trajectories to reweight the distribution
   */
  void DmpTrajectoryDistribution::update(
      std::vector<EigenVectornd> &params,
      std::vector<double> &ps,
      unsigned int nsamples,
      double diagonal_noise)
  {
    update(params,ps,nsamples,diagonal_noise,def_step_size);
  }

  /**
   * update
   * take a set of trajectories and samples
   * use the trajectories to reweight the distribution
   */
  void DmpTrajectoryDistribution::update(
      std::vector<EigenVectornd> &params,
      std::vector<double> &ps,
      unsigned int nsamples)
  {
    update(params,ps,nsamples,diagonal_sigma,def_step_size);
  }


  /**
   * update
   * take a set of trajectories and samples
   * use the trajectories to reweight the distribution
   */
  void DmpTrajectoryDistribution::update(
      std::vector<EigenVectornd> &params,
      std::vector<double> &ps,
      unsigned int full_nsamples,
      double diagonal_noise,
      double step_size)
  {

    unsigned int nsamples = full_nsamples;
    double psum = 0;
    for (unsigned int i = 0; i < full_nsamples; ++i) {
      assert(not isnan(ps[i]));
      if (params[i].size() == 0) {
        std::cout << __FILE__ << ":" << __LINE__ << ": Not enough params; skipping the rest.\n";
        nsamples = i;
        break;
      }
      psum += ps[i];
    }

    if (psum == 0) {
      return;
    }

    if (dist.k == 1) {

      // one cluster only
      // compute mean

      //std::cout << "BEFORE:\n";
      //std::cout << dist.ns[0].P << "\n";

      dist.ns[0].mu *= (1 - step_size); //setZero();
      dist.ns[0].P *= (1 - step_size); //setZero();

      for (unsigned int i = 0; i < nsamples; ++i) {
        //std::cout << "mu rows = " << dist.ns[0].mu.rows() << ", vec rows = " << vec.rows() << std::endl;
        //std::cout << "mu cols = " << dist.ns[0].mu.cols() << ", vec cols = " << vec.cols() << std::endl;
        double wt = step_size * ps[i] / psum;
        assert (not isnan(wt));
        dist.ns[0].mu += params[i] * wt;
      }

      for (unsigned int i = 0; i < nsamples; ++i) {
        double wt = step_size * ps[i] / psum;
        //std::cout << wt << ", " << ps[i] << ", " << psum << std::endl;
        //dist.ns[0].P += wt * (params[i] - dist.ns[0].mu) * (params[i] - dist.ns[0].mu).transpose();
        dist.ns[0].P += (wt * (params[i] - dist.ns[0].mu)).array().square().matrix().asDiagonal();// * (params[i] - dist.ns[0].mu);
      }

      //std::cout << "AFTER:\n";
      //std::cout << dist.ns[0].P << "\n";

    } else {

      // set up weighted data
      // and then fit GMM again

      std::vector<std::pair<EigenVectornd,double> > data(nsamples);//ps.size());
      for (unsigned int i = 0; i < nsamples; ++i) {
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

  /**
   * set the skip between calling collision detection
   */
  unsigned int DmpTrajectoryDistribution::setCollisionDetectionStep(unsigned int step) {
    collision_detection_step = step;
  }

}
