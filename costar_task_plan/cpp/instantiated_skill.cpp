#include <grid/instantiated_skill.h>
#include <grid/visualize.h>
//#include <grid/utils/params.hpp>

using namespace costar_plan_msgs;

using trajectory_msgs::JointTrajectory;
using trajectory_msgs::JointTrajectoryPoint;

namespace grid {


  /**
   * set all variables back to original values
   * set all children to not done
   */
  void InstantiatedSkill::reset() {
    model_norm = p.base_model_norm;
    best_p = LOW_PROBABILITY;
    cur_iter = 0;
    good_iter = 0;
    good_iter = 0;
    if(dmp_dist and touched) {
      //dmp_dist->initializePose(*features,*skill);
      dmp_dist->addNoise(0.0005);
    }
    touched = false;
    best_idx = 0;
    for (double &d: iter_lls) {
      d = 0;
    }
    for (InstantiatedSkillPtr ptr: next) {
      ptr->reset();
    }
    //for (unsigned int i = 0; i < T.size(); ++i) {
    //  last_T[i] = 1.0 / T.size();
    //  T[i] = 1.0 / T.size();
    //}
  }

  /**
   * normalize the transition probabilities
   */
  void InstantiatedSkill::updateTransitions() {


    //if (p.random_transitions) { return; }

    double last_tsum = 0;
    double tsum = 0;
    for (unsigned int i = 0; i < T.size(); ++i) {
      last_tsum += last_T[i];
      tsum += T[i];
    }

    if (tsum < 1e-200 or p.random_transitions) {
      for (unsigned int i = 0; i < T.size(); ++i) {
        T[i] = last_T[i]/last_tsum;
      }
      return;
    } else {
      for(unsigned int i = 0; i < T.size(); ++i) {
        T[i] = ((1 - p.step_size)*(last_T[i]/last_tsum))
          + (p.step_size * (T[i] / tsum));
        last_T[i] = T[i];
      }
    }

    if (p.verbosity > 0) {
      std::cout << "Transitions: ";
      for(unsigned int i = 0; i < T.size(); ++i) {
        std::cout << T[i] << " ";
      }
      std::cout << std::endl;
    }
  }


  /*
   * set prior
   */
  InstantiatedSkill &InstantiatedSkill::setPrior(const double &prior_) {
    prior = prior_;
  }

  // randomly sample an index from the probabilities
  unsigned int InstantiatedSkill::sampleIndex(unsigned int nsamples) const {
    // sample a random index from the skill

    assert(fabs(acc.at(nsamples-1) - 1) < 1e-5);

    double r = (double)rand() / RAND_MAX;
    //double r = unif_rand(re);
    for (unsigned int i = 0; i < nsamples; ++i) {
      if (r < acc.at(i)) return i;
    }
    return 0;
  }

  void InstantiatedSkill::accumulateProbs(const std::vector<double> &prev_ps, unsigned int len) {
    /************* ACCUMULATE PROBABILITIES *************/
    for (unsigned int i = 0; i < len; ++i) {
      if (i > 0) {
        acc[i] = exp(prev_ps[i]) + acc[i-1] + 0.1;
      } else {
        acc[i] = exp(prev_ps[i]) + 0.01;
      }
      //acc[i] = i+1;
    }
    for (double &d: acc) {
      d /= acc[len-1];
      assert (not isnan(d));
    }

    if (p.verbosity > 3) {
      std::cout << "accumulating for " << id << ": ";
      for (double &d: acc) {
        std::cout << d << " ";
      }
      std::cout << "\n";
    }
  }

  void InstantiatedSkill::copyEndPoints(const std::vector<JointTrajectoryPoint> &prev_end_pts,
                                        const std::vector<double> &prev_ps,
                                        unsigned int len)
  {
    for (unsigned int i = 0; i < len; ++ i) {
      start_ps[i] = prev_ps.at(i);
      end_pts[i].positions = prev_end_pts.at(i).positions;
      end_pts[i].velocities = prev_end_pts.at(i).velocities;
    }
  }

  /**
   * add some noise and refresh norm terms
   */
  void InstantiatedSkill::refresh(int horizon) {
    //std::cout << "refreshing\n";
    //model_norm = p.base_model_norm;
    good_iter = 0;
    if(dmp_dist) {
      dmp_dist->addNoise(0.0001);
    }
    if (not p.random_transitions) {
      for (double &t : T) {
        t = 1;
      }
    }
    updateTransitions();
    model_norm = p.base_model_norm;
    if (horizon > 0) {
      for (auto &child: next) {
        child->refresh(horizon-1);
      }
    }
  }

  void InstantiatedSkill::updateCurrentAttachedObjectFrame() {
    if (skill and skill->hasAttachedObject()) {
      std::cout << "UPDATING: " << skill->getAttachedObject() << "\n";
      useCurrentFeatures = true;
      currentAttachedObjectFrame = features->lookup(AGENT).Inverse() * features->lookup(skill->getAttachedObject());
    }
    for (auto &child: next) {
      child->updateCurrentAttachedObjectFrame();
    }
  }

  /** 
   * find best entries
   */
  void InstantiatedSkill::updateBest(unsigned int nsamples) {
    best_p = 0;
    //if (skill) std::cout << skill->getName() << " ";
    for (unsigned int i = 0; i < nsamples; ++i) {
      //std::cout << ps[i] << " ";
      if (ps[i] > best_p) {
        best_p = ps[i];
        best_idx = i;
      }
    }
    //std::cout << "\n";
  }

  /**
   * print out debug info on child probabilities and current ("my") probabilities
   */
  void InstantiatedSkill::debugPrintCurrentChildProbabilities(unsigned int samples) {
    for (unsigned int i = 0; i < samples; ++i) {
      if (skill) {
        std::cout << "[" << id << "] " << skill->getName()
          << ": "<<my_ps[i];
      } else {
        std::cout << "[" << id << "] [no skill]"
          << ": "<<my_ps[i];
      }
      for (unsigned int next = 0; next < T.size(); ++next) {
        std::cout << " + "<< next_ps[next][i];
      }
      std::cout << " = " << my_ps[i] << " + " << avg_next_ps[i];
      std::cout << "\n";
    }
  }

  /**
   * run a single iteration of the loop. return a set of trajectories.
   * this is very similar to code in the demo
   */
  void InstantiatedSkill::step(const std::vector<double> &prev_ps,
                               const std::vector<JointTrajectoryPoint> &prev_end_pts,
                               std::vector<double> &ps_out,
                               //std::vector<unsigned int> &prev_counts,
                               double &probability,
                               unsigned int len,
                               int horizon,
                               unsigned int nsamples)
  {

    unsigned int next_len = nsamples;
    last_samples = nsamples;
    if (len == 0 || horizon < 0 || nsamples == 0) {
      std::cout << "SKIPPING\n";
      probability = 1e-200;
      return;
    } else if (horizon == 0 || next.size() == 0) {
      initializeNextPs(next_ps,0);
    } else {
      initializeNextPs(next_ps,LOW_PROBABILITY);
    }
    touched = true;
    initializePs(prev_p_sums,0);
    initializePs(avg_next_ps,0);
    initializeCounts(prev_counts,0u);
    accumulateProbs(prev_ps,len);


    /************* SAMPLE TRAJECTORIES IF NECESSARY *************/
    if (not skill) {
      copyEndPoints(prev_end_pts, prev_ps, len);
      next_len = len;
    } else if (done) {
      next_len = 1;
      my_ps[0] = MAX_PROBABILITY;
      end_pts[0].positions = trajs[best_idx].points.rbegin()->positions;
      end_pts[0].velocities = trajs[best_idx].points.rbegin()->velocities;
    } else {

      // sample start points
      for (unsigned int i = 0; i < nsamples; ++i) {

        unsigned int idx = sampleIndex(len);
        assert (idx < len);

        // sample an index
        start_pts[i].positions = prev_end_pts[idx].positions;
        start_pts[i].velocities = prev_end_pts[idx].velocities;
        start_ps[i] = prev_ps[idx];
        prev_idx[i] = idx;
        ps_out[idx] = 0;
      }

      if (not skill->isStatic()) {
        // sample trajectories
        features->setUseDiff(true);
        next_len = dmp_dist->sample(start_pts,params,trajs,nsamples);
      } else {
        features->setUseDiff(false);
        // just stay put
        for (unsigned int i = 0; i < nsamples; ++i) {
          trajs[i].points.resize(1);
          trajs[i].points[0].positions = start_pts[i].positions;
          trajs[i].points[0].velocities.resize(start_pts[i].positions.size());
          for (double &v : trajs[i].points[0].velocities) {
            v = 0.;
          }
        }
      }

      // compute probabilities
      for (unsigned int j = 0; j < nsamples; ++j) {

        // TODO: speed this up
        std::vector<Pose> poses = robot->FkPos(trajs[j]);

        skill->resetModel();
        skill->addModelNormalization(model_norm);

        if (useCurrentFeatures) {
          //std::cout << currentAttachedObjectFrame << "\n";
          features->getFeaturesForTrajectory(
              traj_features,
              skill->getFeatures(),
              poses,
              skill->hasAttachedObject(),
              currentAttachedObjectFrame);
        } else {
          features->getFeaturesForTrajectory(
              traj_features,
              skill->getFeatures(),
              poses,
              dmp_dist->hasAttachedObject(),
              dmp_dist->getAttachedObjectFrame());
        }
        skill->normalizeData(traj_features);
        FeatureVector v = skill->logL(traj_features);
        my_ps[j] = log(v.array().exp().sum() / v.size()); // would add other terms first

        if (p.verbosity > 1) {
          std::cout << "[" << id << "] " << j << ": " << skill->getName() << ": "<< my_ps[j]<<" + "<< start_ps[j]<<"\n";
        }

        if (trajs[j].points.size() < 1) {
          my_ps[j] = LOW_PROBABILITY;
          continue;
        }

        end_pts[j].positions.resize(robot->getDegreesOfFreedom());
        end_pts[j].velocities.resize(robot->getDegreesOfFreedom());

        // set up all the end points!
        for (unsigned int ii = 0; ii < robot->getDegreesOfFreedom(); ++ii) {
          end_pts[j].positions[ii] = trajs[j].points.rbegin()->positions[ii];
          end_pts[j].velocities[ii] = trajs[j].points.rbegin()->velocities[ii];
        }
      }
    }

    // check to make sure this is a valid path to explore
    double avg_so_far = 0;
    for (unsigned int j = 0; j < nsamples; ++j) {
      ps[j] = my_ps[j] + start_ps[j];
      avg_so_far += exp(my_ps[j]);
    }
    avg_so_far /= nsamples;

    // do we want to continue?
    // if so descend through the tree
    // descent through the tree
    if (horizon > 0 && log(avg_so_far) > LOW_PROBABILITY) {

      unsigned int next_skill_idx = 0;

      for (auto &ns: next) {

        unsigned int next_nsamples = floor((T[next_skill_idx]*nsamples));
        if (next_nsamples > 0) {
          ns->step(my_ps, end_pts,
                   //next_ps[next_skill_idx], next_counts[next_skill_idx], T[next_skill_idx], // outputs
                   next_ps[next_skill_idx], T[next_skill_idx], // outputs
                   next_len, horizon-1, next_nsamples); // params
        }

        if (p.verbosity > 0) {
          std::cout << " >>> probability of " << ns->skill->getName()
            << " " << ns->id << " = " << T[next_skill_idx]
            << std::endl;
        }

        ++next_skill_idx;
      }

      // compute the probabilities
      for (unsigned int i = 0; i < T.size(); ++i) {
        //double next_nsamples_ratio = floor((T[i]*nsamples)) / (double)nsamples;
        for(unsigned int j = 0; j < nsamples; ++j) {
          avg_next_ps[j] += T[i] * exp(next_ps[i][j]);
        }
      }
    } else {
      for (unsigned int i = 0; i < nsamples; ++i) {
       avg_next_ps[i] = 1; 
      }
    }


    for (unsigned int i = 0; i < nsamples; ++i) {
      avg_next_ps[i] = log(avg_next_ps[i]);
      ps[i] = start_ps[i] + my_ps[i] + avg_next_ps[i];
      my_future_ps[i] = my_ps[i] + avg_next_ps[i];
    }


    if (p.verbosity > 1) debugPrintCurrentChildProbabilities(nsamples);

    // update probabilities for all
    {
      // now loop over all the stuff!
      // go over probabilities and make sure they work
      // use the start_idx field to match start_idx to 
      double prev_psum_sum = 0;
      probability = 0;
      for (unsigned int i = 0; i < nsamples; ++i) {
        if (p.verbosity > 2) {
          std::cout << " - propogating p(" << i << ") = " << my_ps[i] + avg_next_ps[i] << " back to " << prev_idx[i] << " ... " << log(probability) << "\n";
        }
        prev_p_sums[prev_idx[i]] += exp(my_ps[i]+avg_next_ps[i]);
        ++prev_counts[prev_idx[i]];
        probability += exp(my_ps[i]+avg_next_ps[i]);
      }
      //probability /= nsamples;
      last_probability = probability;

      // update transitions based on these results
      for (unsigned int i = 0; i < len; ++i) {
        if (prev_counts[i] > 0) {
          ps_out[i] += log(prev_p_sums[i] / prev_counts[i]);
        } else {
          ps_out[i] += 0;
        }
        if (p.verbosity > 1) {
          std::cout << " - future sum for " << i << " = " << ps_out[i]
            << " (" << prev_counts[i] << " chosen, sum = " << log(prev_p_sums[i]) << ")"
            << "\n";
        }
      }
    }

    double sum = 0;
    double future_sum = 0;
    // normalize everything
    {
      for (unsigned int i = 0; i < nsamples; ++i) {
        sum += exp(ps[i]);
        future_sum += exp(my_future_ps[i]);
      }
      // normalize here
      for (unsigned int i = 0; i < nsamples; ++i) {
        ps[i] = exp(ps[i]) / sum;
        my_future_ps[i] = exp(my_future_ps[i]) / future_sum;
      }
    }

    updateBest(nsamples);
    updateTransitions();

    bool skip = false;
    if (log(sum) < LOW_PROBABILITY) {
      skip = true;
      //std::cout << "nothing here \n";
    }

    if(skill and not skill->isStatic() and not skip) {

      if (p.verbosity > 4) {
        std::cout << skill->getName() << " probabilities: ";
        for (double &p: ps) {
          std::cout << p << " ";
        }
        std::cout << std::endl;
      }

      if (nsamples > 1) {
        dmp_dist->update(params,my_future_ps,nsamples,p.noise,p.step_size);
        if (p.fixed_distribution_noise) {
          dmp_dist->addNoise(p.distribution_noise); 
        } else {
          dmp_dist->addNoise(pow(0.1,(good_iter)+4));
        }
      }
    }

    // compute ll for this iteration
    iter_lls[cur_iter] = sum / p.ntrajs;

    // decrease normalization
    if (cur_iter > 0 and 
        iter_lls[cur_iter] > 1e-20 and
        iter_lls[cur_iter] > iter_lls[cur_iter-1])
    {
      model_norm *= p.model_norm_step;
      ++good_iter;
      //std::cout << "THAT WAS GOOD\n";
    }

    if (p.verbosity >= 0) {
      if (skill) {
        std::cout << "[" << id << "] " << skill->getName() << " >>>> AVG P = " << (sum / len) << std::endl;
      } else {
        std::cout << "[" << id << "] [no skill] >>>> AVG P = " << (sum / len) << std::endl;
      }
    }

    ++cur_iter;
  }

  /**
   * descend through the tree
   * execute as we reach nodes that require it
   * use gripper tool to send messages
   */
  bool InstantiatedSkill::execute(GridPlanner &gp, actionlib::SimpleActionClient<costar_plan_msgs::CommandAction> &ac,
                                  int horizon, bool replan, int replan_depth)
  {


    assert(ros::ok());
    ros::spinOnce();

    std::cout << "EXECUTING: ";
    if (skill) {
      std::cout << skill->getName() << "\n";
    } else {
      std::cout << "n/a\n";
    }
    std::cout << "Replanning? " << replan << ": ";
    std::cout << "best idx = " << best_idx << ", ";
    std::cout << "best p = " << best_p << "\n";


    if (not touched) {
      replan = true;
    } else if (horizon > 0) {
      for (unsigned int i = 0; i < next.size(); ++i) {
        if ((not next[i]->touched) and T[i] > 1e-100) {
          std::cout << "Child " << i << " not touched yet!\n";
          replan = true;
        }
      }
    }

    // trigger action server
    CommandGoal cmd;
    if (skill) {
      cmd.name = skill->getName();
      if (not skill->isStatic()) {
        cmd.traj = trajs[best_idx];
      }
    }
    if (features) {
      cmd.keys = features->getClasses();
      cmd.values = features->getIds();
    }

    std::cout << "waiting for server... (" << horizon << ")\n";
    ac.waitForServer();
    std::cout << "sending command...\n";
    ac.sendGoal(cmd);
    std::cout << "waiting for result\n";
    ac.waitForResult();
    done = true;

    if (horizon > 0 && next.size() > 0) {
      // replan if necessary
      if (replan or (skill and skill->isStatic())) {

        std::cout << "In main replan loop.\n";

        useCurrentFeatures = true;
        updateCurrentAttachedObjectFrame();

        std::cout << ">> " << gp.currentPos()[0] << "\n";
        if (robot) {
          robot->updateHint(gp.currentPos());
          robot->updateVelocityHint(gp.currentVel());
        }
        features->updateWorldfromTF();
        updateCurrentAttachedObjectFrame();

        std::vector<trajectory_msgs::JointTrajectoryPoint> starts(1);
        std::vector<double> root_next_ps(1);

        start_ps.resize(1);
        start_ps[0] = MAX_PROBABILITY;


        for (auto &pt: starts) {
          pt.positions = gp.currentPos();
          pt.velocities = gp.currentVel();
        }

        reset();
        double probability = MAX_PROBABILITY;
        int my_horizon = horizon;
        if (replan_depth > 0) {
          std::cout << "setting replan depth: ";
          my_horizon = replan_depth;
          std::cout << "horizon = " << my_horizon << "\n";
        } else {
          std::cout << "horizon = " << my_horizon << "\n";
        }
        for (unsigned int i = 0; i < p.iter; ++i) {

          assert(ros::ok());
          ros::spinOnce();
          features->updateWorldfromTF();

          step(start_ps,starts,
               root_next_ps,probability,
               1,my_horizon,p.ntrajs);
          publish();

          //replan = false;
          if (i > 0 and fabs(iter_lls[i] - iter_lls[i-1]) < (p.update_horizon * iter_lls[i])) {
            if (not replan_depth and my_horizon < horizon) {
              my_horizon++;
              refresh(my_horizon);
            } else {
              break;
            }
          }
        }
      }

      // continue execution
      double best_t_p = 0;
      unsigned int idx = 0;
      for (unsigned int i = 0; i < T.size(); ++i) {
        if (T[i] > best_t_p) {
          idx = i;
          best_t_p = T[i];
        }
      }

      //replan = replan or (skill and skill->isStatic());
      return next[idx]->execute(gp,ac,horizon-1,false,replan_depth);
    } else {
      std::cout << "Execution done: " << horizon << ", " << next.size() << "\n";
      return true;
    }
  }

  void InstantiatedSkill::publish() {
    if (not touched) {
      return;
    } else if (pub and skill) {
      pub->publish(toPoseArray(trajs,features->getWorldFrame(),robot));
    }
    double best_t_p = 0;
    unsigned int idx = 0;
    if (next.size() > 0) {
      for (unsigned int i = 0; i < T.size(); ++i) {
        if (T[i] > best_t_p) {
          idx = i;
          best_t_p = T[i];
        }
      }
      next[idx]->publish();
    }
  }


  const Pose &InstantiatedSkill::getAttachedObjectFrame() const {
    return currentAttachedObjectFrame;
  }

}
