#ifndef _GRID_LOAD_WAM_SKILLS
#define _GRID_LOAD_WAM_SKILLS

#include <grid/skill.h>

namespace grid {

  std::unordered_map<std::string, TestFeaturesPtr> setupTestFeaturesForTrials() {


    /* Initialize Base Set of Features */
    TestFeaturesPtr tf11(new TestFeatures());
    tf11->setAgentFrame("wam/wrist_palm_link")
      .setWorldFrame("wam/base_link")
      .addFeature("node",POSE_FEATURE)
      .addFeature("link",POSE_FEATURE)
      .addFeature("time",TIME_FEATURE);

    TestFeaturesPtr tf12(new TestFeatures());
    tf12->setAgentFrame("wam/wrist_palm_link")
      .setWorldFrame("wam/base_link")
      .addFeature("node",POSE_FEATURE)
      .addFeature("link",POSE_FEATURE)
      .addFeature("time",TIME_FEATURE);

    TestFeaturesPtr tf13(new TestFeatures());
    tf13->setAgentFrame("wam/wrist_palm_link")
      .setWorldFrame("wam/base_link")
      .addFeature("node",POSE_FEATURE)
      .addFeature("link",POSE_FEATURE)
      .addFeature("time",TIME_FEATURE);

    TestFeaturesPtr tf14(new TestFeatures());
    tf14->setAgentFrame("wam/wrist_palm_link")
      .setWorldFrame("wam/base_link")
      .addFeature("node",POSE_FEATURE)
      .addFeature("link",POSE_FEATURE)
      .addFeature("time",TIME_FEATURE);

    TestFeaturesPtr tf21(new TestFeatures());
    tf21->setAgentFrame("wam/wrist_palm_link")
      .setWorldFrame("wam/base_link")
      .addFeature("node",POSE_FEATURE)
      .addFeature("link",POSE_FEATURE)
      .addFeature("time",TIME_FEATURE);

    TestFeaturesPtr tf22(new TestFeatures());
    tf22->setAgentFrame("wam/wrist_palm_link")
      .setWorldFrame("wam/base_link")
      .addFeature("node",POSE_FEATURE)
      .addFeature("link",POSE_FEATURE)
      .addFeature("time",TIME_FEATURE);

    TestFeaturesPtr tf23(new TestFeatures());
    tf23->setAgentFrame("wam/wrist_palm_link")
      .setWorldFrame("wam/base_link")
      .addFeature("node",POSE_FEATURE)
      .addFeature("link",POSE_FEATURE)
      .addFeature("time",TIME_FEATURE);

    TestFeaturesPtr tf24(new TestFeatures());
    tf24->setAgentFrame("wam/wrist_palm_link")
      .setWorldFrame("wam/base_link")
      .addFeature("node",POSE_FEATURE)
      .addFeature("link",POSE_FEATURE)
      .addFeature("time",TIME_FEATURE);

    tf11->setFrame("gbeam_node_1/gbeam_node","node").setFrame("gbeam_link_1/gbeam_link","link");
    tf12->setFrame("gbeam_node_1/gbeam_node","node").setFrame("gbeam_link_1/gbeam_right","link");
    tf13->setFrame("gbeam_node_1/gbeam_node","node").setFrame("gbeam_link_1/gbeam_left","link");
    tf14->setFrame("gbeam_node_1/gbeam_node","node").setFrame("gbeam_link_1/gbeam_back","link");
    tf21->setFrame("gbeam_node_2/gbeam_node","node").setFrame("gbeam_link_1/gbeam_link","link");
    tf22->setFrame("gbeam_node_2/gbeam_node","node").setFrame("gbeam_link_1/gbeam_right","link");
    tf23->setFrame("gbeam_node_2/gbeam_node","node").setFrame("gbeam_link_1/gbeam_left","link");
    tf24->setFrame("gbeam_node_2/gbeam_node","node").setFrame("gbeam_link_1/gbeam_back","link");

    std::unordered_map<std::string, TestFeaturesPtr> features;

    features["node1,link1"] = tf11;
    features["node1,link2"] = tf12;
    features["node1,link3"] = tf13;
    features["node1,link4"] = tf14;
    features["node2,link1"] = tf21;
    features["node2,link2"] = tf22;
    features["node2,link3"] = tf23;
    features["node2,link4"] = tf24;

    return features;
  }


  std::unordered_map<std::string, SkillPtr> loadWamSkills() {

    std::unordered_map<std::string, SkillPtr> skills;

    SkillPtr approach(new Skill("approach",3));
    SkillPtr approach_right(new Skill("approach_right",3));
    SkillPtr approach_left(new Skill("approach_left",3));
    SkillPtr grasp(new Skill("grasp",1));
    SkillPtr align(new Skill("align",3));
    SkillPtr place(new Skill("place",2));
    SkillPtr release(new Skill("release",1));
    SkillPtr disengage(new Skill("disengage",3));

    /* SET UP THE SKILLS */
    approach->appendFeature("link").appendFeature("time").setInitializationFeature("link").setStatic(false).setPrior(3.0 / 6.0);
    approach_left->appendFeature("link").appendFeature("time").setInitializationFeature("link").setStatic(false).setPrior(1.0 / 6.0);
    approach_right->appendFeature("link").appendFeature("time").setInitializationFeature("link").setStatic(false).setPrior(2.0 / 6.0);
    grasp->appendFeature("link").setInitializationFeature("link").setStatic(true);
    align->appendFeature("node").appendFeature("time").setInitializationFeature("node").attachObject("link").setStatic(false);
    place->appendFeature("node").appendFeature("time").setInitializationFeature("node").attachObject("link").setStatic(false);
    release->appendFeature("node").setInitializationFeature("node").attachObject("link").setStatic(true);
    disengage->appendFeature("link").appendFeature("time").setInitializationFeature("link").setStatic(false);

    /* SET UP THE ROBOT KINEMATICS */
    RobotKinematicsPtr rk_ptr = RobotKinematicsPtr(new RobotKinematics("robot_description","wam/base_link","wam/wrist_palm_link"));

    /* LOAD TRAINING DATA FOR APPROACH */
    {
      std::string filenames[] = {"data/sim/approach01.bag", "data/sim/approach02.bag", "data/sim/approach03.bag"};
      load_and_train_skill(*approach, rk_ptr, filenames, 3);
    }
    /* LOAD TRAINING DATA FOR APPROACH RIGHT */
    {
      std::string filenames[] = {"data/sim/approach_right01.bag", "data/sim/approach_right02.bag", "data/sim/approach_right03.bag"};
      load_and_train_skill(*approach_right, rk_ptr, filenames, 3);
    }
    /* LOAD TRAINING DATA FOR APPROACH LEFT */
    {
      std::string filenames[] = {"data/sim/approach_left01.bag", "data/sim/approach_left02.bag", "data/sim/approach_left03.bag"};
      load_and_train_skill(*approach_left, rk_ptr, filenames, 3);
    }
    /* LOAD TRAINING DATA FOR GRASP */
    {
      std::string filenames[] = {"data/sim/grasp01.bag", "data/sim/grasp02.bag", "data/sim/grasp03.bag"};
      load_and_train_skill(*grasp, rk_ptr, filenames, 3);
    }
    /* LOAD TRAINING DATA FOR ALIGN */
    {
      std::string filenames[] = {"data/sim/align1.bag", "data/sim/align2.bag", "data/sim/align3.bag"};
      load_and_train_skill(*align, rk_ptr, filenames, 3);
    }
    /* LOAD TRAINING DATA FOR PLACE */
    {
      std::string filenames[] = {"data/sim/place1.bag", "data/sim/place3.bag"};
      load_and_train_skill(*place, rk_ptr, filenames, 2);
    }
    /* LOAD TRAINING DATA FOR RELEASE */
    {
      std::string filenames[] = {"data/sim/release1.bag", "data/sim/release2.bag", "data/sim/release3.bag",
        //"data/sim/release1b.bag", "data/sim/release2b.bag", "data/sim/release3b.bag"
      };
      load_and_train_skill(*release, rk_ptr, filenames, 3);
    }
    /* LOAD TRAINING DATA FOR DISENGAGE */
    {
      std::string filenames[] = {"data/sim/disengage1.bag", "data/sim/disengage2.bag", "data/sim/disengage3.bag"};
      load_and_train_skill(*disengage, rk_ptr, filenames, 3);
    }

    skills["approach"] = approach;
    skills["approach_right"] = approach_right;
    skills["approach_left"] = approach_left;
    skills["grasp"] = grasp;
    skills["align"] = align;
    skills["place"] = place;
    skills["release"] = release;
    skills["disengage"] = disengage;

    return skills;

  }
}

#endif
