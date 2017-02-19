import argparse

import task_tree_search.road_world as rw
import task_tree_search.models.adversary as adv

def agents():
  return ["ddpg", "cdqn", "supervised", "cem"]

def keras_rl_agents():
  return ["ddpg", "cdqn"]

def trainers():
  return ["supervised", "cem"]

# For sampling new MCTS nodes from
def get_action_sets():
    return ["default", "nn"]

def get_rollout_types():
    return ["none", "simulation", "action_value"]

def get_rollout_models():
    return ["none", "supervised", "rl"]

def get_mcts_agents():
    return ["none", "dqn"]

def get_initialization_types():
    return ["none", "action_set"]

def get_initialization_priors():
    return ["default", "nn"]

def get_animate_modes():
    return ["none", "tree", "pretty"]

def get_test_modes():
    return ["mcts", "option"]

def get_base_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--visualize",
            action="store_true",
            help="Display graphs and other stuff.")
    parser.add_argument("--ltl",
            action="store_true",
            help="Use LTL constraints.")
    parser.add_argument("--lateral",
            action="store_true",
            help="Use simplified lateral motions instead of steering angle.")
    parser.add_argument("--cpu",
            action="store_true",
            help="Force use of cpu0")
    parser.add_argument("--verbose",
            action="store_true",
            help="Print out causes of termination conditions and other information.")

    return parser

def get_train_parser():
    parser = argparse.ArgumentParser(add_help=False,
            parents=[get_base_parser()])
    parser.add_argument("--option_name", default="Planning")
    parser.add_argument("--action_set", choices=get_action_sets(), default="default",
            help="Use a set of hand-coded policies ('default') or neural net options ('nn').")
    parser.add_argument("--mcts",
            action="store_true",
            help="Force MCTS version of discrete action selection problem (training only).")

    return parser

def get_mcts_parent_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rollout_type", choices=get_rollout_types(), default="none")
    parser.add_argument("--rollout_model", choices=get_rollout_models(), default="none")
    parser.add_argument("--agent_name", choices=get_mcts_agents(), default="dqn",
            help="Use weights learned by a particular agent (e.g, 'dqn'), or no weights at all ('none')")
    parser.add_argument("--initialization_type",
        choices=get_initialization_types(),
        default="action_set")
    parser.add_argument("--initialization_prior",
        choices=get_initialization_priors(),
        default="default")
    parser.add_argument("--seed", type=int, default=None,
            help="Random seed to use.")
    parser.add_argument("--profile",
            action="store_true",
            help="Run a profiler on the planner.")
    parser.add_argument("--depth",
            type=int,
            default=10)
    parser.add_argument("--run",
            action="store_true",
            help="Run in a loop until we actually hit a termination condition.")
    parser.add_argument("--graph",
            action="store_true",
            help="Show output graphs displaying information about the chosen trajectory.")
    parser.add_argument("--iter", type=int, default=10,
            help="Number of MCTS iterations to perform during each step.")
    parser.add_argument("--dfs",
            action="store_true",
            help="MCTS replace rollouts with depth first search expansion")
    parser.add_argument("--hz",
            type=int,
            default=0)
    parser.add_argument("--animate",
            action="store_true",
            help="Animate MCTS results at given framerate.")
    parser.add_argument("--save",
            action="store_true",
            help="save resulting images from a test run to disk.")

    return parser

def get_option_parent_parser():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("option",choices=rw.options.getOptionsList(),default="Default")
    parser.add_argument("--agent",choices=agents(),
            default="ddpg",
            help="type of RL agent to use when learning behavior.")
    parser.add_argument("--seed", default=None,
            help="Random seed to use.")
    parser.add_argument("--demo",
            action="store_true",
            help="loop animation after training is complete")
    parser.add_argument("--visualize",
            action="store_true",
            help="visualize during training")
    parser.add_argument("--test_episodes",type=int,default=100,
            help="number of episodes to show during testing")
    parser.add_argument("--train_steps",type=int,default=50000,
            help="number of training steps to perform")
    parser.add_argument("--test_only",
            action="store_true",
            help="test learned weights")
    parser.add_argument("--hz",
            type=int,
            default=0)
    parser.add_argument("--cpu",
            action="store_true",
            help="force use cpu0")
    parser.add_argument("--device",
            default=None,
            help="set tensorflow device")
    parser.add_argument("--lateral",
            action="store_true",
            help="Use simplified lateral motions instead of steering angle.")
    parser.add_argument("--ltl",
            action="store_true",
            help="Use LTL constraints.")
    parser.add_argument("--restart",
            action="store_true",
            help="Load neural net model from a previous training iteration.")
    parser.add_argument("--adversary",
            choices=adv.getAvailableAdversaries(),
            default="random",
            help="Set a policy for choosing environments during learning.")
    #parser.add_argument("--hard",
    #        action="store_true",
    #        help="Enable 'hard retrain' mode to improve performance.")
    parser.add_argument("--verbose",
            action="store_true",
            help="Print out causes of termination conditions and other information.")

    return parser

def get_option_parser():
    parser=argparse.ArgumentParser(prog="option",parents=[get_option_parent_parser()])
    return parser

def get_mcts_parser():
    parser = argparse.ArgumentParser(
            prog="mcts",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='Run an MCTS test. Example configurations may include:\n\n' + \
                    '\tmcts --rollout_type action_value --rollout_model supervised --action_set nn\n' + \
                    '\tmcts --rollout_type simulation --rollout_model rl --action_set nn\n' + \
                    '\tmcts --rollout_type simulation --action_set default',
            epilog="Specify the parameters you want for your particular test.",
            parents=[get_mcts_parent_parser(), get_train_parser()])

    return parser

def get_test_parser():
    mcts_parser = get_mcts_parent_parser()
    parser = argparse.ArgumentParser(
            prog="test",
            description="This is the test tool. " + \
                    "You can use it to play a test forward at a reasonable" + \
                    " speed, or to aggregate measures and data about any" + \
                    "tests that you may want to perform.",
            parents=[mcts_parser])
    parser.add_argument("--num_tests", type=int, default=1,
            help="Number of tests to run.")
    parser.add_argument("--animate_mode", choices=get_animate_modes(), default="tree",
            help="Animate the chosen sequence of actions." + \
                 "This can be 'none' to avoid any windows, or 'tree' to " + \
                 "show the MCTS process. " + \
                 "Choosing 'pretty' will open in PyGame window and animate" + \
                 "at 100 fps.")
    parser.add_argument("--show_graphs", type=bool, default=True,
            help="Display graphs showing performance. " + \
                    "These include acceleration, velocity, reward, etc.")
    parser.add_argument("--show_trees", type=bool, default=True,
            help="Display trees output by the planner.")
    parser.add_argument("--test_mode", choices=get_test_modes(), default="mcts")
    parser.add_argument("--time", type=bool, default=True,
            help="Time MCTS planning test iterations and output related statistics.")

    return parser

def get_evaluate_option_parser():
  parser = argparse.ArgumentParser(
      prog="evaluate_option",
      description="This collects metrics on the evaluation of a particular" + \
          " Road World option.",
      parents=[get_base_parser()])
  parser.add_argument("option",choices=rw.options.getOptionsList(),default="Default")
  parser.add_argument("--test_episodes",type=int,default=100,
          help="number of episodes to show during testing")
  parser.add_argument("--test_length",type=int,default=1000,
          help="maximum length of a test")
  parser.add_argument("--agent_name",choices=agents(),
          default="ddpg",
          help="type of RL agent used when learning behavior.")
  parser.add_argument("--seed", type=int, default=None,
          help="Random seed to use.")
  parser.add_argument("--use_fast",
          action="store_true",
          help="Use fast network model.")
  parser.add_argument("--baseline",
          action="store_true",
          help="Use manual policy.")
  parser.add_argument("--override",
          action="store_true",
          help="Force use the 'Planning' problem definition.")

  return parser
