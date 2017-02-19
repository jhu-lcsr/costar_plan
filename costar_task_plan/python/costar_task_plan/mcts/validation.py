import matplotlib.pyplot as plt

'''
Descends through the tree, collecting valuable metrics. These metrics can then
be exposed to display all kinds of information.

At each node, it starts a thread associated with a particular simulation or performance.
'''
class Validator(object):

  '''
  Plots should be a list of tuples; e.g. ("x","y")
  '''
  def __init__(self,
      state_vars,
      action_vars,
      plots=[],
      rows=1,
      cols=1,):
    self.threads = []
    self.best_thread = []
    self.best_thread = 0
    self.next_thread = 0
    self.state_vars = state_vars
    self.action_vars = action_vars
    self.plots = plots
    self.rows = rows
    self.cols = cols

  def __call__(self, policies, node):
    path = policies.extract(node)
    self.process(node)
    self.best_thread = self.process_path(path)

  '''
  This actually does the work. It descends through a set of MCTS nodes and
  collects data from each of them, adding each one to the appropriate "thread"
  tracking that simulated trajectory.

  If thread is none, there's no parent. Otherwise we can copy data from the
  parent.
  '''
  def process(self, node, paths=[]):
    pass

  '''
  Get all the relevant data from a particular MCTS path.
  '''
  def process_path(self, path):
    data = []
    for node in path:
      for (s,a) in node.traj:
        sample = {}
        current_state_vars = vars(s)
        current_action_vars = vars(a)
        for var in self.state_vars:
          sample["state.%s"%var] = current_state_vars[var]
        for var in self.action_vars:
          sample["action.%s"%var] = current_action_vars[var]
        data.append(sample)

    if len(self.plots) > 0:
      plt.figure()
    for i,(x,y) in enumerate(self.plots):
      # make two arrays for x and y axis
      xdata = [sample[x] for sample in data]
      ydata = [sample[y] for sample in data]

      plt.subplot(self.rows, self.cols, i+1)
      plt.plot(xdata,ydata)
      plt.xlabel(x)
      plt.ylabel(y)

