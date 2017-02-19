
import matplotlib.pyplot as plt

def graphStatesAndActions(states, actions, plots, rename={}, rows=2, cols=2, fs=[]):
  data = []
  for (s,a) in zip(states, actions):
    sample = {}
    current_state_vars = vars(s)
    current_action_vars = vars(a)
    for name, f in fs:
        sample[name] = f(s,a)
    for var in current_state_vars:
      name = "state.%s"%var
      if name in rename.keys():
        name = rename[name]
      sample[name] = current_state_vars[var]
    for var in current_action_vars:
      name = "action.%s"%var
      if name in rename.keys():
        name = rename[name]
      sample[name] = current_action_vars[var]

    data.append(sample)

  if len(plots) > 0:
    plt.figure()

  for i,(x,y) in enumerate(plots):
    # make two arrays for x and y axis
    xdata = [sample[x] for sample in data   ]
    ydata = [sample[y] for sample in data]

    plt.subplot(rows, cols, i+1)
    plt.plot(xdata,ydata)
    plt.xlabel(x)
    plt.ylabel(y)

