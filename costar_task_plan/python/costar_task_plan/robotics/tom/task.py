
# Create the task model for the oranges task.
# This will create a number of oranges and instantiate all the appropriate
# DMP option distributions with pre and post condiitons, so that we can
# test things out in the toy sim.
def GetOrangesTask(num_oranges=1):

  if num_oranges < 1:
    raise RuntimeError('Must have at least one orange'
        'to be able to plan the TOM oranges task.')


    
