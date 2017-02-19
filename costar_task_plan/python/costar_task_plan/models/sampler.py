

'''
Sampler Neural Net
This neural net replaces the Z(mu, sigma) distribution from which we draw
trajectories. It's an LSTM-RNN neural network. Input is the state of the sample
position, output is the 
'''
class SamplerNetwork(object):
  
  def __init__(self, feature_layers=[32,32], timesteps=5):
    # create a number of input layers
    for layer_size in feature_layers:
      pass


