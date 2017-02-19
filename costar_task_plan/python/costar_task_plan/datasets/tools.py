
import numpy as np

'''
Simple function: figure out how often different action IDs are showing up in
our data. I expect to see that certain ids show up far more often than
others.
'''
def get_target_frequency(targets, num_targets):

    target_frequencies = [0] * num_targets

    for ex in targets:
        for a in ex:
            idx = int(a)
            target_frequencies[idx] = target_frequencies[idx] + 1

    return target_frequencies


def rebalance_dataset(data, targets):
    target_frequencies = get_target_frequency(targets)

def combine(x):
  nx = []
  for xx in x:
    nx = nx + xx.tolist()

  return np.array(nx)

def to_recurrent_samples(seq, window_length):
  data = np.zeros((len(seq) - window_length + 1, window_length, seq[0].shape[0]))
  for i in xrange(len(seq) - window_length + 1):
    for j in xrange(i,i+window_length):
      data[i,j-i,:] = seq[j]
  return data

