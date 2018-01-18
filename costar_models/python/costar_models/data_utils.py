from __future__ import print_function


from keras import backend as K
from keras.preprocessing.image import Iterator
from matplotlib import pyplot as plt

import numpy as np

def ToOneHot2D(f, dim):
    '''
    Convert all to one-hot vectors. If we have a "-1" label, example was
    considered unlabeled and should just get a zero...
    '''
    if len(f.shape) == 1:
        f = np.expand_dims(f, -1)
    assert len(f.shape) == 2
    shape = f.shape + (dim,)
    oh = np.zeros(shape)
    #oh[np.arange(f.shape[0]), np.arange(f.shape[1]), f]
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            idx = f[i,j]
            if idx >= 0:
                oh[i,j,idx] = 1.
    return oh

def MakeOption1h(option, num_labels):
    opt_1h = np.zeros((1,num_labels))
    opt_1h[0,option] = 1.
    return opt_1h

def GetNextGoal(imgs, labels):
    # Takes a list of target images and labels, and rotates them both
    # to the left by group
    imgs = np.copy(imgs)
    labels = np.copy(labels)
    irev = np.flip(imgs, axis=0)
    lrev = nd.flip(labels, axis=0)
    last_l = np.copy(lrev[0])
    last_i = np.copy(irev[0])
    write_l = last_l
    write_i = last_i
    # skip to relevant area
    j = 0
    while lrev[j] == last_l:
        j += 1
    for i in range(j, lrev.shape[0]):
        # check for switch
        if lrev[i] != last_l:
            write_l = last_l
            write_I = last_I
            last_l = np.copy(lrev[i])
            last_I = np.copy(Irev[i])
        lrev[i] = write_l
        Irev[i] = write_I
    return imgs, labels

"""
# Data generator that creates sequences for input into PredNet.
class SequenceGenerator(Iterator):
    def __init__(self, data_file, source_file, nt,
                 batch_size=8, shuffle=False, seed=None,
                 output_mode='error', sequence_start_mode='all', N_seq=None,
                 data_format=K.image_data_format()):
        self.X = hkl.load(data_file)  # X will be like (n_images, nb_cols, nb_rows, nb_channels)
        print (self.X.shape)
        self.X = np.expand_dims(self.X, axis=3)
        print ('new shape', self.X.shape)
        self.sources = hkl.load(source_file) # source for each image so when creating sequences can assure that consecutive frames are from same video
        self.nt = nt
        self.batch_size = batch_size
        self.data_format = data_format
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode

        if self.data_format == 'channels_first':
            self.X = np.transpose(self.X, (0, 3, 1, 2))
        self.im_shape = self.X[0].shape

        if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
            self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
        elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
            curr_location = 0
            possible_starts = []
            while curr_location < self.X.shape[0] - self.nt + 1:
                if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                    possible_starts.append(curr_location)
                    curr_location += self.nt
                else:
                    curr_location += 1
            self.possible_starts = possible_starts

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        if N_seq is not None and len(self.possible_starts) > N_seq:  # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)
        super(SequenceGenerator, self).__init__(len(self.possible_starts), batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = np.zeros((current_batch_size, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(index_array):
            idx = self.possible_starts[idx]
            batch_x[i] = self.preprocess(self.X[idx:idx+self.nt])
        if self.output_mode == 'error':  # model outputs errors, so y should be zeros
            batch_y = np.zeros(current_batch_size, np.float32)
        elif self.output_mode == 'prediction':  # output actual pixels
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        return X.astype(np.float32) / 255

    def create_all(self):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx:idx+self.nt])
        return X_all
"""
