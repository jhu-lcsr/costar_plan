
import numpy as np

def SplitIntoChunks(datasets, labels,
        chunk_length=100,
        forward_and_back=True,
        step_size=10):
    '''
    Split data into segments of the given length. This will return a much
    larger data set, but with the dimensionality changed so that we can easily
    look at certain sections.
    '''

    data_size = datasets[0].shape[0]

    new_data = []
    for data in datasets:
        i = 0
        dataset = []
        while i + chunk_length < data_size:
            block_labels = labels[i:i+chunk_length]
            if not np.all(block_labels == block_labels[0]):
                break
            dataset.append(data[i:i+chunk_length]) 
            i += step_size
        if forward_and_back:
            i = data_size
            while i - step_size >= 0:
                block_labels = labels[i-chunk_length:i]
                if not np.all(block_labels == block_labels[0]):
                    break
                dataset.append(data[i-chunk_length:i]) 
                i -= chunk_length
        new_data.append(np.array(dataset))
    for data in new_data:
        print data.shape
    return new_data

def FirstInChunk(data):
    '''
    Take the first thing in each chunk and return a dataset of just these.
    '''
    return data[:,0]

def LastInChunk(data):
    '''
    Take the last thing in each chunk and just return these.
    '''
    return data[:,-1]
