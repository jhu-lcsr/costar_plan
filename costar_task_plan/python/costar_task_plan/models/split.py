
import numpy as np

def SplitIntoChunks(datasets, labels,
        chunk_length=100,
        forward_and_back=True,
        step_size=10,
        padding=False):
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
            block = data[i:i+chunk_length]
            if not np.all(block_labels == block_labels[0]):
                break
            elif padding:
                block = AddPadding(block,block_labels)
            dataset.append(block)
            i += step_size
        if forward_and_back:
            i = data_size
            while i - step_size >= 0:
                block_labels = labels[i-chunk_length:i]
                block = data[i-chunk_length:i]
                if not np.all(block_labels == block_labels[0]):
                    break
                elif padding:
                    AddPadding(block,block_labels)
                dataset.append(block) 
                i -= chunk_length
        new_data.append(np.array(dataset))
    return new_data

def SplitIntoTemporalChunks(datasets, labels,
        chunk_length=100,
        step_size=10,
        padding=False):
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
            block = data[i:i+chunk_length]
            if not np.all(block_labels == block_labels[0]):
                break
            elif padding:
                block = AddPadding(block,block_labels)
            dataset.append(block)
            i += step_size
        new_data.append(np.array(dataset))
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

def AddPadding(data,labels):
    label = labels[0]
    seq = labels == label
    last_of_label = np.argmax(np.cumsum(seq))
    data[last_of_label+1:] = data[last_of_label]
    return data
