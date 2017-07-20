
import numpy as np

def SplitIntoChunks(datasets, labels,
        chunk_length=100,
        step_size=10,
        padding=True):
    '''
    Split data into segments of the given length. This will return a much
    larger data set, but with the dimensionality changed so that we can easily
    look at certain sections.

    Parameters:
    -----------
    dataset: data to split
    labels: labels to split over
    step_size: how far to step between blocks (i.e. how much overlap is
               allowed)
    chunk_length: how long blocks are
    padding: should we duplicate beginning/ending frames to pad trials
    '''

    max_label = max(labels)
    min_label = min(labels)

    if padding:
        req_i = 0
    else:
        req_i = chunk_length

    new_data = {}
    for label in xrange(min_label, max_label+1):
        for idx, data in enumerate(datasets):
            subset = data[labels==label]
            dataset = []
            i = 1
            data_size = subset.shape[0]
            if data_size == 0:
                continue
            while i < data_size:
                start_block = max(0,i-chunk_length)
                end_block = min(i,data_size)
                i += step_size
                block = data[start_block:end_block]
                if padding:
                    block = AddPadding(block,
                            chunk_length,
                            start_block,
                            end_block,
                            data_size)
                elif end_block - start_block is not chunk_length:
                    continue
                assert block.shape[0] == chunk_length
                dataset.append(block)
            if not idx in new_data:
                new_data[idx] = np.array(dataset)
            else:
                new_data[idx] = np.append(
                        new_data[idx],
                        values=np.array(dataset),
                        axis = 0)
    #print len(new_data)
    #print type(new_data)
    #for d in new_data.values():
    #    print d.shape
    for d in new_data.values():
        if not d.shape[0] == new_data.values()[0].shape[0]:
            raise RuntimeError('error combining datasets')
    return new_data.values()

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

def AddPadding(data,chunk_length,start_block,end_block,data_size):
    #print data.shape
    if start_block == 0:
        entry = data[0]
        #if len(data.shape) < 3:
        #    print data
        for _ in xrange(chunk_length - data.shape[0]):
            data = np.insert(data,0,axis=0,values=entry)
        #if len(data.shape) < 3:
        #    print data
    elif end_block == data_size:
        entry = np.expand_dims(data[-1],axis=0)
        for _ in xrange(chunk_length - data.shape[0]):
            data = np.append(data,axis=0,values=entry)
    
    return data
