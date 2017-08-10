
import numpy as np

def SplitIntoChunks(datasets, labels,
        chunk_length=100,
        step_size=10,
        front_padding=False,
        rear_padding=False,
        stagger=False):
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
    padding = front_padding or rear_padding

    new_data = {}
    stagger_data = {}

    for label in xrange(min_label, max_label+1):
        for idx, data in enumerate(datasets):
            subset = data[labels==label]

            #print "SUBSET":,
            #print np.argmax(subset,axis=-1), subset.shape

            dataset = []
            # staggered dataset for dynamics learning
            sdataset = []


            # Set up data size
            data_size = subset.shape[0]
            if data_size == 0:
                continue
    
            # padding: add entries to the front or back
            if front_padding:
                i = 0
            else:
                i = chunk_length
            if rear_padding:
                max_i = data_size + chunk_length - 1
            else:
                max_i = data_size

            while i < max_i:
                start_block = max(0,i-chunk_length+1)
                end_block = min(i,data_size)
                block = subset[start_block:end_block+1]
                #print i, start_block, end_block+1, "/", data_size,
                #print subset.shape, np.argmax(block[-1])
                if padding:
                    block = AddPadding(block,
                            chunk_length,
                            start_block,
                            end_block,
                            data_size)
                elif (end_block + 1) - start_block is not chunk_length:
                    i += step_size
                    continue
                if not block.shape[0] == chunk_length:
                    print "block shape/chunk length:", block.shape, chunk_length
                    raise RuntimeError('dev error: block not of the ' + \
                                       'correct length.')
                dataset.append(block)

                if stagger:
                    #print "D> start/end:", start_block, end_block
                    start_block = max(0,i-chunk_length+2)
                    end_block = min(i+1,data_size)
                    #print "S> start/end:", start_block, end_block
                    # Get the next state info for learning dynamics models.
                    sblock = data[start_block:end_block+1]
                    if padding:
                        sblock = AddPadding(sblock,
                                chunk_length,
                                start_block,
                                end_block,
                                data_size)
                    elif end_block - start_block is not chunk_length:
                        raise RuntimeError('could not create staggered block')
                    assert sblock.shape[0] == chunk_length
                    sdataset.append(sblock)


                i += step_size

            if not idx in new_data:
                new_data[idx] = np.array(dataset)
                if stagger:
                    stagger_data[idx] = np.array(sdataset)
            else:
                new_data[idx] = np.append(
                        new_data[idx],
                        values=np.array(dataset),
                        axis = 0)
                if stagger:
                    stagger_data[idx] = np.append(
                            stagger_data[idx],
                            values=np.array(sdataset),
                            axis = 0)
    #print len(new_data)
    #print type(new_data)
    #for d in new_data.values():
    #    print d.shape
    for d in new_data.values():
        if not d.shape[0] == new_data.values()[0].shape[0]:
            raise RuntimeError('error combining datasets')

    if stagger:
        stagger_values = stagger_data.values()
    else:
        stagger_values = None
    return new_data.values(), stagger_values

def SplitIntoActions(
        datasets,
        example_labels,
        action_labels):
    '''
    Split based on when the high-level made decisions. We start out with an
    explicit representation of our immediate goals and our initial state, so we
    can anticipate when we will be done.

    Parameters:
    -----------
    datasets: observed features, actions, etc. that should be split according
              to example and action labels.
    example_labels: of same length as all entries in datasets; this is the
                    trial number that each sequence belongs to.
    action_labels: high-level action label that this belongs to

    Returns:
    --------
    '''

    min_example = np.min(example_labels)
    max_example = np.max(example_labels)
    min_action = np.min(action_labels)
    max_action = np.max(action_labels)

    # stores a list of all the END RESULTS of actions, in order, by the example
    # in which they took place.
    changepoints_by_example = {}

    # take out each example
    for example in xrange(min_example,max_example+1):

        # Just some simple setup
        changepoints_by_example[example] = []

        start = 0

        # pull out just the action labels for this example
        subset = action_labels[example_labels==example]

        # iterate over the length of the example to pull out start, end
        # indices for each action
        for i in xrange(len(subset)):

            print example, i, subset[i]
            last_i = len(subset) - 1

            # come up with the set of decision points
            if i == 0 or not subset[i-1] == subset[i] or i == last_i:

                print "action changes:", subset[i]

                if i == last_i or not subset[i-1] == subset[i]:
                    # add the subset because we found an end
                    changepoints_by_example[example].append((start,i))
                    start = i
    
    frame_data, result_data = [], []
    for example, actions in changepoints_by_example.items():
        for start_idx, end_idx in actions:
            for i in xrange(start_idx, end_idx):
                # create the data to train the sequence predictor.
                pass

    return frame_data, result_data


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
    if start_block == 0:
        entry = data[0]
        for _ in xrange(chunk_length - data.shape[0]):
            data = np.insert(data,0,axis=0,values=entry)
    elif end_block == data_size:
        entry = np.expand_dims(data[-1],axis=0)
        #if len(data.shape) < 3:
        #    print data
        for _ in xrange(chunk_length - data.shape[0]):
            data = np.append(data,axis=0,values=entry)
        #if len(data.shape) < 3:
        #    print data
    
    return data
