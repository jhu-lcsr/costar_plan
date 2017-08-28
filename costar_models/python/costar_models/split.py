from __future__ import print_function

import numpy as np

def SplitIntoChunks(datasets, labels,
        reward=None,
        reward_threshold=0,
        chunk_length=100,
        front_padding=False,
        rear_padding=False,):
    '''
    Split data into segments of the given length. This will return a much
    larger data set, but with the dimensionality changed so that we can easily
    look at certain sections.

    Parameters:
    -----------
    dataset: data to split
    labels: labels to split over
    reward: used to include only "good" examples from a planner
    reward_threshold: remove any examples from this
    step_size: how far to step between blocks (i.e. how much overlap is
               allowed)
    chunk_length: how long blocks are
    padding: should we duplicate beginning/ending frames to pad trials
    '''

    max_label = max(labels)
    min_label = min(labels)
    padding = front_padding or rear_padding

    if reward is not None:
        datasets.append(reward)

    new_data = []
    for data in datasets:
        shape = (data.shape[0], chunk_length,) + data.shape[1:]
        new_data.append(np.zeros(shape))
    next_idx = np.zeros((len(datasets),),dtype=int)

    for label in xrange(min_label, max_label+1):

        if sum(labels==label) == 0:
            continue

        # prune any rewards that are not acceptable here. we assume that we
        # care the most about the terminal reward -- if the terminal reward is
        # not greater than zero, we will throw out the example
        if reward is not None and reward[labels==label][-1] < reward_threshold:
            # Since this was too low, just skip it
            print("<<< EXCLUDING FAILED EXAMPLE = ", label)
            continue
        else:
            print(">>> INCLUDING EXAMPLE = ", label, "with reward =",
                    reward[labels==label][-1])

        for idx, data in enumerate(datasets):
            subset = data[labels==label]

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
                    print("block shape/chunk length:", block.shape,
                            chunk_length)
                    raise RuntimeError('dev error: block not of the ' + \
                                       'correct length.')

                new_data[idx][next_idx[idx]] = block
                next_idx[idx] += 1
                i += 1

    for d in new_data:
        if not d.shape[0] == new_data[0].shape[0]:
            raise RuntimeError('error combining datasets')

    return new_data

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

            print(example, i, subset[i])
            last_i = len(subset) - 1

            # come up with the set of decision points
            if i == 0 or not subset[i-1] == subset[i] or i == last_i:

                print("action changes:", subset[i])

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

def NextAction(datasets, action_labels, examples):
    '''
    Create extra datasets marking transitions between actions. This is so we
    can predict the effects of high-level actions, not just low level actions,
    when doing our various operations.

    Parameters:
    -----------
    datasets: list of data that needs to be updated
    action_labels: list of labels for actions (e.g. "PICKUP(OBJ)")
    examples: ID number for the sequence the data belongs to

    Returns:
    --------
    new_data: data of the same shape as datasets, but containing terminal
              states of all the labeled high-level actions
    '''

    # Loop over all entries in action labels and examples
    if len(action_labels.shape) == 1:
        action_labels = np.expand_dims(action_labels, -1)
    if len(examples.shape) == 1:
        examples = np.expand_dims(examples, -1)
    if not action_labels.shape == examples.shape:
        print(action_labels.shape)
        print(examples.shape)
        raise RuntimeError('all matrices must be of the same shape')
    elif len(action_labels.shape) is not 2:
        print(action_labels.shape)
        raise RuntimeError('all data should be of the shape ' + \
                           '(NUM_EXAMPLES, data)')
    for data in datasets:
        if not data.shape[0] == examples.shape[0]:
            print(data.shape, examples.shape)
            raise RuntimeError('all data must be of the same length')
    
    new_datasets = []
    for data in datasets:
        new_datasets.append(np.zeros_like(data))

    idx = 0 # idx of the data
    switch = 1
    while switch < examples.shape[0]:
        end_of_trial = False
        # loop over every entry; break if this is the last entry
        while switch < examples.shape[0] - 1:
            if examples[switch-1] == examples[switch] and \
               action_labels[switch-1] == action_labels[switch]:
                   switch += 1
                   continue
            elif not examples[switch-1] == examples[switch]:
                # We do not want to predict the beginning of the next trial!
                end_of_trial = True
                switch -= 1
                break
            else:
                break
        while idx < switch:
            for goal_data, data in zip(new_datasets, datasets):
                goal_data[idx] = data[switch]
            idx += 1
        if end_of_trial:
            idx += 1
        switch = idx+1

    # Set goal for the last frame, for completeness
    for goal_data, data in zip(new_datasets, datasets):
        goal_data[-1] = data[-1]

    return new_datasets

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
        for _ in xrange(chunk_length - data.shape[0]):
            data = np.append(data,axis=0,values=entry)
    
    return data
