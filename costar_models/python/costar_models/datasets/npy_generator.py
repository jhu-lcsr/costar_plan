from __future__ import print_function

import numpy as np
import os

class NpzGeneratorDataset(object):
    '''
    Get the list of objects from a folder full of NP arrays. 
    '''

    def __init__(self, name, split=0.1, ):
        '''
        Set name of directory
        '''
        self.name = name 
        self.split = split
        self.train = []
        self.test = []

    def write(self, *args, **kwargs):
        raise NotImplementedError('this dataset does not save things')


    def load(self, success_only=False):
        '''
        Read the file; get the list of acceptable entries; split into train and
        test sets.
        '''
        files = os.listdir(self.name)
        files.sort()
        sample = {}
        i = 0
        acceptable_files = []
        for f in files:
            if not f[0] == '.':
                #print("%d:"%(i+1), f)
                if success_only:
                    name = f.split('.')
                    if name[1] == 'failure':
                        continue
                if i < 2:
                    fsample = np.load(os.path.join(self.name,f))
                    for key, value in fsample.items():
                        if key not in sample:
                            sample[key] = value
                        if value.shape[0] == 0:
                            continue
                        sample[key] = np.concatenate([sample[key],value],axis=0)
                i += 1
                acceptable_files.append(f)

        idx = np.array(range(len(acceptable_files)))
        length = max(1,int(self.split*len(acceptable_files)))
        print("---------------------------------------------")
        print("Loaded data.")
        print("# Total examples:", len(acceptable_files))
        print("# Validation examples:",length)
        print("---------------------------------------------")
        self.test = [acceptable_files[i] for i in idx[:length]]
        self.train = [acceptable_files[i] for i in idx[length:]]
        for i, filename in enumerate(self.test):
            print("%d:"%(i+1), filename)
            if filename in self.train:
                raise RuntimeError('error with test/train setup! ' + \
                                   filename + ' in training!')
        np.random.shuffle(self.test)
        np.random.shuffle(self.train)
        return sample

    def sampleTrainFilename(self):
        return os.path.join(self.name,
                self.train[np.random.randint(len(self.train))])

    def sampleTestFilename(self):
        return os.path.join(self.name,
                self.test[np.random.randint(len(self.test))])

    def testFiles(self):
        return self.test

    def trainFiles(self):
        return self.train

    def numTest(self):
        return len(self.test)

    def loadTest(self, i):
        if i > len(self.test):
            raise RuntimeError('index %d greater than number of files'%i)
        filename = self.test[i]
        success = 'success' in filename
        return np.load(os.path.join(self.name,filename)), success

    def sampleTrain(self):
        filename = self.sampleTrainFilename()
        try:
            sample = np.load(filename)
        except Exception as e:
            raise RuntimeError("Could not load file " + filename + ": " + str(e))
        return sample, filename

    def sampleTest(self):
        filename = self.sampleTestFilename()
        sample = np.load(filename)
        return sample, filename
