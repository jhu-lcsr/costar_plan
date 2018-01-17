from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

class JigsawsDataset(object):
    '''
    Load videos and transcriptions from the jigsaws dataset
    '''

    def __init__(self, name):
        self.name = name 
        self.split = split
        self.train = []
        self.test = []
        self.files = {}

    def load(self, success_only=False):
        '''
        Read the file; get the list of acceptable entries; split into train and
        test sets.
        '''
        transcriptions = os.path.join(self.name, "transcriptions")
        files = os.listdir(transcriptions)
        files.sort()
        sample = {}
        i = 0
        acceptable_files = []
        for f in files:
            if not f[0] == '.':
                if success_only:
                    name = f.split('.')
                    if name[1] == 'failure':
                        continue
                if i < 2:
                    fsample = self._load(os.path.join(self.name,f))
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
            #print("%d:"%(i+1), filename)
            if filename in self.train:
                raise RuntimeError('error with test/train setup! ' + \
                                   filename + ' in training!')
        np.random.shuffle(self.test)
        np.random.shuffle(self.train)
        return sample

    def _load(self, filename):
        '''
        Helper to load the file
        '''
        filename_base = filename.split('.')[0]
        txt_file = os.path.join(self.name, "transcriptions", filename)
        with fin as file(txt_file, "r"):
            res = fin.readLine()
            print(res)
        avi_file = os.path.join(self.name, "video", filename_base,
                "_capture1.avi")
        vid = cv2.VideoCapture(avi_file)
        while vid.isOpened():
            ret, frame = vid.read()

            # Return this
            if not ret:
                break

            # Plot 
            plt.imshow(frame)
            plt.show()
            break

        f = h5f.File(filename, 'r')
        return f

