from __future__ import print_function

import os
import keras
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_MODEL_DIRECTORY = os.path.expanduser('~/.costar/models')


class LogCallback(keras.callbacks.Callback):
    def __init__(self,
            name="model",
            model_directory=DEFAULT_MODEL_DIRECTORY):
        self.directory = model_directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.file = open(os.path.join(self.directory,"%s_log.csv"%name),'w')

    def on_epoch_end(self, epoch, logs={}):
        print(epoch,logs)
        if epoch == 0:
            msg = ""
            for i, key in enumerate(logs.keys()):
                msg += str(key)
                if i < len(logs.keys())-1:
                    msg += ","
            msg += "\n"
            self.file.write(msg)
            self.file.flush()

        msg = ""
        for i, (key, value) in enumerate(logs.items()):
            msg += str(value)
            if i < len(logs.keys())-1:
                msg += ","
        msg += "\n"
        self.file.write(msg)
        self.file.flush()

class PredictorShowImage(keras.callbacks.Callback):
    '''
    Save an image showing what some number of frames and associated predictions
    will look like at the end of an epoch.
    '''

    variables = ["x","y","z","roll","pitch","yaw","gripper"]

    def __init__(self, predictor, features, targets,
            model_directory=DEFAULT_MODEL_DIRECTORY,
            name="model",
            num_hypotheses=4,
            verbose=False,
            use_prev_option=True,
            noise_dim=64,
            use_noise=False,
            min_idx=0, max_idx=66, step=11):
        '''
        Set up a data set we can use to output validation images.

        Parameters:
        -----------
        predictor: model used to generate predictions
        targets: training target info, in compressed form
        num_hypotheses: how many outputs to expect
        verbose: print out extra information
        '''
        self.verbose = verbose
        self.use_prev_option = use_prev_option
        self.predictor = predictor
        self.idxs = range(min_idx, max_idx, step)
        self.num = len(self.idxs)
        self.features = [f[self.idxs] for f in features]
        self.targets = [np.squeeze(t[self.idxs]) for t in targets]
        self.epoch = 0
        self.num_hypotheses = num_hypotheses
        self.directory = os.path.join(model_directory,'debug')
        self.noise_dim = noise_dim
        self.use_noise = use_noise
        self.files=[]
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.header = "V(h),max(p(A|H)),o_{i-1}"
        for i in range(self.num_hypotheses):
            for var in self.variables:
                self.header += ",%s%d"%(var,i)
        for var in self.variables:
            self.header += ",observed_%s"%(var)
        for i in range(self.num):
            self.files.append(open(os.path.join(model_directory,"%s_poses_log%d.csv"%(name,i)),'w'))
            self.files[-1].write(self.header+"\n")
            self.files[-1].flush()

    def on_epoch_end(self, epoch, logs={}):
        # take the model and print it out
        self.epoch += 1
        imglen = 64*64*3
        if len(self.targets[0].shape) == 2:
            img = self.targets[0][:,:imglen]
        elif len(self.targets[0].shape) == 3:
            assert self.targets[0].shape[1] == 1
            img = self.targets[0][:,0,:imglen]
        else:
            raise RuntimeError('did not recognize big train target shape; '
                               'are you sure you meant to use this callback'
                               'and not a normal image callback?')
        img = np.reshape(img, (self.num,64,64,3))
        if self.use_noise:
            z= np.random.random((self.targets[0].shape[0], self.num_hypotheses, self.noise_dim))
            data, arms, grippers, label, probs, v = self.predictor.predict(self.features + [z])
        else:
            data, arms, grippers, label, probs, v = self.predictor.predict(self.features)
        plt.ioff()
        if self.verbose:
            print("============================")
        for j in range(self.num):
            msg = ''
            name = os.path.join(self.directory,
                    "predictor_epoch%d_result%d.png"%(self.epoch,j))
            if self.verbose:
                print("----------------")
                print(name)
                print("max(p(o' | x)) =", np.argmax(probs[j]))
                print("v(x) =", v[j])
                print("o_{i-1} = ", self.features[3][j])
                msg += "%f,%d,%d"%(v[j],np.argmax(probs[j]),self.features[3][j])
            fig = plt.figure(figsize=(3+int(1.5*self.num_hypotheses),2))
            plt.subplot(1,2+self.num_hypotheses,1)
            plt.title('Input Image')
            plt.imshow(self.features[0][j])
            plt.subplot(1,2+self.num_hypotheses,2+self.num_hypotheses)
            plt.title('Observed Goal')
            plt.imshow(img[j])
            arm_target = self.targets[0][j,imglen:imglen+6]
            gripper_target = self.targets[0][j,imglen+6]
            for i in range(self.num_hypotheses):
                if self.verbose:
                    print("Arms = ", arms[j][i])
                    print("Gripper = ", grippers[j][i])
                    print("Label = ", np.argmax(label[j][i]))
                for q, q0 in zip(arms[j][i],arm_target):
                    msg += ",%f"%(q-q0)
                msg += ",%f"%(grippers[j][i][0]-gripper_target)
                plt.subplot(1,2+self.num_hypotheses,i+2)
                plt.imshow(np.squeeze(data[j][i]))
                plt.title('Hypothesis %d'%(i+1))
            fig.savefig(name, bbox_inches="tight")
            for q0 in arm_target:
                msg += ",%f"%q0
            msg += ",%f"%gripper_target
            self.files[j].write(msg+"\n")
            self.files[j].flush()
            if self.verbose:
                print("Arm/gripper target =",
                        self.targets[0][j,imglen:imglen+7])
                print("Label target =",
                        np.argmax(self.targets[0][j,(imglen+7):]))
                print("Label target 2 =", np.argmax(self.targets[1][j]))
                print("Value target =", np.argmax(self.targets[2][j]))
            plt.close(fig)


class ImageCb(keras.callbacks.Callback):
    '''
    Save an image showing what some number of frames and associated predictions
    will look like at the end of an epoch. This will only show the input,
    target, and predicted target image.
    '''

    def __init__(self, predictor, features, targets,
            model_directory=DEFAULT_MODEL_DIRECTORY,
            name="model",
            min_idx=0, max_idx=66, step=11,
            *args, **kwargs):
        '''
        Set up a data set we can use to output validation images.

        Parameters:
        -----------
        predictor: model used to generate predictions (can be different from
                   the model being trained)
        targets: training target info, in compressed form
        verbose: print out extra information
        '''
        self.predictor = predictor
        self.idxs = range(min_idx, max_idx, step)
        self.num = len(self.idxs)
        self.features = features[0][self.idxs]
        self.targets = [np.squeeze(t[self.idxs]) for t in targets]
        self.epoch = 0
        self.directory = os.path.join(model_directory,'debug')
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        img = self.predictor.predict(self.features)
        for j in range(self.num):
            name = os.path.join(self.directory,
                    "image_ae_epoch%d_result%d.png"%(self.epoch,j))
            fig = plt.figure()
            plt.subplot(1,3,1)
            plt.title('Input Image')
            plt.imshow(self.features[j])
            plt.subplot(1,3,3)
            plt.title('Observed Goal')
            plt.imshow(self.targets[0][j])
            plt.subplot(1,3,2)
            plt.imshow(np.squeeze(img[j]))
            plt.title('Output')
            fig.savefig(name, bbox_inches="tight")
            plt.close(fig)

class PredictorShowImageOnly(keras.callbacks.Callback):
    '''
    Save an image showing what some number of frames and associated predictions
    will look like at the end of an epoch.
    '''

    def __init__(self, predictor, features, targets,
            model_directory=DEFAULT_MODEL_DIRECTORY,
            num_hypotheses=4,
            verbose=False,
            noise_dim=64,
            use_noise=False,
            name="model",
            use_prev_option=True,
            min_idx=0, max_idx=66, step=11):
        '''
        Set up a data set we can use to output validation images.

        Parameters:
        -----------
        predictor: model used to generate predictions
        targets: training target info, in compressed form
        num_hypotheses: how many outputs to expect
        verbose: print out extra information
        '''
        self.verbose = verbose
        self.predictor = predictor
        self.idxs = range(min_idx, max_idx, step)
        self.num = len(self.idxs)
        self.features = [f[self.idxs] for f in features]
        self.targets = [np.squeeze(t[self.idxs]) for t in targets]
        self.epoch = 0
        self.num_hypotheses = num_hypotheses
        self.directory = os.path.join(model_directory,'debug')
        self.noise_dim = noise_dim
        self.use_noise = use_noise
        self.num_random = 3
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def on_epoch_end(self, epoch, logs={}):
        # take the model and print it out
        self.epoch += 1
        imglen = 64*64*3
        if len(self.targets[0].shape) == 2:
            img = self.targets[0][:,:imglen]
        elif len(self.targets[0].shape) == 3:
            assert self.targets[0].shape[1] == 1
            img = self.targets[0][:,0,:imglen]
        else:
            raise RuntimeError('did not recognize big train target shape; '
                               'are you sure you meant to use this callback'
                               'and not a normal image callback?')
        img = np.reshape(img, (self.num,64,64,3))
        data = [0] * self.num_random
        if self.use_noise:
            for k in range(self.num_random):
                z= np.random.random((self.targets[0].shape[0], self.num_hypotheses, self.noise_dim))
                data[k] = self.predictor.predict(self.features + [z])
        else:
            for k in range(self.num_random):
                data[k] = self.predictor.predict(self.features)
        plt.ioff()
        if self.verbose:
            print("============================")
        for j in range(self.num):
            name = os.path.join(self.directory,
                    "image_predictor_epoch%d_result%d.png"%(self.epoch,j))
            fig = plt.figure()#figsize=(3+int(1.5*self.num_hypotheses),2))
            for k in range(self.num_random):
                rand_offset = (k*(2+self.num_hypotheses))
                plt.subplot(self.num_random,2+self.num_hypotheses,1+rand_offset)
                #print (self.num_random,2+self.num_hypotheses,1+rand_offset)
                plt.title('Input Image')
                plt.imshow(self.features[0][j])
                plt.subplot(self.num_random,2+self.num_hypotheses,2+self.num_hypotheses+rand_offset)
                #print(self.num_random,2+self.num_hypotheses,2+self.num_hypotheses+rand_offset)
                plt.title('Observed Goal')
                plt.imshow(img[j])
                for i in range(self.num_hypotheses):
                    plt.subplot(self.num_random,2+self.num_hypotheses,i+2+rand_offset)
                    #print(self.num_random,2+self.num_hypotheses,2+i+rand_offset)
                    plt.imshow(np.squeeze(data[k][i]))
                    plt.title('Hypothesis %d'%(i+1))
            if self.verbose:
                print(name)
            fig.savefig(name, bbox_inches="tight")
            plt.close(fig)

class PredictorGoals(keras.callbacks.Callback):
    '''
    Save an image showing what some number of frames and associated predictions
    will look like at the end of an epoch.
    '''

    def __init__(self, predictor, features, targets,
            model_directory=DEFAULT_MODEL_DIRECTORY,
            num_hypotheses=4,
            verbose=False,
            use_prev_option=True,
            noise_dim=64,
            name="model",
            use_noise=False,
            min_idx=0, max_idx=66, step=11):
        '''
        Set up a data set we can use to output validation images.

        Parameters:
        -----------
        predictor: model used to generate predictions
        targets: training target info, in compressed form
        num_hypotheses: how many outputs to expect
        verbose: print out extra information
        '''
        self.verbose = verbose
        self.predictor = predictor
        self.idxs = range(min_idx, max_idx, step)
        self.num = len(self.idxs)
        self.features = [f[self.idxs] for f in features]
        self.targets = [np.squeeze(t[self.idxs]) for t in targets]
        self.epoch = 0
        self.num_hypotheses = num_hypotheses
        self.directory = os.path.join(model_directory,'debug')
        self.noise_dim = noise_dim
        self.use_noise = use_noise
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def on_epoch_end(self, epoch, logs={}):
        # take the model and print it out
        self.epoch += 1
        if self.use_noise:
            z= np.random.random((self.targets[0].shape[0], self.num_hypotheses, self.noise_dim))
            arms, grippers, label, probs, v = self.predictor.predict(
                    self.features[:4] + [z])
        else:
            arms, grippers, label, probs, v = self.predictor.predict(
                    self.features[:4])
        plt.ioff()
        if self.verbose:
            print("============================")
        for j in range(self.num):
            name = os.path.join(self.directory,
                    "predictor_epoch%d_result%d.png"%(self.epoch,j))
            if self.verbose:
                print("----------------")
                print(name)
                print("max(p(o' | x)) =", np.argmax(probs[j]))
                print("v(x) =", v[j])
            for i in range(self.num_hypotheses):
                if self.verbose:
                    print("Arms = ", arms[j][i])
                    print("Gripper = ", grippers[j][i])
                    print("Label = ", np.argmax(label[j][i]))
            if self.verbose:
                print("Arm/gripper target = ",
                        self.targets[0][j,:7])
                print("Label target = ",
                        np.argmax(self.targets[0][j,7:]))

