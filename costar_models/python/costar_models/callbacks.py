
import keras
import matplotlib.pyplot as plt

class PredictorShowImage(keras.callbacks.Callback):
    '''
    Save an image showing what some number of frames and associated predictions
    will look like at the end of an epoch.
    '''

    def __init__(self, predictor, features, targets, num_hypotheses=4, verbose=False, min_idx=0, max_idx=66, step=11):
        self.verbose = verbose
        self.predictor = predictor
        self.idxs = range(min_idx, max_idx, step)
        self.num = len(self.idxs)
        self.features = [f[idxs] for f in features]
        self.targets = [t[idxs] for t in targets]
        self.epoch = 0
        self.num_hypotheses = num_hypotheses

    def on_epoch_end(self, epoch, logs={}):
        # take the model and print it out
        self.epoch += 1
        imglen = 64*64*3
        img = allt[:,:imglen]
        img = np.reshape(img, (6,64,64,3))
        data, arms, grippers = self.predictor.predict(self.features)
        for j in range(self.num):
            name = "predictor_epoch%d_result%d.png"%(self.epoch,j)
            fig = plt.figure()
            plt.subplot(1,2+self.num_hypotheses,1)
            plt.title('Input Image')
            plt.imshow(features[0])
            plt.subplot(1,2+self.num_hypotheses,2+self.num_hypotheses)
            plt.title('Observed Goal')
            plt.imshow(targets[0])
            for i in range(self.num_hypotheses):
                plt.subplot(1,2+self.num_hypotheses,i)
                plt.imshow(np.squeeze(data[j][i]))
                plt.title('Hypothesis %d'%(i+1))
            plt.show()
            fig.savefig(name, bbox_inches="tight")

