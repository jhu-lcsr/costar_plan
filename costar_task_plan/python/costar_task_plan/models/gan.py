
class GAN(object):
    def __init__(self, generator, discriminator, batch_size):
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size

    def summary(self):
        print generator.summary()
        print discriminator.summary()

    def fit(self, x, y, iter=1000, save_interval=0):
        #idx = np.random.randint(0,
        #    self.x_train.shape[0], size=self.batch_size)
        #x = self.x_train[idx]
        #y = self.y_train[idx]
        pass

class SimpleGAN(GAN):
    '''
    Feed forward network -- good for simple data sets.
    '''
    
    def __init__(self, input_shape, output_shape):
        pass

class SimpleImageGan(GAN):
    '''
    This is designed specifically for MNIST, but could be used in principle
    with plenty of other data sets.
    '''

    def __init__(self, img_rows=28, img_cols=28, channel=1):
        pass

class SimpleLSTMGAN(GAN):
    '''
    This is the one we want to use for trajectory generation.
    '''
    pass


class ImageLSTMGAN(GAN):
    '''
    Generate sequences of images and symbols.
    '''
    pass
