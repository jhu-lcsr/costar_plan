
class GAN(object):
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

    def summary(self):
        print generator.summary()
        print discriminator.summary()

class SimpleGAN(GAN):
    
    def __init__(self, input_shape, output_shape):
        pass


class SimpleImageGan(GAN):

    def __init__(self, input_shape, output_shape):
        pass

class SimpleLSTMGAN(GAN):
    pass


