#!/usr/bin/env python

import os
import random 

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Input, merge
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD
from keras.datasets import mnist
from keras import backend as K

# Many of these functions are modified from @osh's implementation of a
# deep convolutional generative adversarial network and can be viewed
# here: https://github.com/osh/KerasGAN

def plot_mnist(vector):
    """
    Expects a numpy array input of length 748
    """
    vector = vector.reshape((28, 28))
    plt.imshow(vector, cmap='gray')
    plt.show()

def plot_generated(z_input, generator_model, examples=9,
                   custom_input=False, plot_dim=(3,3), size=(10,10)):
    """
    Draws randomly from a {0,1} uniform distribution, uses the
    generative model to build predictions from the input noise
    and plots the generated examples
    """
    if custom_input:
        noise = z_input
    else:
        noise = np.random.uniform(0, 1, size=[examples, z_input])
    
    generated_images = generator_model.predict(noise)

    fig = plt.figure(figsize=size)
    for i in range(examples):
        plt.subplot(plot_dim[0], plot_dim[1], i+1)
        img = generated_images[i, :]
        img = img.reshape((28, 28))
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
        
    return fig

def plot_metrics(metrics):
    """
    Plots loss and F score over training epochs
    """
    plt.figure(figsize=(10,8))
    plt.plot(metrics["d"], label='discriminitive loss')
    plt.plot(metrics["g"], label='generative loss')
    plt.plot(metrics["f"], label='F score')
    plt.legend()
    plt.show()

def generative_adversarial_network(generator_model, discriminator_model):
    """
    Compiling a generator and discriminator Sequential() models to build the
    generative adversarial framework
    """
    gan_model = Sequential()
    gan_model.add(generator_model)
    discriminator_model.trainable = False
    gan_model.add(discriminator_model)
    return gan_model

def train_for_n(z_input, generator_model, discriminator_model, gan_model,
                loss_dict, X_train, z_training_figures=None, z_group=None,
                z_plot_freq=100, visualize_train=False, epoch=500,
                plot_freq=25, batch=10):
    """
    Trains the GAN over input epochs according to batch sizes and will
    periodically display the training performance.
    
    Arguments:
    z_input - an int corresponding to the length of the random noise vector
    generator_model - a Sequential() Keras model with output dim 784
    discriminator_model - a Sequential() Keras model with output dim 1
    gan_model - combined Sequential() model of the generator and discriminator
    loss_dict - a dictionary that stores the results of each training epoch
    X_train - the training data X matrix
    plot_freq - after how many epochs a performance plot is displayed
    batch - how many samples input into training on each epoch
    
    Output:
    Trains the discriminator and generator simultaneously with the
    generator optimized to generate fake samples
    """
    with tqdm(total=epoch) as pbar:
        for e in range(epoch):  
            pbar.update(1)

            # Make random generative images   
            noise = np.random.uniform(0, 1, size=[batch, z_input])
            generated_images = generator_model.predict_on_batch(noise)
            
            # Generate consistent z vectors to visualize training process
            if visualize_train and e < 500:
                if e%z_plot_freq == z_plot_freq-1:
                    fig = plot_generated(z_input=z_group, custom_input=visualize_train,
                                         generator_model=generator_model)
                    z_training_figures.append(fig)
            
            # Subset random batch of training data
            rand_train_index = np.random.randint(0, X_train.shape[0], size=batch)
            image_batch = X_train[rand_train_index, :] 

            # Combine generated images with training data
            X = np.concatenate((image_batch, generated_images))
            y = np.zeros(int(2*batch))
            y[batch:] = 1  # Fake images get 1, real images get 0
            y = y.astype(int)

            # Train the discriminator to correctly detect fake images from real
            discriminator.trainable = True
            d_loss, d_f_score = discriminator_model.train_on_batch(x=X, y=y)
            discriminator.trainable = False

            # Coerce generator to try to make real samples
            noise = np.random.uniform(0, 1, size=[batch, z_input])
            y = np.zeros(batch)
            y = y.astype(int)
            g_loss = gan_model.train_on_batch(x=noise, y=y)

            loss_dict["d"].append(d_loss)
            loss_dict["g"].append(g_loss)
            loss_dict["f"].append(d_f_score) 

            # Update plots
            if e%plot_freq == plot_freq-1:
                plot_metrics(loss_dict)
                plot_generated(z_input=z_input, generator_model=generator_model)

random.seed(123)

# Define constants
z_input_vector = 100
n_train_samples = 50000
z_plot_freq = 250
epoch = 6000
plot_freq = 500
batch = 300

generator_optimizer = SGD(lr=0.1, momentum=0.3, decay=1e-5)
discriminator_optimizer = SGD(lr=0.1, momentum=0.1, decay=1e-5)
gan_optimizer = SGD(lr=0.1, momentum=0.3)




# Construct the Generator
# Mini-batch normalization (Ioffe and Szegedy 2015) is essential in this step  because it
# prevents the generator from collapsing into a unviversal output and allows for faster
# training (see Salimans et al. 2016). The model is a traditional MLP GAN similarly
# proposed by Goodfellow et al. 2014
generator = Sequential()
generator.add(Dense(input_dim=100, output_dim=1600, init='glorot_uniform'))
generator.add(BatchNormalization(mode=0))
generator.add(LeakyReLU(alpha=0.3))
generator.add(Dense(1200, init='glorot_uniform'))
generator.add(BatchNormalization(mode=0))
generator.add(LeakyReLU(alpha=0.3))
generator.add(Dense(1000, init='glorot_uniform'))
generator.add(BatchNormalization(mode=0))
generator.add(LeakyReLU(alpha=0.3))
generator.add(Dense(784, init='glorot_uniform', activation='sigmoid'))
generator.compile(loss='binary_crossentropy', optimizer=generator_optimizer)
generator.summary()

metrics = ['accuracy']

# Discriminator Model
discriminator = Sequential()
discriminator.add(Dense(input_dim=784, output_dim=240, init='glorot_uniform'))
discriminator.add(LeakyReLU(alpha=0.3))
discriminator.add(Dense(output_dim=240, init='glorot_uniform'))
discriminator.add(LeakyReLU(alpha=0.1))
discriminator.add(Dense(output_dim=240, init='glorot_uniform', activation='relu'))
discriminator.add(Dropout(0.2))
discriminator.add(Dense(1, init='glorot_uniform'))
discriminator.add(Activation('sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer,
                      metrics=metrics)
discriminator.summary()

# Discriminator Model
discriminator = Sequential()
discriminator.add(Dense(input_dim=784, output_dim=240, init='glorot_uniform'))
discriminator.add(LeakyReLU(alpha=0.3))
discriminator.add(Dense(output_dim=240, init='glorot_uniform'))
discriminator.add(LeakyReLU(alpha=0.1))
discriminator.add(Dense(output_dim=240, init='glorot_uniform', activation='relu'))
discriminator.add(Dropout(0.2))
discriminator.add(Dense(1, init='glorot_uniform'))
discriminator.add(Activation('sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer,
                      metrics=metrics)
discriminator.summary()

# Build the gan framework
gan = generative_adversarial_network(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=gan_optimizer)
gan.summary()

# Load the MNIST data and plot a couple examples
(X_train, y_train), (X_test, y_test) = mnist.load_data()

random_mnist_idx = random.sample(range(0,X_train.shape[0]), 9)

plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    img = X_train[random_mnist_idx[i], :].reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.axis('off')

try:
    os.mkdir('figures')
except Exception, e:
    pass
plt.savefig(os.path.join('figures', 'random_MNIST_examples.png'))


# Process the input data
X_train = X_train.reshape(X_train.shape[0], 28*28)
X_test = X_test.reshape(X_test.shape[0], 28*28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

train_index = random.sample(range(0, X_train.shape[0]), n_train_samples)
X_real = X_train[train_index,:]
noise_gen = np.random.uniform(0, 1, size=[n_train_samples, z_input_vector])
noise_gen.shape

# Create a z matrix to track throughout training process
z_group_matrix = np.random.uniform(0, 1, size=[9, z_input_vector])
z_training_figures = []

initial_generated_images = generator.predict_on_batch(noise_gen)

# Example randomly generated images before training
plt.figure(figsize=(10, 10))
for i in range(0, 9):
    plt.subplot(3, 3, i+1)
    x = initial_generated_images[i].reshape((28, 28))
    plt.imshow(x, cmap='gray')
plt.savefig(os.path.join('figures', 'random_z_vectors.png'))

# Combine real and fake data to pretrain the classifier
X = np.concatenate((X_real, initial_generated_images))
n = X_real.shape[0]
y = np.zeros(int(2*n))
y[n:] = 1
y = y.astype(int)

discriminator.trainable = True
discriminator.fit(X, y, nb_epoch=1, batch_size=300)

# Determine the accuracy of the model
y_hat = discriminator.predict_on_batch(X)
accuracy = (2*n - np.sum(np.abs(y - y_hat.round().transpose()))) / (2*n)
print('Accuracy = {}'.format(accuracy))

# Train generative adversarial net for 2,000 epochs with 300 batch size
# once the model trains with 2000 epochs, lower the learning rate
gen_losses = {"d":[], "g":[], "f":[]}
train_for_n(z_input=z_input_vector, generator_model=generator,
            discriminator_model=discriminator, gan_model=gan,
            z_training_figures=z_training_figures,
            z_group=z_group_matrix,
            z_plot_freq=z_plot_freq, visualize_train=True,
            loss_dict=gen_losses, X_train = X_real,
            epoch=epoch, plot_freq=plot_freq, batch=batch)

for im in range(len(z_training_figures)):
    z_training_figures[im].savefig('figures/training/training_gif_{}_.png'.format(im))
