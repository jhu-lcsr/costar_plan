
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, RepeatVector, Reshape
from keras.layers import UpSampling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.layers import Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import Concatenate
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam

import keras.backend as K

def GetCameraColumn(img_shape, dim, dropout_rate, num_filters, dense_size):
    '''
    Convolutions for an image, terminating in a dense layer of size dim.
    '''
    height4 = img_shape[0]/4
    width4 = img_shape[1]/4
    height2 = img_shape[0]/2
    width2 = img_shape[1]/2
    width = img_shape[1]
    channels = img_shape[2]

    samples = Input(shape=img_shape)
    #x = Concatenate(axis=3)([samples, labels2])
    x = Conv2D(num_filters, # + num_labels
               kernel_size=[5, 5], 
               strides=(2, 2),
               #padding="same")(x)
               padding="same")(samples)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(dropout_rate)(x)

    # Add conv layer with more filters
    #labels2 = RepeatVector(height2*width2)(labels)
    #labels2 = Reshape((height2,width2,num_labels))(labels2)
    #x = Concatenate(axis=3)([x, labels2])
    x = Conv2D(num_filters, # + num_labels
               kernel_size=[5, 5], 
               strides=(2, 2),
               padding="same")(x)
    #x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(num_filters, # + num_labels
               kernel_size=[5, 5], 
               strides=(2, 2),
               padding="same")(x)
    #x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(num_filters, # + num_labels
               kernel_size=[5, 5], 
               strides=(2, 2),
               padding="same")(x)
    #x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(dropout_rate)(x)

    # Add dense layer
    x = Flatten()(x)
    #x = Concatenate(axis=1)([x, labels])
    x = Dense(int(0.5 * dense_size))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(dropout_rate)(x)

    # Single output -- sigmoid activation function
    #x = Concatenate(axis=1)([x, labels])
    x = Dense(dim)(x)
    x = LeakyReLU(alpha=0.2)(x)
    return [samples], x

def GetArmGripperColumns(arm, gripper, dim, dropout_rate, dense_size):
    '''
    Take arm and gripper as two separate inputs, process via dense layer.
    '''
    arm_in = Input((arm,))
    gripper_in = Input((gripper,))
    x = Concatenate()([arm_in, gripper_in])
    x = Dense(dense_size)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(dim)(x)
    x = LeakyReLU(alpha=0.2)(x)

    return [arm_in, gripper_in], x

def GetSeparateEncoder(img_shape, img_col_dim, dropout_rate, img_num_filters,
        img_dense_size, arm_size, gripper_size, robot_col_dim,
        robot_col_dense_size,
        combined_dense_size):

        img_ins, img_out = GetCameraColumn(
                img_shape,
                img_col_dim,
                dropout_rate,
                img_num_filters,
                img_dense_size,)
        robot_ins, robot_out = GetArmGripperColumns(
                arm_size, 
                gripper_size,
                robot_col_dim,
                dropout_rate,
                robot_col_dense_size,)

        x = Concatenate()([img_out, robot_out])
        x = Dense(combined_dense_size)(x)
        x = LeakyReLU(alpha=0.2)(x)

        return img_ins + robot_ins, x

def MakeStacked(ins, x, num_to_stack):
    '''
    Stacked latent representations -- for temporal convolutions in particular
    '''
    print TimeDistributed(x)
    new_ins = []
    new_xs = []
    x = Model(ins, x)
    for i in xrange(num_to_stack):
        new_x_ins = []
        for inx in ins:
            new_x_ins.append(Input(inx.shape[1:]))
        new_ins += new_x_ins
        new_xs.append(x(new_x_ins))
    x = Lambda(lambda x: K.stack(x,axis=2))(new_xs)

    return new_ins, x

def GetEncoder(img_shape, arm_size, gripper_size, dim, dropout_rate,
        filters, discriminator=False, tile=False,
        pre_tiling_layers=0,
        post_tiling_layers=2,
        time_distributed=0,):

    if time_distributed <= 0:
        ApplyTD = lambda x: x
        arm_in = Input((arm_size,))
        gripper_in = Input((gripper_size,))
        height4 = img_shape[0]/4
        width4 = img_shape[1]/4
        height2 = img_shape[0]/2
        width2 = img_shape[1]/2
        width = img_shape[1]
        channels = img_shape[2]
    else:
        ApplyTD = lambda x: TimeDistributed(x)
        arm_in = Input((time_distributed, arm_size,))
        gripper_in = Input((time_distributed, gripper_size,))
        height4 = img_shape[1]/4
        width4 = img_shape[2]/4
        height2 = img_shape[1]/2
        width2 = img_shape[2]/2
        width = img_shape[2]
        channels = img_shape[3]
    samples = Input(shape=img_shape)

    '''
    Convolutions for an image, terminating in a dense layer of size dim.
    '''

    x = samples

    x = ApplyTD(Conv2D(filters,
                kernel_size=[5, 5], 
                strides=(1, 1),
                padding='same'))(x)
    x = ApplyTD(Activation('relu'))(x)
    x = ApplyTD(Dropout(dropout_rate))(x)

    for i in xrange(pre_tiling_layers):

        x = ApplyTD(Conv2D(filters,
                   kernel_size=[5, 5], 
                   strides=(1, 1),
                   padding='same'))(x)
        x = ApplyTD(LeakyReLU(alpha=0.2))(x)
        x = ApplyTD(Dropout(dropout_rate))(x)

    # ===============================================
    # ADD TILING
    if tile:
        robot = Concatenate(axis=-1)([arm_in, gripper_in])
        if time_distributed > 0:
            tile_shape = (1, 1, width4, height4, 1)
            robot = Reshape([time_distributed, 1, 1, arm_size+gripper_size])(robot)
        else:
            tile_shape = (1, width2, height2, 1)
            robot = Reshape([1,1,arm_size+gripper_size])(robot)
        robot = Lambda(lambda x: K.tile(x, tile_shape))(robot)
        x = Concatenate(axis=-1)([x,robot])
        ins = [samples, arm_in, gripper_in]
    else:
        ins = [samples]

    for i in xrange(post_tiling_layers):
        x = ApplyTD(Conv2D(filters,
                   kernel_size=[5, 5], 
                   strides=(2, 2),
                   padding='same'))(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

    x = ApplyTD(Flatten())(x)
    x = ApplyTD(Dense(dim))(x)
    x = ApplyTD(LeakyReLU(alpha=0.2))(x)

    # Single output -- sigmoid activation function
    if discriminator:
        x = Dense(1,activation="sigmoid")(x)

    return ins, x

def GetDecoder(dim, img_shape, arm_size, gripper_size,
        dropout_rate, filters):

    '''
    Initial decoder: just based on getting images out of the world state
    created via the encoder.
    '''

    height8 = img_shape[0]/8
    width8 = img_shape[1]/8
    height4 = img_shape[0]/4
    width4 = img_shape[1]/4
    height2 = img_shape[0]/2
    width2 = img_shape[1]/2
    nchannels = img_shape[2]

    z = Input((dim,))

    x = Dense(filters/2 * height8 * width8)(z)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Reshape((width8,height8,filters/2))(x)
    x = Dropout(dropout_rate)(x)

    for i in xrange(3):
        x = Conv2DTranspose(filters,
                   kernel_size=[5, 5], 
                   strides=(2, 2),
                   padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

    for i in xrange(1):
        x = Conv2D(filters, # + num_labels
                   kernel_size=[5, 5], 
                   strides=(1, 1),
                   padding="same")(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout_rate)(x)

    x = Conv2D(nchannels, (1, 1), padding='same')(x)
    x = Activation('sigmoid')(x)

    return z, x

def GetTCNStack(x, filters, num_levels=2, dense_size=128, dropout_rate=0.5):
    '''
    Add some convolutions to a simple image
    '''

    for i in xrange(num_levels):
        x = Conv2D(filters,
                kernel_size=[5,5],
                strides=(2,2),
                padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout_rate)(x)
    x = Flatten()(x)
    x = Dense(dense_size)(x)

    return x

def GetInvCameraColumn(noise, img_shape, dropout_rate, dense_size):
    '''
    Take noise vector, upsample into an image of size img.
    '''
    pass

def GetInvArmGripperColumn(noise, arm, gripper, dropout_rate, dense_size):
    '''
    Get arm and gripper from noise.
    '''
    pass

def GetEncoder2(img_shape, arm_size, gripper_size, dim, dropout_rate,
        filters, discriminator=False, tile=False,
        pre_tiling_layers=0,
        post_tiling_layers=2):

    '''
    Convolutions for an image, terminating in a dense layer of size dim.
    '''
    height4 = img_shape[0]/4
    width4 = img_shape[1]/4
    height2 = img_shape[0]/2
    width2 = img_shape[1]/2
    width = img_shape[1]
    channels = img_shape[2]

    samples = Input(shape=img_shape)
    arm_in = Input((arm_size,))
    gripper_in = Input((gripper_size,))

    x = samples

    for i in xrange(pre_tiling_layers):
        x = Conv2D(filters,
                   kernel_size=[5, 5], 
                   strides=(2, 2),
                   padding='same')(x)
        #x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

    # ===============================================
    # ADD TILING
    if tile:
        tile_shape = (1, width2, height2, 1)
        robot = Concatenate()([arm_in, gripper_in])
        robot = Reshape([1,1,arm_size+gripper_size])(robot)
        robot = Lambda(lambda x: K.tile(x, tile_shape))(robot)
        x = Concatenate(axis=3)([x,robot])
        ins = [samples, arm_in, gripper_in]
    else:
        ins = [samples]

    for i in xrange(post_tiling_layers):
        x = Conv2D(filters,
                   kernel_size=[5, 5], 
                   strides=(2, 2),
                   padding='same')(x)
        #x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

    x = Flatten()(x)
    x = Dense(dim)(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Single output -- sigmoid activation function
    if discriminator:
        x = Dense(1,activation="sigmoid")(x)

    return ins, x

def GetDecoder2(dim, img_shape, arm_size, gripper_size,
        dropout_rate, filters):

    '''
    Initial decoder: just based on getting images out of the world state
    created via the encoder.
    '''

    height4 = img_shape[0]/4
    width4 = img_shape[1]/4
    height2 = img_shape[0]/2
    width2 = img_shape[1]/2
    nchannels = img_shape[2]

    z = Input((dim,))

    x = Dense(filters/2 * height2 * width2)(z)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Reshape((width2,height2,filters/2))(x)
    x = Dropout(dropout_rate)(x)

    for i in xrange(1):
        x = Conv2DTranspose(filters,
                   kernel_size=[5, 5], 
                   strides=(2, 2),
                   padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

    for i in xrange(1):
        x = Conv2D(filters/2, # + num_labels
                   kernel_size=[5, 5], 
                   strides=(1, 1),
                   padding="same")(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout_rate)(x)

    x = Conv2D(nchannels, (1, 1), padding='same')(x)
    x = Activation('sigmoid')(x)

    return z, x
