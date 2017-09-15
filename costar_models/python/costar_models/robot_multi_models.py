from __future__ import print_function

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, RepeatVector, Reshape
from keras.layers import UpSampling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout
from keras.layers import Dense, Conv2D, Activation, Flatten, LSTM, ConvLSTM2D
from keras.layers import Lambda, Conv3D, MaxPooling2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import Concatenate
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam

import keras.backend as K

from .planner import *

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
    new_ins = []
    new_xs = []
    x = Model(ins, x)
    for i in range(num_to_stack):
        new_x_ins = []
        for inx in ins:
            new_x_ins.append(Input(inx.shape[1:]))
        new_ins += new_x_ins
        new_xs.append(x(new_x_ins))
    x = Lambda(lambda x: K.stack(x,axis=2))(new_xs)

    return new_ins, x

def GetEncoderConvLSTM(img_shape, arm_size, gripper_size,
        dropout_rate,
        filters,
        tile=False, dropout=True, leaky=True,
        pre_tiling_layers=0,
        post_tiling_layers=2,
        kernel_size=[3,3],
        time_distributed=10,):

    arm_in = Input((time_distributed, arm_size,))
    gripper_in = Input((time_distributed, gripper_size,))
    height4 = img_shape[1]/4
    width4 = img_shape[2]/4
    height2 = img_shape[1]/2
    width2 = img_shape[2]/2
    height = img_shape[1]
    width = img_shape[2]
    channels = img_shape[3]
    samples = Input(shape=img_shape)

    '''
    This is set up to use 3D convolutions to operate over a bunch of temporally
    grouped frames. The assumption is that this will allow us to capture
    temporal dependencies between actions better than we otherwise would be
    able to.
    '''

    if leaky:
        relu = lambda: LeakyReLU(alpha=0.2)
    else:
        relu = lambda: Activation('relu')

    x = samples

    for i in range(pre_tiling_layers):

        x = ConvLSTM2D(filters,
                   kernel_size=kernel_size, 
                   return_sequences=True,
                   strides=(2, 2),
                   padding='same')(x)
        x = relu()(x)
        if dropout:
            x = Dropout(dropout_rate)(x)

    # ===============================================
    # ADD TILING
    if tile:
        tile_width = int(width/(pre_tiling_layers+1))
        tile_height = int(height/(pre_tiling_layers+1))
        x = TileArmAndGripper(arm_in, gripper_in,
                tile_width, tile_height,
                time_distributed)
        ins = [samples, arm_in, gripper_in]
    else:
        ins = [samples]

    for i in range(post_tiling_layers):
        if i == post_tiling_layers - 1:
            ret_seq = False
        else:
            ret_seq = True
        x = ConvLSTM2D(filters,
                   kernel_size=kernel_size, 
                   return_sequences=ret_seq,
                   strides=(2, 2),
                   padding='same')(x)
        x = relu()(x)
        if dropout:
            x = Dropout(dropout_rate)(x)

    return ins, x



def GetEncoder3D(img_shape, arm_size, gripper_size, dropout_rate,
        filters, tile=False, dropout=True, leaky=True,
        pre_tiling_layers=0,
        post_tiling_layers=2,
        kernel_size=[3,3,3],
        time_distributed=10,):

    arm_in = Input((time_distributed, arm_size,))
    gripper_in = Input((time_distributed, gripper_size,))
    height4 = img_shape[1]/4
    width4 = img_shape[2]/4
    height2 = img_shape[1]/2
    width2 = img_shape[2]/2
    height = img_shape[1]
    width = img_shape[2]
    channels = img_shape[3]
    samples = Input(shape=img_shape)

    '''
    This is set up to use 3D convolutions to operate over a bunch of temporally
    grouped frames. The assumption is that this will allow us to capture
    temporal dependencies between actions better than we otherwise would be
    able to.
    '''

    if leaky:
        relu = lambda: LeakyReLU(alpha=0.2)
    else:
        relu = lambda: Activation('relu')

    x = samples

    for i in range(pre_tiling_layers):

        x = Conv3D(filters,
                   kernel_size=kernel_size, 
                   strides=(1, 2, 2),
                   padding='same')(x)
        x = relu()(x)
        if dropout:
            x = Dropout(dropout_rate)(x)

    # ===============================================
    # ADD TILING
    if tile:
        tile_width = int(width/(pre_tiling_layers+1))
        tile_height = int(height/(pre_tiling_layers+1))

        x = TileArmAndGripper(arm_in, gripper_in,
                tile_width, tile_height,
                time_distributed)
        ins = [samples, arm_in, gripper_in]
    else:
        ins = [samples]

    for i in range(post_tiling_layers):
        x = Conv3D(filters,
                   kernel_size=kernel_size, 
                   strides=(2, 2, 2),
                   padding='same')(x)
        x = relu()(x)
        if dropout:
            x = Dropout(dropout_rate)(x)

    return ins, x

def GetEncoder(img_shape, arm_size, gripper_size, dim, dropout_rate,
        filters, discriminator=False, tile=False, dropout=True, leaky=True,
        dense=True, option=None, flatten=True, batchnorm=False,
        pre_tiling_layers=0,
        post_tiling_layers=2,
        kernel_size=[3,3], output_filters=None,
        time_distributed=0,):


    if output_filters is None:
        output_filters = filters

    if time_distributed <= 0:
        ApplyTD = lambda x: x
        arm_in = Input((arm_size,))
        gripper_in = Input((gripper_size,))
        if option is not None:
            option_in = Input((1,))
            option_x = OneHot(size=option)(option_in)
            option_x = Reshape((option,))(option_x)
        else:
            option_in, option_x = None, None
        height4 = img_shape[0]/4
        width4 = img_shape[1]/4
        height2 = img_shape[0]/2
        width2 = img_shape[1]/2
        height = img_shape[0]
        width = img_shape[1]
        channels = img_shape[2]
    else:
        ApplyTD = lambda x: TimeDistributed(x)
        arm_in = Input((time_distributed, arm_size,))
        gripper_in = Input((time_distributed, gripper_size,))
        if option is not None:
            option_in = Input((time_distributed,1,))
            option_x = TimeDistributed(OneHot(size=option),name="label_to_one_hot")(option_in)
            option_x = Reshape((time_distributed,option,))(option_x)
        else:
            option_in, option_x = None, None
        height4 = img_shape[1]/4
        width4 = img_shape[2]/4
        height2 = img_shape[1]/2
        width2 = img_shape[2]/2
        height = img_shape[1]
        width = img_shape[2]
        channels = img_shape[3]

    samples = Input(shape=img_shape)

    '''
    Convolutions for an image, terminating in a dense layer of size dim.
    '''

    if leaky:
        relu = lambda: LeakyReLU(alpha=0.2)
    else:
        relu = lambda: Activation('relu')

    x = samples

    x = ApplyTD(Conv2D(filters,
                kernel_size=kernel_size, 
                strides=(1, 1),
                padding='same'))(x)
    x = ApplyTD(relu())(x)
    if batchnorm:
        x = ApplyTD(BatchNormalization(momentum=0.9))(x)
    if dropout:
        x = ApplyTD(Dropout(dropout_rate))(x)

    for i in range(pre_tiling_layers):

        x = ApplyTD(Conv2D(filters,
                   kernel_size=kernel_size, 
                   strides=(1, 1),
                   padding='same'))(x)
        if batchnorm:
            x = ApplyTD(BatchNormalization(momentum=0.9))(x)
        x = ApplyTD(relu())(x)
        #x = MaxPooling2D(pool_size=(2,2))(x)
        if dropout:
            x = ApplyTD(Dropout(dropout_rate))(x)

    # ===============================================
    # ADD TILING
    if tile:
        tile_width = width #int(width/(pre_tiling_layers+))
        tile_height = height #int(height/(pre_tiling_layers+1))
        if option is not None:
            ins = [samples, arm_in, gripper_in, option_in]
        else:
            ins = [samples, arm_in, gripper_in]
        x = TileArmAndGripper(x, arm_in, gripper_in, tile_height, tile_width,
                option, option_x, time_distributed)
    else:
        ins = [samples]

    skips = []
    for i in range(post_tiling_layers):
        if i == post_tiling_layers - 1:
            nfilters = output_filters
        else:
            nfilters = filters
        x = ApplyTD(Conv2D(nfilters,
                   kernel_size=kernel_size, 
                   strides=(2, 2),
                   padding='same'))(x)
        if batchnorm:
            x = ApplyTD(BatchNormalization(momentum=0.9))(x)
        x = relu()(x)
        if dropout:
            x = Dropout(dropout_rate)(x)
        skips.append(x)

    if flatten or dense or discriminator:
        x = ApplyTD(Flatten())(x)
    if dense:
        x = ApplyTD(Dense(dim))(x)
        x = ApplyTD(relu())(x)

    # Single output -- sigmoid activation function
    if discriminator:
        x = Dense(1,activation="sigmoid")(x)

    return ins, x, skips

def AddOptionTiling(x, option_length, option_in, height, width):
    tile_shape = (1, width, height, 1)
    option = Reshape([1,1,option_length])(option_in)
    option = Lambda(lambda x: K.tile(x, tile_shape))(option)
    x = Concatenate(
            axis=-1,
            name="add_option_%dx%d"%(width,height),
        )([x, option])
    return x

def GetDecoder(dim, img_shape, arm_size, gripper_size,
        dropout_rate, filters, kernel_size=[3,3], dropout=True, leaky=True,
        batchnorm=True,dense=True, option=None, num_hypotheses=None,
        tform_filters=None,
        stride2_layers=2, stride1_layers=1):

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

    if tform_filters is None:
        tform_filters = filters

    if leaky:
        relu = lambda: LeakyReLU(alpha=0.2)
    else:
        relu = lambda: Activation('relu')

    if option is not None:
        oin = Input((1,),name="input_next_option")

    if dense:
        z = Input((dim,),name="input_image")
        x = Dense(filters/2 * height4 * width4)(z)
        if batchnorm:
            x = BatchNormalization(momentum=0.9)(x)
        x = relu()(x)
        x = Reshape((width4,height4,tform_filters/2))(x)
    else:
        z = Input((width8*height8*tform_filters,),name="input_image")
        x = Reshape((width8,height8,tform_filters))(z)
    x = Dropout(dropout_rate)(x)

    height = height4
    width = width4
    for i in range(stride2_layers):

        x = Conv2DTranspose(filters,
                   kernel_size=kernel_size, 
                   strides=(2, 2),
                   padding='same')(x)
        if batchnorm:
            x = BatchNormalization(momentum=0.9)(x)
        x = relu()(x)
        #x = UpSampling2D(size=(2,2))(x)
        if dropout:
            x = Dropout(dropout_rate)(x)

        if option is not None:
            opt = OneHot(option)(oin)
            x = AddOptionTiling(x, option, opt, height, width)

        height *= 2
        width *= 2

    for i in range(stride1_layers):
        x = Conv2D(filters, # + num_labels
                   kernel_size=kernel_size, 
                   strides=(1, 1),
                   padding="same")(x)
        if batchnorm:
            x = BatchNormalization(momentum=0.9)(x)
        x = relu()(x)
        if dropout:
            x = Dropout(dropout_rate)(x)
        if option is not None:
            opt = OneHot(option)(oin)
            x = AddOptionTiling(x, option, opt, height, width)

    x = Conv2D(nchannels, (1, 1), padding='same')(x)
    x = Activation('sigmoid')(x)

    ins = [z]
    if option is not None:
        ins.append(oin)

    return ins, x

def GetTCNStack(x, filters, num_levels=2, dense_size=128, dropout_rate=0.5):
    '''
    Add some convolutions to a simple image
    '''

    for i in range(num_levels):
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
        filters, discriminator=False, tile=False, option=None,
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
    if option is not None:
        option_in = Input((option,))

    x = samples

    for i in range(pre_tiling_layers):
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
        if option is not None:
            robot = Concatenate()([arm_in, gripper_in, option_in])
            robot = Reshape([1,1,arm_size+gripper_size+option])(robot)
        else:
            robot = Concatenate()([arm_in, gripper_in])
            robot = Reshape([1,1,arm_size+gripper_size])(robot)
        robot = Lambda(lambda x: K.tile(x, tile_shape))(robot)
        x = Concatenate(axis=3)([x,robot])
        ins = [samples, arm_in, gripper_in]
    else:
        ins = [samples]

    for i in range(post_tiling_layers):
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

