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
from .temporary import *

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
    x = Conv2D(num_filters, # + num_labels
               kernel_size=[5, 5], 
               strides=(2, 2),
               padding="same")(samples)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(dropout_rate)(x)

    # Add conv layer with more filters
    x = Conv2D(num_filters, # + num_labels
               kernel_size=[5, 5], 
               strides=(2, 2),
               padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(num_filters, # + num_labels
               kernel_size=[5, 5], 
               strides=(2, 2),
               padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(num_filters, # + num_labels
               kernel_size=[5, 5], 
               strides=(2, 2),
               padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(dropout_rate)(x)

    # Add dense layer
    x = Flatten()(x)
    x = Dense(int(0.5 * dense_size))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(dropout_rate)(x)

    # Single output -- sigmoid activation function
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
        x, _ = TileArmAndGripper(arm_in, gripper_in,
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

        x, _ = TileArmAndGripper(arm_in, gripper_in,
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

def GetEncoder(img_shape, state_sizes, dim, dropout_rate,
        filters, discriminator=False, tile=False, dropout=True, leaky=True,
        pose_col_dim=None,
        dense=True, option=None, flatten=True, batchnorm=False,
        pre_tiling_layers=0,
        post_tiling_layers=2,
        stride1_post_tiling_layers=0,
        kernel_size=[3,3], output_filters=None,
        time_distributed=0,
        use_spatial_softmax=False,
        config="arm"):
    '''
    This is the "master" version of the encoder creation function. It takes in
    a massive number of parameters and creates the appropriate inputs and
    structure.

    Parameters:
    -----------
    img_shape:
    pose_size:
    gripper_size:
    dim:
    dropout_rate:
    filters:
    discriminator: 
    tile:
    dropout:
    leaky:
    dense:
    option:
    flatten:
    batchnorm:
    pre_tiling_layers:
    post_tiling_layers:
    kernel_size: 
    output_filters:
    time_distributed: True if you want to wrap this layer in a time distributed
                       setup... sort of deprecated right now.
    config: arm or mobile (arm supported for now)

    1'''

    if not config in ["arm", "mobile"]:
        raise RuntimeError("Encoder config type must be in [arm, mobile]")
    elif config == "arm":
        pose_size, gripper_size = state_sizes
    elif config == "mobile":
        if isinstance(state_sizes,list):
            pose_size = state_sizes[0]
        else:
            pose_size = state_sizes
    else:
        raise RuntimeError('huh? what did you do?')

    if pose_col_dim is None:
        pose_col_dim = dim

    if output_filters is None:
        output_filters = filters

    # ===============================================
    # Parse all of our many options to set up the appropriate inputs for this
    # problem.

    if time_distributed <= 0:
        ApplyTD = lambda x: x
        pose_in = Input((pose_size,),name="pose_position_in")
        if config == "arm":
            gripper_in = Input((gripper_size,),name="gripper_state_in")
        if option is not None:
            option_in = Input((1,),name="prev_option_in")
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
        pose_in = Input((time_distributed, pose_size,))
        if config == "arm":
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
        x = ApplyTD(BatchNormalization())(x)
    if dropout:
        x = ApplyTD(Dropout(dropout_rate))(x)

    # ===============================================
    # Create preprocessing layers that just act on the image. These do not
    # modify the image size at all.
    for i in range(pre_tiling_layers):

        x = ApplyTD(Conv2D(filters,
                   kernel_size=kernel_size, 
                   strides=(1, 1),
                   padding='same'))(x)
        if batchnorm:
            x = ApplyTD(BatchNormalization())(x)
        x = ApplyTD(relu())(x)
        if dropout:
            x = ApplyTD(Dropout(dropout_rate))(x)
    
    # ===============================================
    # Skip connections
    skips = [x]

    # ===============================================
    # ADD TILING
    if tile:
        tile_width = width 
        tile_height = height 
        if option is not None:
            if config == "arm":
                ins = [samples, pose_in, gripper_in, option_in]
            elif config == "mobile":
                ins = [samples, pose_in, option_in]
        else:
            if config == "arm":
                ins = [samples, pose_in, gripper_in]
            elif config == "mobile":
                ins = [samples, pose_in]

        if config == "arm":
            x, robot = TileArmAndGripper(x, pose_in, gripper_in, tile_height, tile_width,
                    None, None, time_distributed, pose_col_dim)
        elif config == "mobile":
            x, robot = TilePose(x, pose_in, tile_height, tile_width, None, None, time_distributed, pose_col_dim)
        

    else:
        ins = [samples]

    # =================================================
    # Decrease the size of the image
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
            x = ApplyTD(BatchNormalization())(x)
        x = relu()(x)
        if dropout:
            x = Dropout(dropout_rate)(x)
        skips.append(x)

    # =================================================
    # Perform additional operations
    for i in range(stride1_post_tiling_layers):
        if i == post_tiling_layers - 1:
            nfilters = output_filters
        else:
            nfilters = filters
        x = ApplyTD(Conv2D(nfilters,
                   kernel_size=kernel_size, 
                   strides=(1, 1),
                   padding='same'))(x)
        if batchnorm:
            x = ApplyTD(BatchNormalization())(x)
        x = relu()(x)
        if dropout:
            x = Dropout(dropout_rate)(x)

    if option is not None:
        nfilters = output_filters
        option_x = OneHot(size=option)(option_in)
        option_x = Reshape((option,))(option_x)
        x = TileOnto(x,option_x,option,
                (height/(2**post_tiling_layers),
                 width/(2**post_tiling_layers)))

        x = ApplyTD(Conv2D(nfilters,
                   kernel_size=kernel_size, 
                   strides=(1, 1),
                   padding='same'))(x)
        if batchnorm:
            x = ApplyTD(BatchNormalization())(x)
        x = relu()(x)
        if dropout:
            x = Dropout(dropout_rate)(x)

    # =================================================
    # Compute spatial softmax. This converts from an image to a set of
    # keypoints capturing salient features of the image. For example, keypoints
    # capture relevant objects, robot gripper position, etc. See:
    #   "Learning visual feature spaces for robotic manipulation with
    #   deep spatial autoencoders." Finn et al.,
    #   http://arxiv.org/abs/1509.06113.
    if use_spatial_softmax:
        def _ssm(x):
            #return tf.contrib.
            return spatial_softmax(x)
        x = Lambda(_ssm,name="encoder_spatial_softmax")(x)
        #pool_size = (height/(2**post_tiling_layers),
        #         width/(2**post_tiling_layers))
        #x2 = MaxPooling2D(pool_size=pool_size)(x)
        #x2 = Flatten()(x2)
        #x = Concatenate()([x1,x2])
    elif flatten or dense or discriminator:
        x = ApplyTD(Flatten())(x)
        if dense:
            x = ApplyTD(Dense(dim))(x)
            x = ApplyTD(relu())(x)
    # Single output -- sigmoid activation function
    if discriminator:
        x = Dense(1,activation="sigmoid")(x)

    return ins, x, skips, robot

def AddOptionTiling(x, option_length, option_in, height, width):
    tile_shape = (1, width, height, 1)
    option = Reshape([1,1,option_length])(option_in)
    option = Lambda(lambda x: K.tile(x, tile_shape))(option)
    x = Concatenate(
            axis=-1,
            name="add_option_%dx%d"%(width,height),
        )([x, option])
    return x

def GetHuskyEncoder(img_shape, pose_size, dim, dropout_rate,
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
        pose_in = Input((pose_size,))
        
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
        pose_in = Input((time_distributed, pose_size,))
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
        x = ApplyTD(BatchNormalization())(x)
    if dropout:
        x = ApplyTD(Dropout(dropout_rate))(x)

    for i in range(pre_tiling_layers):

        x = ApplyTD(Conv2D(filters,
                   kernel_size=kernel_size, 
                   strides=(1, 1),
                   padding='same'))(x)
        if batchnorm:
            x = ApplyTD(BatchNormalization())(x)
        x = ApplyTD(relu())(x)
        if dropout:
            x = ApplyTD(Dropout(dropout_rate))(x)
    
    # ===============================================
    # Skip connections
    skips = [x]

    # ===============================================
    # ADD TILING
    if tile:
        tile_width = width #int(width/(pre_tiling_layers+))
        tile_height = height #int(height/(pre_tiling_layers+1))
        if option is not None:
            ins = [samples, pose_in, option_in]
        else:
            ins = [samples, pose_in]
        x, robot = TilePose(x, pose_in, tile_height, tile_width,
                option, option_x, time_distributed, dim)
    else:
        ins = [samples]
        

    # =================================================
    # Decrease the size of the image
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
            x = ApplyTD(BatchNormalization())(x)
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

    return ins, x, skips, robot


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
            x = BatchNormalization()(x)
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
            x = BatchNormalization()(x)
        x = relu()(x)
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
            x = BatchNormalization()(x)
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

def GetHuskyDecoder(dim, img_shape, pose_size,
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
            x = BatchNormalization()(x)
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
            x = BatchNormalization()(x)
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
            x = BatchNormalization()(x)
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

