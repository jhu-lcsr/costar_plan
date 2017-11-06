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
        padding="same",
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
        post_tiling_layers_no_skip=0,
        padding="same",
        stride1_post_tiling_layers=0,
        kernel_size=[3,3],
        kernel_size_stride1=None,
        output_filters=None,
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
    arm_size:
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
    padding: 
    time_distributed: True if you want to wrap this layer in a time distributed
                       setup... sort of deprecated right now.
    config: arm or mobile (arm supported for now)

    '''

    use_arm_gripper = False
    if not config in ["arm", "mobile"]:
        raise RuntimeError("Encoder config type must be in [arm, mobile]")
    elif config == "arm":
        arm_size, gripper_size = state_sizes
        if arm_size is not None:
            use_arm_gripper = True
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
    if kernel_size_stride1 is None:
        kernel_size_stride1 = kernel_size

    # ===============================================
    # Parse all of our many options to set up the appropriate inputs for this
    # problem.

    if time_distributed <= 0:
        ApplyTD = lambda x: x
        if use_arm_gripper:
            arm_in = Input((arm_size,),name="arm_position_in")
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
        if use_arm_gripper:
            arm_in = Input((time_distributed, arm_size,))
            gripper_in = Input((time_distributed, gripper_size,))
        if option is not None:
            option_in = Input((time_distributed,1,))
            option_x = TimeDistributed(
                    OneHot(size=option),
                    name="label_to_one_hot")(option_in)
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

    # ===============================================
    # Create preprocessing layers that just act on the image. These do not
    # modify the image size at all.
    # Padding here is locked to the same -- since we always want to use this
    # one with skip connections
    for i in range(pre_tiling_layers):

        x = ApplyTD(Conv2D(int(filters / 2),
                   kernel_size=kernel_size, 
                   strides=(1, 1),
                   padding="same"))(x)
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
    if use_arm_gripper and tile:
        tile_width = width 
        tile_height = height 
        if option is not None:
            ins = [samples, arm_in, gripper_in, option_in]
        else:
            ins = [samples, arm_in, gripper_in]
        x, robot = TileArmAndGripper(x, arm_in, gripper_in, tile_height, tile_width,
                None, None, time_distributed, pose_col_dim)
    else:
        ins = [samples]
        robot = None

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
                   padding=padding))(x)
        if batchnorm:
            x = ApplyTD(BatchNormalization())(x)
        x = relu()(x)
        if dropout:
            x = Dropout(dropout_rate)(x)
        if i + 1 <=  (post_tiling_layers - post_tiling_layers_no_skip):
            skips.append(x)

    # ==================================================
    # Add previous option slightly earlier
    if option is not None:
        nfilters = output_filters
        option_x = OneHot(size=option)(option_in)
        option_x = Reshape((option,))(option_x)
        x = TileOnto(x,option_x,option,
                (height/(2**post_tiling_layers),
                 width/(2**post_tiling_layers)))

        x = ApplyTD(Conv2D(nfilters,
                   kernel_size=kernel_size_stride1, 
                   strides=(1, 1),
                   padding=padding))(x)
        if batchnorm:
            x = ApplyTD(BatchNormalization())(x)
        x = relu()(x)
        if dropout:
            x = Dropout(dropout_rate)(x)

    # =================================================
    # Perform additional operations
    for i in range(stride1_post_tiling_layers):
        if i == stride1_post_tiling_layers - 1:
            nfilters = output_filters
        else:
            nfilters = filters
        x = ApplyTD(Conv2D(nfilters,
                   kernel_size=kernel_size_stride1, 
                   strides=(1, 1),
                   padding=padding))(x)
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
    if dense and use_spatial_softmax:
        def _ssm(x):
            return spatial_softmax(x)
        x = Lambda(_ssm,name="encoder_spatial_softmax")(x)
    elif flatten or dense or discriminator:
        x = ApplyTD(Flatten())(x)
        if dense:
            dim = int(dim)
            x = ApplyTD(Dense(dim))(x)
            x = ApplyTD(relu())(x)

    # Single output -- sigmoid activation function
    if discriminator:
        x = Dense(1,activation="sigmoid")(x)

    return ins, x, skips, robot

