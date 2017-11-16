
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
                   kernel_size=kernel_size_stride1, 
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

