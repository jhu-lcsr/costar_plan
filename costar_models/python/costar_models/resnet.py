'''
NOT A REAL PYTHON FILE
These are clipped from planner.py, until further notice.
'''


def GetResnetBlock():
        for i in range(1):
            # =================================================================
            # Start ResNet with a convolutional block
            # This will decrease the size and apply a convolutional filter
            x0 = x
            # ------------------------------------
            x = BatchNormalization(momentum=0.9,)(x)
            x = Dropout(dropout_rate)(x)
            x = Activation("relu")(x)
            x = Conv2D(filters,
                    kernel_size=kernel_size, 
                    strides=(2, 2),
                    padding='same',)(x)
            # ------------------------------------
            x = BatchNormalization(momentum=0.9,)(x)
            x = Dropout(dropout_rate)(x)
            x = Activation("relu")(x)
            x = Conv2D(filters,
                    kernel_size=kernel_size, 
                    strides=(1, 1),
                    padding='same',)(x)

            # ------------------------------------
            if i >= 0:
                # add in the convolution to the beginning of this block
                x0 = BatchNormalization(momentum=0.9,name="norm_ag_%d"%i)(x0)
                x0 = Conv2D(
                        filters,
                        kernel_size=kernel_size,
                        strides=(2,2),
                        padding="same",)(x0)
            x = Add()([x, x0])

            # =================================================================
            # Add Resnet identity blocks after downsizing 
            # Note: currently disabled
            for _ in range(2):
                x0 = x
                # ------------------------------------
                x = BatchNormalization(momentum=0.9,)(x)
                x = Dropout(dropout_rate)(x)
                x = Activation("relu")(x)
                x = Conv2D(filters,
                        kernel_size=kernel_size, 
                        strides=(1, 1),
                        padding='same',)(x)
                # ------------------------------------
                x = BatchNormalization(momentum=0.9,)(x)
                x = Dropout(dropout_rate)(x)
                x = Activation("relu")(x)
                x = Conv2D(filters,
                        kernel_size=kernel_size, 
                        strides=(1, 1),
                        padding='same',)(x)
                # ------------------------------------
                # Recombine
                x = Add()([x, x0])

        x = Flatten()(x)

def GetResnetImageBlock():
            # ====================================
            # Start a Resnet convolutional block
            # The goal in making this change is to increase the representative
            # power and learning rate of the network -- since we were having
            # some trouble with convergence before.
            x0 = x
            # ------------------------------------
            x = BatchNormalization(momentum=0.9,)(x)
            x = Dropout(dropout_rate)(x)
            x = Activation("relu")(x)
            x = Conv2DTranspose(filters,
                    kernel_size=kernel_size, 
                    strides=(2, 2),
                    padding='same',)(x)
            # ------------------------------------
            x = BatchNormalization(momentum=0.9,)(x)
            x = Dropout(dropout_rate)(x)
            x = Activation("relu")(x)
            x = Conv2DTranspose(filters,
                    kernel_size=kernel_size, 
                    strides=(1, 1),
                    padding='same',)(x)

            # ------------------------------------
            # add in the convolution to the beginning of this block
            x0 = BatchNormalization(momentum=0.9,)(x0)
            x0 = Conv2DTranspose(
                    filters,
                    kernel_size=kernel_size,
                    strides=(2,2),
                    padding="same",)(x0)
            x = Add()([x, x0])
def GetResnetArmGripperEtc():
        for i in range(1):
            # =================================================================
            # Start ResNet with a convolutional block
            # This will decrease the size and apply a convolutional filter
            x0 = x
            # ------------------------------------
            x = BatchNormalization(momentum=0.9,)(x)
            x = Dropout(dropout_rate)(x)
            x = Activation("relu")(x)
            x = Conv2D(filters,
                    kernel_size=kernel_size, 
                    strides=(2, 2),
                    padding='same',)(x)
            # ------------------------------------
            x = BatchNormalization(momentum=0.9,)(x)
            x = Dropout(dropout_rate)(x)
            x = Activation("relu")(x)
            x = Conv2D(filters,
                    kernel_size=kernel_size, 
                    strides=(1, 1),
                    padding='same',)(x)

            # ------------------------------------
            if i >= 0:
                # add in the convolution to the beginning of this block
                x0 = BatchNormalization(momentum=0.9,name="norm_ag_%d"%i)(x0)
                x0 = Conv2D(
                        filters,
                        kernel_size=kernel_size,
                        strides=(2,2),
                        padding="same",)(x0)
            x = Add()([x, x0])

            # =================================================================
            # Add Resnet identity blocks after downsizing 
            # Note: currently disabled
            for _ in range(2):
                x0 = x
                # ------------------------------------
                x = BatchNormalization(momentum=0.9,)(x)
                x = Dropout(dropout_rate)(x)
                x = Activation("relu")(x)
                x = Conv2D(filters,
                        kernel_size=kernel_size, 
                        strides=(1, 1),
                        padding='same',)(x)
                # ------------------------------------
                x = BatchNormalization(momentum=0.9,)(x)
                x = Dropout(dropout_rate)(x)
                x = Activation("relu")(x)
                x = Conv2D(filters,
                        kernel_size=kernel_size, 
                        strides=(1, 1),
                        padding='same',)(x)
                # ------------------------------------
                # Recombine
                x = Add()([x, x0])

        x = Flatten()(x)

def GetResnetTransform():
            x0 = x
            x = BatchNormalization(momentum=0.9,
                                    name="normalize_%d_%d"%(idx,j))(x)
            x = Activation("relu",name="reluA_%d_%d"%(idx,j))(x)
            x = Conv2D(filters,
                    kernel_size=[5,5], 
                    strides=(1, 1),
                    padding='same',
                    name="transformA_%d_%d"%(idx,j))(x)
            x = BatchNormalization(momentum=0.9,
                                    name="normalizeB_%d_%d"%(idx,j))(x)
            x = Activation("relu",name="reluB_%d_%d"%(idx,j))(x)
            x = Conv2D(filters,
                    kernel_size=[5,5], 
                    strides=(1, 1),
                    padding='same',
                    name="transformB_%d_%d"%(idx,j))(x)
            # Resnet block addition
            x = Add()([x, x0])
