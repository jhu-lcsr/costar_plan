    def _makePredictor(self, features):
        '''
        Create model to predict possible manipulation goals.
        '''
        (images, arm, gripper) = features
        img_shape = images.shape[1:]
        arm_size = arm.shape[-1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[-1]
        else:
            gripper_size = 1

        ins, enc = GetEncoder(img_shape,
                arm_size,
                gripper_size,
                self.img_col_dim,
                self.dropout_rate,
                self.img_num_filters,
                leaky=False,
                dropout=False,
                pre_tiling_layers=0,
                post_tiling_layers=3,
                kernel_size=[5,5],
                dense=False,
                batchnorm=True,
                tile=True,
                #option=64,
                flatten=False,
                )
        gins, genc = GetEncoder(img_shape,
                arm_size,
                gripper_size,
                self.img_col_dim,
                self.dropout_rate,
                self.img_num_filters,
                leaky=False,
                dropout=False,
                pre_tiling_layers=0,
                post_tiling_layers=3,
                kernel_size=[5,5],
                dense=False,
                batchnorm=True,
                tile=True,
                #option=64,
                flatten=False,
                )


        image_outs = []
        arm_outs = []
        gripper_outs = []
        train_outs = []

        # =====================================================================
        # Create many different image decoders
        for i in xrange(self.num_hypotheses):
            #x = Conv2D(self.img_num_filters,
            #        kernel_size=[5,5], 
            #        strides=(1, 1),
            #        padding='same')(enc)
            #x = LeakyReLU(0.2)(x)
            x = enc

            
            # =====================================================================
            # Make the decoder.
            rep, dec = GetImageDecoder(self.img_col_dim,
                                img_shape,
                                dropout_rate=self.dropout_rate,
                                kernel_size=[5,5],
                                filters=self.img_num_filters,
                                stride2_layers=3,
                                stride1_layers=0,
                                dropout=False,
                                leaky=True,
                                dense=False,
                                batchnorm=True,)
            # Now.
            # Decode arm/gripper state.
            # Predict the next joint states and gripper position. We add these back
            # in from the inputs once again, in order to make sure they don't get
            # lost in all the convolution layers above...
            height4 = img_shape[0]/4
            width4 = img_shape[1]/4
            height8 = img_shape[0]/8
            width8 = img_shape[1]/8
            x = Reshape((width8,height8,self.img_num_filters))(rep)
            x = Conv2D(self.img_num_filters/2,
                    kernel_size=[5,5], 
                    strides=(2, 2),
                    padding='same')(x)
            x = Flatten()(x)
            x = LeakyReLU(0.2)(x)
            x = Dense(self.combined_dense_size)(x)
            x = Dropout(self.dropout_rate)(x)
            x = LeakyReLU(0.2)(x)
            arm_out_x = Dense(arm_size,name="next_arm")(x)
            gripper_out_x = Dense(gripper_size,
                    name="next_gripper_flat")(x)

            decoder = Model(rep, [dec, arm_out_x, gripper_out_x])

            # =====================================================================
            # Apply the decoder
            # This maps from our latent world state back into observable images.
            #decoder = Model(rep, dec)
            img_x, arm_x, gripper_x = decoder(enc)
            img_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="img_hypothesis_%d"%i)(img_x)
            arm_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="arm_hypothesis_%d"%i)(arm_x)
            gripper_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="gripper_hypothesis_%d"%i)(gripper_x)

            # Create the training outputs
            train_x = Concatenate(axis=-1,name="combine_train_%d"%i)([
                            Flatten(name="flatten_img_%d"%i)(img_x),
                            Flatten(name="flatten_arm_%d"%i)(arm_x),
                            Flatten(name="flatten_gripper_%d"%i)(gripper_x)])
            train_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="flattened_hypothesis_%d"%i)(train_x)

            image_outs.append(img_x)
            arm_outs.append(arm_x)
            gripper_outs.append(gripper_x)
            train_outs.append(train_x)

        image_out = Concatenate(axis=1)(image_outs)
        arm_out = Concatenate(axis=1)(arm_outs)
        gripper_out = Concatenate(axis=1)(gripper_outs)
        train_out = Concatenate(axis=1)(train_outs)

        # =====================================================================
        # Training the actor policy
        y = Concatenate(axis=-1,name="combine_goal_current")([enc, genc])
        y = Conv2D(self.img_num_filters/4,
                kernel_size=[5,5], 
                strides=(2, 2),
                padding='same')(y)
        y = Dropout(self.dropout_rate)(y)
        y = LeakyReLU(0.2)(y)
        y = Flatten()(y)
        y = Dense(self.combined_dense_size)(y)
        y = Dropout(self.dropout_rate)(y)
        y = LeakyReLU(0.2)(y)
        arm_cmd_out = Lambda(lambda x: K.expand_dims(x, axis=1),name="arm_action")(
                Dense(arm_size)(y))
        gripper_cmd_out = Lambda(lambda x: K.expand_dims(x, axis=1),name="gripper_action")(
                Dense(gripper_size)(y))


        # =====================================================================
        # Create models to train

        #predictor = Model(ins, [decoder(enc), arm_out, gripper_out])
        predictor = Model(ins, [image_out, arm_out, gripper_out])
        #predictor.summary()
        actor = Model(ins + gins, [arm_out, gripper_out])
        #actor.summary()
        train_predictor = Model(ins + gins, [train_out,
                                             arm_cmd_out,
                                             gripper_cmd_out,])

        return predictor, train_predictor, actor


