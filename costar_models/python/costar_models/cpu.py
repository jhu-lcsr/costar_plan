 
def SetCPU(args):
    if 'cpu' in args and args['cpu']:
        import tensorflow as tf
        import keras.backend as K

        with tf.device('/cpu:0'):
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
            sess = tf.Session(config=config)
            K.set_session(sess)
