 
def ConfigureGPU(args):
    cpu = True if 'cpu' in args and args['cpu'] else False

    fraction = 1
    if 'gpu_fraction' in args and args['gpu_fraction']:
        fraction = args['gpu_fraction']

    if fraction < 1. or cpu:
        import tensorflow as tf
        import keras.backend as K
        
        if cpu:
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
            sess = tf.Session(config=config)
        else:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        K.set_session(sess)

