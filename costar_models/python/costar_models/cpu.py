 
def ConfigureGPU(args):
    if 'cpu' in args and args['cpu']:
        cpu = True
    else:
        cpu = False
    if 'gpu_fraction' in args and args['gpu_fraction']:
        fraction = args['gpu_fraction']
    else:
        fraction = 1.

    if fraction < 1. or cpu:
        import tensorflow as tf
        import keras.backend as K
        
        if cpu:
            with tf.device('/cpu:0'):
                config = tf.ConfigProto(
                    device_count={'GPU': 0},
                    gpu_options=gpu_options
                )
                sess = tf.Session(config=config)
        else:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        K.set_session(sess)

