import os
import GPy
import GPyOpt
import numpy as np
import cornell_grasp_train
import cornell_grasp_dataset_reader
import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def optimize(train_file=None, validation_file=None, seed=1):
    np.random.seed(seed)
    # TODO(ahundt) hyper optimize more input feature_combo_names (ex: remove sin theta cos theta), optimizers, etc
    # continuous variables and then discrete variables
    # I'm going with super conservative values for the first run to get an idea how it works
    search_space = [
        {'name': 'lr',
         'type': 'continuous',
         'domain': (0.001, 0.1)},
        # {'name': 'dropout_rate',
        #  'type': 'continuous',
        #  'domain': (0.0, 0.75)},
        {'name': 'dropout_rate',
         'type': 'discrete',
         'domain': [0.0, 0.25, 0.5]},
        {'name': 'vector_dense_filters',
         'type': 'discrete',
         'domain': [2**x for x in range(2, 10)]},
        {'name': 'vector_branch_num_layers',
         'type': 'discrete',
         'domain': [x for x in range(1, 4)]},
        {'name': 'image_model_name',
         'type': 'discrete',
         'domain': ['vgg', 'resnet', 'densenet']},
        # {'name': 'trainable',
        #  'type': 'discrete',
        #  'domain': [True, False]},
        {'name': 'trunk_filters',
         'type': 'discrete',
         'domain': [2**x for x in range(7, 12)]},
        {'name': 'trunk_layers',
         'type': 'discrete',
         'domain': [x for x in range(0, 8)]},
    ]

    load_dataset_in_advance = True
    top = 'classification'
    feature_combo_name = 'preprocessed_image_raw_grasp'
    train_data = None
    validation_data = None

    if load_dataset_in_advance:
        batch_size = 10
        train_file = None
        validation_file = None
        # we load the dataset in advance here because it takes a long time!
        if train_file is None:
            train_file = os.path.join(FLAGS.data_dir, FLAGS.train_filename)
        if validation_file is None:
            validation_file = os.path.join(FLAGS.data_dir, FLAGS.evaluate_filename)

        [samples_in_val_dataset, steps_per_epoch_train,
         steps_in_val_dataset, val_batch_size] = cornell_grasp_train.epoch_params(batch_size)

        [image_shapes, vector_shapes, data_features, model_name,
         monitor_loss_name, label_features, monitor_metric_name,
         loss, metrics] = cornell_grasp_train.feature_selection(feature_combo_name, top)

        train_data, validation_data = cornell_grasp_train.load_dataset(
            validation_file, label_features, data_features,
            samples_in_val_dataset, train_file, batch_size,
            val_batch_size)
    else:
        # since we adaptively initialize the dataset
        # we can also optimize batch size.
        # This will be noticeably slower.
        search_space = search_space + [
            {'name': 'batch_size',
             'type': 'discrete',
             'domain': [2**x for x in range(2, 4)]}
        ]
    # number of samples to take before trying hyperopt
    initial_num_samples = 5  # 50
    num_cores = 15
    baysean_batch_size = 1
    # deep learning algorithms don't give exact results
    algorithm_gives_exact_results = False
    # how many optimization steps to take after the initial sampling
    maximum_iterations_of_optimization = 10  # 100

    hyperopt = GPyOpt.methods.BayesianOptimization(
        f=cornell_grasp_train.run_training,  # function to optimize
        domain=search_space,  # where are we going to search
        initial_design_numdata=initial_num_samples,
        model_type="GP_MCMC",
        acquisition_type='EI_MCMC',  # EI
        evaluator_type="predictive",  # Expected Improvement
        batch_size=baysean_batch_size,
        num_cores=num_cores,
        exact_feval=algorithm_gives_exact_results)

    # run the optimization, this will take a long time!
    hyperopt.run_optimization(max_iter=maximum_iterations_of_optimization)
    x_best = hyperopt.x_opt
    # myBopt.X[np.argmin(myBopt.Y)]
    print('optimization final best result: ' + str(x_best))
    # print optimized model
    print("optimized parameters: {0}".format(hyperopt.x_opt))
    print("optimized loss: {0}".format(hyperopt.fx_opt))

    hyperopt.plot_convergence()
    hyperopt.plot_acquisition()


def main(_):
    optimize()

if __name__ == '__main__':
    # next FLAGS line might be needed in tf 1.4 but not tf 1.5
    # FLAGS._parse_flags()
    tf.app.run(main=main)
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
