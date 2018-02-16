import os
import copy
import six
import GPy
import GPyOpt
import numpy as np
import cornell_grasp_train
import cornell_grasp_dataset_reader
import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def add_param(name, domain, domain_type='discrete', search_space=None, index_dict=None, enable=True, required=True, default=None):
    """

    # Arguments

        search_space: list of hyperparameter configurations required by BayseanOptimizer
        index_dict: dictionary that will be used to lookup real values
            and types when we get the hyperopt callback with ints or floats
        enable: this parameter will be part of hyperparameter search
        required: this parameter must be passed to the model
        default: default value if required

    """
    if search_space is None:
        search_space = []
    if index_dict is None:
        index_dict = {'current_index': 0}
    if 'current_index' not in index_dict:
        index_dict['current_index'] = 0

    if enable:
        trainable_index = index_dict['current_index']
        numerical_domain = domain
        lookup_as = float
        # convert string domains to a domain of integer indexes
        if domain_type == 'discrete':
            if isinstance(domain, list) and isinstance(domain[0], str):
                numerical_domain = [i for i in range(len(domain))]
                lookup_as = str
            if isinstance(domain, list) and isinstance(domain[0], bool):
                numerical_domain = [i for i in range(len(domain))]
                lookup_as = bool
            if isinstance(domain, list) and isinstance(domain[0], float):
                lookup_as = float
            else:
                lookup_as = int

        opt_dict = {
            'name': name,
            'type': domain_type,
            'domain': numerical_domain}

        if enable:
            search_space += [opt_dict]
            # create a second version for us to construct the real function call
            opt_dict = copy.deepcopy(opt_dict)
            opt_dict['lookup_as'] = lookup_as
        else:
            opt_dict['lookup_as'] = None

        opt_dict['enable'] = enable
        opt_dict['required'] = required
        opt_dict['default'] = default
        opt_dict['index'] = trainable_index
        opt_dict['domain'] = domain
        index_dict[name] = opt_dict
        index_dict['current_index'] += 1
    return search_space, index_dict


def params_to_args(x, index_dict):
    """ Convert GPyOpt Bayesian Optimizer params back into function call arguments

    Arguments:

        x: the callback parameter of the GPyOpt Bayesian Optimizer
        index_dict: a dictionary with all the information necessary to convert back to function call arguments
    """
    # x is a funky 2d numpy array, so we convert it back to normal parameters
    kwargs = {}
    for key, opt_dict in six.iteritems(index_dict):
        if key == 'current_index':
            continue

        if opt_dict['enable']:
            param_value = x[:, opt_dict['index']]
            if opt_dict['domain'] == 'discrete':
                # the value is an integer indexing into the lookup dict
                kwargs[opt_dict['name']] = opt_dict['lookup_as'](opt_dict['domain'][param_value])
            else:
                # the value is a param to use directly
                kwargs[opt_dict['name']] = opt_dict['lookup_as'](param_value)
        elif opt_dict['required']:
            kwargs[opt_dict['name']] = opt_dict['default']
    return kwargs


def optimize(train_file=None, validation_file=None, seed=1):
    np.random.seed(seed)
    # TODO(ahundt) hyper optimize more input feature_combo_names (ex: remove sin theta cos theta), optimizers, etc
    # continuous variables and then discrete variables
    # I'm going with super conservative values for the first run to get an idea how it works

    load_dataset_in_advance = True
    # since we adaptively initialize the dataset
    # we can also optimize batch size.
    # This will be noticeably slower.
    batch_size = FLAGS.batch_size

    search_space, index_dict = add_param('learning_rate', (0.001, 0.1), 'continuous')
    search_space, index_dict = add_param('dropout_rate', [0.0, 0.25, 0.5], search_space=search_space, index_dict=index_dict)
    search_space, index_dict = add_param('vector_dense_filters', [2**x for x in range(5, 10)], search_space=search_space, index_dict=index_dict)
    search_space, index_dict = add_param('vector_branch_num_layers', [x for x in range(1, 4)], search_space=search_space, index_dict=index_dict)
    search_space, index_dict = add_param('image_model_name', ['vgg', 'resnet', 'densenet'], search_space=search_space, index_dict=index_dict)
    search_space, index_dict = add_param('trainable', [True, False], search_space=search_space, index_dict=index_dict,
                                         enable=False)
    search_space, index_dict = add_param('trunk_filters', [2**x for x in range(6, 12)], search_space=search_space, index_dict=index_dict)
    search_space, index_dict = add_param('trunk_layers', [x for x in range(0, 8)], search_space=search_space, index_dict=index_dict)
    search_space, index_dict = add_param('batch_size', [2**x for x in range(2, 4)], search_space=search_space, index_dict=index_dict,
                                         enable=False, required=True, default=batch_size)

    top = 'classification'
    feature_combo_name = 'preprocessed_image_raw_grasp'
    train_data = None
    validation_data = None

    optimize_batch_size = False

    if load_dataset_in_advance:
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

    # number of samples to take before trying hyperopt
    initial_num_samples = 5  # 50
    num_cores = 15
    baysean_batch_size = 1
    # deep learning algorithms don't give exact results
    algorithm_gives_exact_results = False
    # how many optimization steps to take after the initial sampling
    maximum_iterations_of_optimization = 10  # 100

    def train_callback(x):
        # x is a funky 2d numpy array, so we convert it back to normal parameters
        kwargs = params_to_args(x, index_dict)

        cornell_grasp_train.run_training(
            train_data=train_data,
            validation_data=validation_data,
            **kwargs)

        #     learning_rate=val(learning_rate_index),
        #     batch_size=val(batch_size_index),
        #     image_model_name=val(image_model_name_index),
        #     vector_dense_filters=val(vector_dense_filters_index),
        #     vector_branch_num_layers=val(vector_branch_num_layers_index),
        #     trunk_filters=val(trunk_filters_index),
        #     trunk_layers=val(trunk_layers_index),
        #     train_data=train_data,
        #     validation_data=validation_data,
        #     dropout_rate=val(dropout_index)
        # )

    hyperopt = GPyOpt.methods.BayesianOptimization(
        f=train_callback,  # function to optimize
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
