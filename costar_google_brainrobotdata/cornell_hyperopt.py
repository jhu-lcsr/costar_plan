import os
import sys
import copy
import six
import GPy
import GPyOpt
import numpy as np
import grasp_utilities
import cornell_grasp_train
import cornell_grasp_dataset_reader
import tensorflow as tf
import traceback
import keras
from tensorflow.python.platform import flags

# progress bars https://github.com/tqdm/tqdm
# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)

FLAGS = flags.FLAGS


class HyperparameterOptions(object):

    def __init__(self, verbose=1):
        self.index_dict = {}
        self.search_space = []
        self.verbose = verbose

    def add_param(self, name, domain, domain_type='discrete', enable=True, required=True, default=None):
        """

        # Arguments

            search_space: list of hyperparameter configurations required by BayseanOptimizer
            index_dict: dictionary that will be used to lookup real values
                and types when we get the hyperopt callback with ints or floats
            enable: this parameter will be part of hyperparameter search
            required: this parameter must be passed to the model
            default: default value if required

        """
        if self.search_space is None:
            self.search_space = []
        if self.index_dict is None:
            self.index_dict = {'current_index': 0}
        if 'current_index' not in self.index_dict:
            self.index_dict['current_index'] = 0

        if enable:
            param_index = self.index_dict['current_index']
            numerical_domain = domain
            needs_reverse_lookup = False
            lookup_as = float
            # convert string domains to a domain of integer indexes
            if domain_type == 'discrete':
                if isinstance(domain, list) and isinstance(domain[0], str):
                    numerical_domain = [i for i in range(len(domain))]
                    lookup_as = str
                    needs_reverse_lookup = True
                elif isinstance(domain, list) and isinstance(domain[0], bool):
                    numerical_domain = [i for i in range(len(domain))]
                    lookup_as = bool
                    needs_reverse_lookup = True
                elif isinstance(domain, list) and isinstance(domain[0], float):
                    lookup_as = float
                else:
                    lookup_as = int

            opt_dict = {
                'name': name,
                'type': domain_type,
                'domain': numerical_domain}

            if enable:
                self.search_space += [opt_dict]
                # create a second version for us to construct the real function call
                opt_dict = copy.deepcopy(opt_dict)
                opt_dict['lookup_as'] = lookup_as
            else:
                opt_dict['lookup_as'] = None

            opt_dict['enable'] = enable
            opt_dict['required'] = required
            opt_dict['default'] = default
            opt_dict['index'] = param_index
            opt_dict['domain'] = domain
            opt_dict['needs_reverse_lookup'] = needs_reverse_lookup
            self.index_dict[name] = opt_dict
            self.index_dict['current_index'] += 1

    def params_to_args(self, x):
        """ Convert GPyOpt Bayesian Optimizer params back into function call arguments

        Arguments:

            x: the callback parameter of the GPyOpt Bayesian Optimizer
            index_dict: a dictionary with all the information necessary to convert back to function call arguments
        """
        if len(x.shape) == 1:
            # if we get a 1d array convert it to 2d so we are consistent
            x = np.expand_dims(x, axis=0)
        # x is a funky 2d numpy array, so we convert it back to normal parameters
        kwargs = {}
        for key, opt_dict in six.iteritems(self.index_dict):
            if key == 'current_index':
                continue

            if opt_dict['enable']:
                arg_name = opt_dict['name']
                optimizer_param_column = opt_dict['index']
                if optimizer_param_column > x.shape[-1]:
                    raise ValueError('Attempting to access optimizer_param_column' + str(optimizer_param_column) +
                                     ' outside parameter bounds' + str(x.shape) +
                                     ' of optimizer array with index dict: ' + str(self.index_dict) +
                                     'and array x: ' + str(x))
                param_value = x[:, optimizer_param_column]
                if opt_dict['type'] == 'discrete':
                    # the value is an integer indexing into the lookup dict
                    if opt_dict['needs_reverse_lookup']:
                        domain_index = int(param_value)
                        domain_value = opt_dict['domain'][domain_index]
                        value = opt_dict['lookup_as'](domain_value)
                    else:
                        value = opt_dict['lookup_as'](param_value)

                else:
                    # the value is a param to use directly
                    value = opt_dict['lookup_as'](param_value)

                kwargs[arg_name] = value
            elif opt_dict['required']:
                kwargs[opt_dict['name']] = opt_dict['default']
        return kwargs

    def get_domain(self):
        return self.search_space


def optimize(
        seed=1,
        verbose=1,
        initial_num_samples=200,
        num_cores=15,
        baysean_batch_size=1,
        problem_type=None,
        log_dir='./',
        run_name='',
        param_to_optimize='val_loss'):
    """ Run hyperparameter optimization
    """
    np.random.seed(seed)
    # TODO(ahundt) hyper optimize more input feature_combo_names (ex: remove sin theta cos theta), optimizers, etc
    # continuous variables and then discrete variables
    # I'm going with super conservative values for the first run to get an idea how it works
    # since we adaptively initialize the dataset
    # we can also optimize batch size.
    # This will be noticeably slower.
    batch_size = FLAGS.batch_size
    if problem_type == 'classification':
        FLAGS.problem_type = problem_type
        feature_combo_name = 'image_preprocessed_width_1'
        top = 'classification'
    elif problem_type == 'grasp_regression':
        feature_combo_name = 'image_preprocessed'
        # Override some default flags for this configuration
        # see other configuration in cornell_grasp_train.py choose_features_and_metrics()
        FLAGS.problem_type = problem_type
        FLAGS.feature_combo = feature_combo_name
        FLAGS.crop_to = 'image_contains_grasp_box_center'

    learning_rate_enabled = False

    hyperoptions = HyperparameterOptions()
    # Configuring hyperparameters

    # The trainable flag referrs to the imagenet pretrained network being trainable or not trainable.
    # We are defaulting to a reasonable learning rate found by prior searches and disabling trainability so that
    # we can restrict the search to more model improvements due to the outsized effects of these changes on performance
    # during the random search phase. We plan to run a separate search on enabling trainable models on one of the best models
    # found when trainable is false. This is due to limitations in processing time availability at the time of writing and
    # multi stage training not yet being configurable during hyperopt.
    hyperoptions.add_param('trainable', [True, False], enable=False)

    # Learning rates are exponential so we take a uniform random
    # input and map it from 1 to 3e-5 on an exponential scale.
    # with a base of 0.9.
    # Therefore the value 50 is 0.9^50 == 0.005 (approx).
    hyperoptions.add_param('learning_rate', (0.0, 100.0), 'continuous',
                           enable=learning_rate_enabled, required=True, default=0.01)
    # disabled dropout rate because in one epoch tests a dropout rate of 0 allows exceptionally fast learning.
    # TODO(ahundt) run a separate search for the best dropout rate after finding a good model
    hyperoptions.add_param('dropout_rate', [0.0, 0.125, 0.2, 0.25, 0.5, 0.75],
                           enable=False, required=True, default=0.25)

    if problem_type == 'grasp_regression':
        # TODO(ahundt) determine how to try out different losses because otherwise we can't compare runs
        # Right now only grasp_regression can configure the loss function
        # hyperoptions.add_param('loss', ['mse', 'mae', 'logcosh'])
        pass
    else:
        # There is no vector branch for grasp regression so we only add it in the other cases
        # Handle motion command inputs and search for the best configuration
        hyperoptions.add_param('vector_dense_filters', [2**x for x in range(6, 13)])
        hyperoptions.add_param('vector_branch_num_layers', [x for x in range(0, 5)])
        hyperoptions.add_param('vector_model_name', ['dense', 'dense_block'])
    # leaving out nasnet_large for now because it needs different input dimensions.
    hyperoptions.add_param('image_model_name', ['vgg', 'densenet', 'nasnet_mobile', 'resnet'])
    # Zero maps to the None option for trunk_filters,
    # because it will automatically match the input data's filter count
    hyperoptions.add_param('trunk_filters', [0] + [2**x for x in range(5, 11)])
    hyperoptions.add_param('trunk_layers', [x for x in range(0, 8)])
    # TODO(ahundt) Enable 'nasnet_normal_a_cell' the option is disabled for now due to a tensor dimension conflict
    hyperoptions.add_param('trunk_model_name', ['vgg_conv_block', 'dense_block', 'resnet_conv_identity_block'])
    hyperoptions.add_param('top_block_filters', [2**x for x in range(5, 12)])
    hyperoptions.add_param('batch_size', [2**x for x in range(2, 4)],
                           enable=False, required=True, default=batch_size)
    # enable this if you're search for grasp classifications
    hyperoptions.add_param('feature_combo_name', ['image_preprocessed_width_1', 'image_preprocessed_sin_cos_width_3'],
                           enable=False, required=True, default=feature_combo_name)
    hyperoptions.add_param('preprocessing_mode', ['tf', 'caffe', 'torch'],
                           enable=False, required=True, default='tf')

    # deep learning algorithms don't give exact results
    algorithm_gives_exact_results = False
    # how many optimization steps to take after the initial sampling
    maximum_hyperopt_steps = 100
    total_max_steps = initial_num_samples + maximum_hyperopt_steps
    # defining a temporary variable scope for the callbacks
    class ProgUpdate():
        hyperopt_current_update = 0
        progbar = tqdm(desc='hyperopt', total=total_max_steps)

    def train_callback(x):
        # x is a funky 2d numpy array, so we convert it back to normal parameters
        kwargs = hyperoptions.params_to_args(x)

        if learning_rate_enabled:
            # Learning rates are exponential so we take a uniform random
            # input and map it from 1 to 3e-5 on an exponential scale.
            kwargs['learning_rate'] = 0.9 ** kwargs['learning_rate']

        if verbose:
            # update counts by 1 each step
            ProgUpdate.progbar.update()
            ProgUpdate.progbar.write('Training with hyperparams: \n' + str(kwargs))
        ProgUpdate.hyperopt_current_update += 1

        history = None

        try:
            history = cornell_grasp_train.run_training(
                hyperparams=kwargs,
                **kwargs)
        except tf.errors.ResourceExhaustedError as exception:
            print('Hyperparams caused algorithm to run out of resources, '
                  'will continue to next stage and return infinity loss for now.'
                  'To avoid this entirely you might set more memory sensitive hyperparam ranges,'
                  'or add constraints to your hyperparam search so it does not choose'
                  'huge values for all the parameters at once'
                  'Error: ', exception)
            loss = float('inf')
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            # deletion must be explicit to prevent leaks
            # https://stackoverflow.com/a/16946886/99379
            del tb
        except ValueError as exception:
            print('Hyperparams encountered a model that failed with an invalid combination of values, '
                  'we will continue to next stage and return infinity loss for now.'
                  'To avoid this entirely you will need to debug your model w.r.t. '
                  'the current hyperparam choice.'
                  'Error: ', exception)
            loss = float('inf')
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            # deletion must be explicit to prevent leaks
            # https://stackoverflow.com/a/16946886/99379
            del tb

        # TODO(ahundt) consider shutting down dataset generators and clearing the session when there is an exception
        # https://github.com/tensorflow/tensorflow/issues/4735#issuecomment-363748412
        keras.backend.clear_session()

        if history is not None:
            # hyperopt seems to be done on val_loss
            # may try 1-val_acc sometime (since the hyperopt minimizes)
            if param_to_optimize in history.history:
                loss = history.history[param_to_optimize][-1]
            else:
                raise ValueError('A hyperopt step completed, but the parameter '
                                 'being optimized over is %s and it '
                                 'was missing from the history'
                                 'so hyperopt must exit. Here are the contents '
                                 'of the history.history dictionary:\n\n %s' %
                                 (param_to_optimize, str(history.history)))
            if verbose > 0:
                if 'val_binary_accuracy' in history.history:
                    acc = history.history['val_binary_accuracy'][-1]
                    ProgUpdate.progbar.write('val_binary_accuracy: ' + str(acc))
        else:
            # we probably hit an exception so consider this infinite loss
            loss = float('inf')

        return loss

    hyperopt = GPyOpt.methods.BayesianOptimization(
        f=train_callback,  # function to optimize
        domain=hyperoptions.get_domain(),  # where are we going to search
        initial_design_numdata=initial_num_samples,
        model_type='GP_MCMC',
        acquisition_type='EI_MCMC',  # EI
        evaluator_type="predictive",  # Expected Improvement
        batch_size=baysean_batch_size,
        num_cores=num_cores,
        exact_feval=algorithm_gives_exact_results)

    # run the optimization, this will take a long time!
    hyperopt.run_optimization(max_iter=maximum_hyperopt_steps)
    x_best = hyperopt.x_opt
    # myBopt.X[np.argmin(myBopt.Y)]
    best_hyperparams = hyperoptions.params_to_args(x_best)
    result_file = os.path.join(log_dir, run_name + '_optimized_hyperparams.json')
    with open(result_file, 'w') as fp:
        json.dump(best_hyperparams, fp)
    print('Hyperparameter Optimization final best result:\n' + str(best_hyperparams))
    print("Optimized loss: {0}".format(hyperopt.fx_opt))

    hyperopt.plot_convergence()
    hyperopt.plot_acquisition()
    ProgUpdate.progbar.close()


def main(_):

    FLAGS.problem_type = 'grasp_regression'
    FLAGS.num_validation = 1
    FLAGS.num_test = 1
    FLAGS.epochs = 1
    FLAGS.fine_tuning_epochs = 0
    print('Overriding some flags, edit cornell_hyperopt.py directly to change them.' +
          ' num_validation: ' + str(FLAGS.num_validation) +
          ' num_test: ' + str(FLAGS.num_test) +
          ' epochs: ' + str(FLAGS.epochs) +
          ' fine_tuning_epochs: ' + str(FLAGS.fine_tuning_epochs) +
          ' problem_type:' + str(FLAGS.problem_type))
    run_name = FLAGS.run_name
    log_dir = FLAGS.log_dir
    run_name = grasp_utilities.timeStamped(run_name)
    optimize(problem_type=FLAGS.problem_type, run_name=run_name, log_dir=log_dir)

if __name__ == '__main__':
    # next FLAGS line might be needed in tf 1.4 but not tf 1.5
    # FLAGS._parse_flags()
    tf.app.run(main=main)
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
