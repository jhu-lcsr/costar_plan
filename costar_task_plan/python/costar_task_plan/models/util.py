
from multi_gan_model import RobotMultiGAN
from multi_regression_model import RobotMultiFFRegression

from multi_tcn_regression_model import RobotMultiTCNRegression
from multi_lstm_regression import RobotMultiLSTMRegression
from multi_trajectory_sampler import RobotMultiTrajectorySampler
from multi_autoencoder_model import RobotMultiAutoencoder
from multi_hierarchical import RobotMultiHierarchical

from multi_unsupervised_model import RobotMultiUnsupervised
from multi_unsupervised1_model import RobotMultiUnsupervised1

def MakeModel(features, model, taskdef, *args, **kwargs):
    '''
    This function will create the appropriate neural net based on images and so
    on.

    Parameters:
    -----------
    features: string describing the set of features (inputs) we are using.
    model: string describing the particular method that should be applied.
    taskdef: a (simulation) task definition used to extract specific
             parameters.
    '''
    # set up some image parameters
    if features in ['rgb', 'multi']:
        nchannels = 3
    elif features in ['depth']:
        nchannels = 1

    model_instance = None
    model = model.lower()

    if features == 'multi':
        if model == 'gan':
            model_instance = RobotMultiGAN(taskdef,
                    model=model,
                    **kwargs)
        elif model == 'ff_regression':
            model_instance = RobotMultiFFRegression(taskdef,
                    model=model,
                    **kwargs)
        elif model == 'tcn_regression':
            model_instance = RobotMultiTCNRegression(taskdef,
                    model=model,
                    **kwargs)
        elif model == 'lstm_regression':
            model_instance = RobotMultiLSTMRegression(taskdef,
                    model=model,
                    **kwargs)
        elif model == "sample":
            model_instance = RobotMultiTrajectorySampler(taskdef,
                    model=model,
                    **kwargs)
        elif model == "autoencoder":
            model_instance = RobotMultiAutoencoder(taskdef,
                    model=model,
                    **kwargs)
        elif model == "hierarchical":
            model_instance = RobotMultiHierarchical(taskdef,
                    model=model,
                    **kwargs)
        elif model == "unsupervised":
            model_instance = RobotMultiUnsupervised(taskdef,
                    model=model,
                    **kwargs)
        elif model == "unsupervised":
            model_instance = RobotMultiUnsupervised(taskdef,
                    model=model,
                    **kwargs)
    
    # If we did not create a model then die.
    if model_instance is None:
        raise NotImplementedError("Combination of model %s and features %s" + \
                                  " is not currently supported by CTP.")

    return model_instance

def GetModels():
    return [None,
            "gan", # simple GAN model for generating images
            "ff_regression", # regression model; just a dense net
            "tcn_regression", # ff regression model with a TCN
            "lstm_regression", # lstm regression model
            "sample", # sampler NN to generate trajectories
            "autoencoder", # autoencoder image test
            "hierarchical", # hierarchical policy for planning
            "unsupervised", # paper implementation for unsupervised method
            "unsupervised1", # alternative implementation
            ]
