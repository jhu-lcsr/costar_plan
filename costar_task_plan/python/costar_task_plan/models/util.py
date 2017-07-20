
from multi_gan_model import RobotMultiGAN
from multi_regression_model import RobotMultiFFRegression
from multi_trajectory_sampler import RobotMultiTrajectorySampler
from multi_autoencoder_model import RobotMultiAutoencoder
from multi_hierarchical import RobotMultiHierarchical

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
    if model == 'gan':
        if features in ['rgb','depth']:
            pass
        elif features == 'multi':
            # This model will handle features 
            model_instance = RobotMultiGAN(taskdef, model=model, **kwargs)
    elif model == 'ff_regression':
        if features in ['rgb','depth']:
            # make a nice little convnet
            pass
        elif features == 'multi':
            model_instance = RobotMultiFFRegression(taskdef, model=model, **kwargs)
    elif model == "sample":
        if features in ['rgb','depth']:
            pass
        elif features == 'multi':
            model_instance = RobotMultiTrajectorySampler(taskdef, model=model, **kwargs)
    elif model == "autoencoder":
        if features in ['rgb','depth']:
            pass
        elif features == 'multi':
            model_instance = RobotMultiAutoencoder(taskdef, model=model, **kwargs)
    elif model == "hierarchical":
        if features in ['rgb','depth']:
            pass
        elif features == 'multi':
            model_instance = RobotMultiHierarchical(taskdef,
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
            ]
