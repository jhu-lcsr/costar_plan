
from multi_gan_model import RobotMultiGAN
from multi_regression_model import RobotMultiFFRegression
from multi_trajectory_sampler import RobotMultiTrajectorySampler
from multi_autoencoder_model import RobotMultiAutoencoder

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
            model_instance = RobotMultiGAN(taskdef, **kwargs)
    elif model == 'ff_regression':
        if features in ['rgb','depth']:
            # make a nice little convnet
            pass
        elif features == 'multi':
            model_instance = RobotMultiFFRegression(taskdef, **kwargs)
    elif model == "sample":
        if features in ['rgb','depth']:
            pass
        elif features == 'multi':
            model_instance = RobotMultiTrajectorySampler(taskdef, **kwargs)
    elif model == "autoencoder":
        if features in ['rgb','depth']:
            pass
        elif features == 'multi':
            model_instance = RobotMultiAutoencoder(taskdef, **kwargs)
    
    # If we did not create a model then die.
    if model_instance is None:
        raise NotImplementedError("Combination of model %s and features %s" + \
                                  " is not currently supported by CTP.")

    return model_instance

def GetModels():
    return [None, "gan", "ff_regression", "lstm_regression", "sample",
            "autoencoder"]
