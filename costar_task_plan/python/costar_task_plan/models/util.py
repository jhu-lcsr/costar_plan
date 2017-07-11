
from multi_gan_model import RobotMultiGAN

def MakeModel(features, model, *args, **kwargs):
    '''
    This function will create the appropriate neural net based on images and so
    on.
    '''
    # set up some image parameters
    if features in ['rgb', 'multi']:
        nchannels = 3
    elif features in ['depth']:
        nchannels = 1

    model_instance = None
    if model == 'gan':
        if features in ['rgb','depth']:
            pass
        elif model == 'gan' and features == 'multi':
            # This model will handle features 
            return RobotMultiGAN()
    elif model == 'dense':
        # just create some dense layers.
        pass
    
    # If we did not create a model then die.
    if model_instance is None:
        raise NotImplementedError("Combination of model %s and features %s" + \
                                  " is not currently supported by CTP.")

    return model_instance

def GetModels():
    return [None, "gan", "dense"]
