from keras.models import model_from_config
from keras.layers.recurrent import Recurrent

# From Keras-RL and modified
def clone_model(model, custom_objects={}):
    # Requires Keras 1.0.7 since get_config has breaking changes.
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    return clone

# another tool taken from Keras-RL
def is_recurrent(model):
  for layer in model.layers:
    if isinstance(layer, Recurrent):
        return True
  return False

# is the RNN stateful?
def is_stateful(model):
  return model.stateful
  #for layer in model.layers:
  #  if layer.stateful:
  #    return True
  #return False
