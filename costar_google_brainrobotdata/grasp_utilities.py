import os
import json
import datetime


def mkdir_p(path):
    """Create the specified path on the filesystem like the `mkdir -p` command

    Creates one or more filesystem directory levels as needed,
    and does not return an error if the directory already exists.
    """
    # http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """ Apply a timestamp to the front of a filename description.

    see: http://stackoverflow.com/a/5215012/99379
    """
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


def load_hyperparams_json(hyperparams_file, fine_tuning=False, fine_tuning_learning_rate=0.001):
    """ Load hyperparameters from a json file
    """
    kwargs = {}
    hyperparams = None
    if hyperparams_file is not None and hyperparams_file:
        with open(hyperparams_file, mode='r') as hyperparams:
            kwargs = json.load(hyperparams)
            hyperparams = kwargs
    if fine_tuning:
        kwargs['trainable'] = True
        kwargs['learning_rate'] = fine_tuning_learning_rate
        # TODO(ahundt) should we actually write the fine tuning settings out to the hyperparams log?
        # hyperparams = kwargs
    return hyperparams, kwargs


def is_sequence(arg):
    """Returns true if arg is a list or another Python Sequence, and false otherwise.

        source: https://stackoverflow.com/a/17148334/99379
    """
    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))