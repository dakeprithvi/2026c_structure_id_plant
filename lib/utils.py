import os
import pickle
import numpy as np
import jax
# Enable 64-bit precision globally. Pickled JAX arrays are stored as float64,
# but JAX silently downcasts them to float32 on deserialization unless x64 is
# enabled. Setting it here (PickleTool's own module) guarantees x64 is on for
# every script that loads/saves pickles via PickleTool.
jax.config.update("jax_enable_x64", True)

class PickleTool:
    """ Class which contains a few static methods for saving and
        loading pkl data files conveniently. """

    @staticmethod
    def load(filename, type='read'):
        """Wrapper to load data."""
        if type == 'read':
            with open(filename, "rb") as stream:
                return pickle.load(stream)
    
    @staticmethod
    def save(data_object, filename):
        """Wrapper to pickle a data object."""
        with open(filename, "wb") as stream:
            pickle.dump(data_object, stream)


def save_file(filename, dir_name, obj=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir   = os.path.join(script_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    if filename.endswith('.pickle'):
        PickleTool.save(obj, path)
    elif filename.endswith('.pdf'):
        obj.savefig(path)
    elif filename.endswith('.txt'):
        return open(path, 'w')
    
    return path

def load_pickle(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    local_path  = os.path.join(script_dir, filename)
    pickle_path = os.path.join(script_dir, 'pickles', filename)

    if os.path.exists(local_path):
        return PickleTool.load(local_path, 'read')
    elif os.path.exists(pickle_path):
        return PickleTool.load(pickle_path, 'read')
    else:
        raise FileNotFoundError(f"'{filename}' not found in current dir or pickles/")
