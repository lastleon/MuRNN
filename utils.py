from os import mkdir
from os.path import exists
import datetime
import numpy as np

# just some helper functions

def mkdir_safely(path):
    if not exists(path):
        mkdir(path)

def get_datetime_str():
    return '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())

def roll_and_add_zeros(array):
    shape = array.shape
    if len(shape) >= 2:
        array = np.copy(array)
        array = np.append(array, np.zeros(shape[1:]))
        array = array[int(np.prod(shape[1:])):]
        array = np.reshape(array, shape)

        return array
    else:
        print("Array must have at least 2 dimensions...")
        raise ValueError()