from os import mkdir
from os.path import exists
import datetime

def mkdir_safely(path):
    if not exists(path):
        mkdir(path)

def get_datetime_str():
    return '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
