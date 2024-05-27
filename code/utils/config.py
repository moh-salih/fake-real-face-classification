import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # utils
ROOT_DIR = os.path.dirname(ROOT_DIR) # code
ROOT_DIR = os.path.dirname(ROOT_DIR) # root of the project



def check_and_mkdir_if_necessary(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path