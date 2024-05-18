import os
import shutil
from Prediction_Model.config.config import DATA_DIR


def empty_dirs():
    dirs = os.listdir(DATA_DIR)
    required_dirs = ['Prediction_Batch_Files','Training_Batch_Files','cement_strength_08012020_120021.csv']
    for dir_name in dirs:
        if dir_name not in required_dirs and os.path.exists(os.path.join(DATA_DIR,dir_name)):
            shutil.rmtree(os.path.join(DATA_DIR,dir_name))

