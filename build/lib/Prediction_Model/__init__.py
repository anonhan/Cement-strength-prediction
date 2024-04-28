import os
from Prediction_Model.config.config import PACKAGE_ROOT

with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as f:
    __version__ = f.read().strip()