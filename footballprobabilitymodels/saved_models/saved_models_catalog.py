import os
from enum import Enum
from joblib import dump, load

THIS_FILE_PATH = os.path.join(os.path.dirname(__file__))


class SavedModelsCatalogE(Enum):

    def __new__(cls, name, path):
        object_new = object.__new__(cls)
        object_new._value_ = name
        object_new.path = path
        object_new.fpm_model = load(path)
        return object_new

    TF_SEQUENTIAL = ('TF_SEQUENTIAL', os.path.join(THIS_FILE_PATH, "FPM_TF_SEQUENTIAL"))
    # XGBOOST = ('XGBOOST', os.path.join(THIS_FILE_PATH, "FPM_XGBOOST"))
