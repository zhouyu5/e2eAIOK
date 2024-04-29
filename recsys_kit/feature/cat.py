from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer

import numpy as np

from .constants import CAT_FILL_VALUE, CAT_UNKNOWN_VALUE

# Categorify notes:
# Categorify op maps nulls to 1, OOVs to 2, automatically. 
# We reserve 0 for padding the sequence features. 
# The encoding of each category starts from 3.

CAT_ENCODER_MAP = {
    "cat_imputer": SimpleImputer(strategy="constant", fill_value=CAT_FILL_VALUE),
    "categorify": preprocessing.OrdinalEncoder(
        dtype=np.int64,
        handle_unknown='use_encoded_value',
        unknown_value=CAT_UNKNOWN_VALUE,
        encoded_missing_value=CAT_UNKNOWN_VALUE,
    ),
}


