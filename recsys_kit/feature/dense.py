from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from .constants import DENSE_FILL_VALUE


DENSE_ENCODER_MAP = {
    "dense_imputer": SimpleImputer(strategy="constant", fill_value=DENSE_FILL_VALUE),
    "robust-scale": preprocessing.RobustScaler(quantile_range=(10.0, 90.0)),
    "min-max": preprocessing.MinMaxScaler(feature_range=(0, 1))
}