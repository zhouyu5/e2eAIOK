from enum import Enum


SAMPLE_NUM_TO_DEBUG = 10000

CAT_FILL_VALUE = -2
CAT_UNKNOWN_VALUE = -1
DENSE_FILL_VALUE = 0

DENSE_FEAT_PREFIX = 'dense_enc'
CAT_FEAT_PREFIX = 'cat_enc'

CAT_INDEX_SHIFT = 3


class FeatureTypeTags(Enum):
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    SEQUENCE = "sequence"
    TEXT = "text"
    TIME = "time"


class FeatureGroupTags(Enum):
    USER = "user"
    ITEM = "item"
    INTREACT = "interact"
    CONTEXT = "context"
    LABEL = "label"


class FeatureMeaningTags(Enum):
    USER_ID = "user_id"
    ITEM_ID = "item_id"
    SESSION_ID = "session_id"
    EMBEDDING = "embedding"
    UNKNOWN = "unknown"