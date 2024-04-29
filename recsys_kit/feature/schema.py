from dataclasses import dataclass, field
from typing import Dict, Optional

from .constants import FeatureTypeTags, FeatureGroupTags, FeatureMeaningTags


@dataclass
class ColumnSchema:
    name: str = field(
        metadata={"help": "The name of the column."}
    )
    feature_type_tag: Optional[FeatureTypeTags] = field(
        default=None, 
        metadata={"help": "The feature type of the column."}
    )
    feature_group_tag: Optional[FeatureGroupTags] = field(
        default=None, 
        metadata={"help": "The feature group tag of the column."}
    )
    feature_meaning_tag: Optional[FeatureMeaningTags] = field(
        default=None, 
        metadata={"help": "The feature meaning tag of the column."}
    )
    cardinality: Optional[int] = field(
        default=None, 
        metadata={"help": "The cardinality of the column."}
    )
    min_value: Optional[int] = field(
        default=None, 
        metadata={"help": "The min value of the column."}
    )
    max_value: Optional[int] = field(
        default=None, 
        metadata={"help": "The max value of the column."}
    )
    unique_value: Optional[int] = field(
        default=None, 
        metadata={"help": "The unique value of the column."}
    )
    missing_ratio: Optional[float] = field(
        default=None, 
        metadata={"help": "The missing value ratio of the column."}
    )
    # continuous: std, mean, num_buckets
    # categorical: cat_map_path
    # sequence: seq_length
    properties: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "The properties of the column."}    
    )

    def is_categorical(self):
        return self.feature_type_tag == FeatureTypeTags.CATEGORICAL
    
    def is_sequence(self):
        return self.feature_type_tag == FeatureTypeTags.SEQUENCE
    
