from torch.utils.data import DataLoader, SequentialSampler
from .collate import (seq_to_ccs_graph, collate_fn_factory_ccs, collate_fn_features)
from .utils import Timer
import pandas as pd

class SIHG4SR_DataLoader(DataLoader):
    def __init__(self, sessions: pd.DataFrame, batch_size, num_workers, order, attent_longest_view, enable_sampler = False, time_feature = "purchase_date", feature_list = ["item_id", "y", "session_id", "feature", "feature_cat", "wf"], recent_n_month = -1, sort = False, shuffle=False, **kwargs):
        processed = sessions
        if recent_n_month != -1:
            max_time = processed[time_feature].max()
            divider = max_time - pd.to_timedelta(int(31 * recent_n_month), unit='d')
            processed = processed[processed[time_feature] > divider]

        if sort:
            processed['index'] = processed.index
            processed = processed.sort_values([time_feature, 'index']).drop(columns=['index'])
            
        dataset = processed[feature_list].to_numpy().tolist()
        
        collate_fn = collate_fn_factory_ccs((seq_to_ccs_graph, ), order=order, attent_longest_view = attent_longest_view)
        
        if enable_sampler:
            sampler = SequentialSampler(dataset)
        else:
            sampler = None
        
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn = collate_fn, pin_memory=True, sampler=sampler)
    