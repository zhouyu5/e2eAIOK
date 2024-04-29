#!/bin/bash


time python -m recsys_kit.feature.fit_feature config/train_config.json
    
time python -m recsys_kit.feature.transform_feature config/train_config.json