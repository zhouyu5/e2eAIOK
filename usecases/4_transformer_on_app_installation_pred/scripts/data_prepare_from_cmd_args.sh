#!/bin/bash

dev_mode=true


time python -m recsys_kit.feature.fit_feature \
    --dev_mode $dev_mode \
    --dataset_name_or_path nathan0/sharechat \
    --dataset_config_name first_domain \
    --download_data_path download_data/ \
    --dense_columns f_42,f_43,f_44,f_45,f_46,f_47,f_48,f_49,f_50,f_51,f_52,f_53,f_54,f_55,f_56,f_57,f_58,f_59,f_60,f_61,f_62,f_63,f_64,f_65,f_66,f_67,f_68,f_69,f_70,f_71,f_72,f_73,f_74,f_75,f_76,f_77,f_78,f_79 \
    --cat_columns f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10,f_11,f_12,f_13,f_14,f_15,f_16,f_17,f_18,f_19,f_20,f_21,f_22,f_23,f_24,f_25,f_26,f_27,f_28,f_29,f_30,f_31,f_32,f_33,f_34,f_35,f_36,f_37,f_38,f_39,f_40,f_41,is_clicked \
    --label_column is_installed \
    --dense_processor dense_imputer,robust-scale,min-max \
    --cat_processor cat_imputer,categorify \
    --fit_transform_output_type polars \
    --data_pipeline_path processed_data/process_pipe.pkl \
    --feature_config_name_or_path config/feature_config.json
    

time python -m recsys_kit.feature.transform_feature \
    --dev_mode $dev_mode \
    --dataset_name_or_path nathan0/sharechat \
    --dataset_config_name first_domain \
    --download_data_path download_data/ \
    --model_config_name_or_path config/model_config.json \
    --data_pipeline_path processed_data/process_pipe.pkl \
    --fit_transform_output_type polars \
    --preprocess_dataset_path processed_data/sharechat_local 

