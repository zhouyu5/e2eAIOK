#! /bin/bash
python -u src/dataprep/data_prepare.py --data_dir dataset/ --feat_dir feat/ --save_dir output/ --predict

# inference local
python -u src/serve/serve.py
