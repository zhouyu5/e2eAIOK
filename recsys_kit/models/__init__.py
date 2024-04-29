from transformers import AutoConfig, AutoModel
from .tf4ctr_model import TF4CTRConfig, TF4CTRModel

AutoConfig.register("tab_tf_ctr", TF4CTRConfig)
AutoModel.register(TF4CTRConfig, TF4CTRModel)