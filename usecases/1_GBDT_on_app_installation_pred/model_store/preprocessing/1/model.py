import json
import numpy as np
import pandas as pd
import triton_python_backend_utils as pb_utils
import sys
sys.path.append('/src')
from app.dataprep.data_prepare import DataPrep

class TritonPythonModel:
    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "output__0"
        )
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        self.dataprep = DataPrep("/feature_store")
        self.dataprep.prepare_features()

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(
                request, "input__0"
            )
            
            feat_arr = self.dataprep.transform(in_0.as_numpy())
            
            out_tensor_0 = pb_utils.Tensor(
                "output__0", feat_arr.astype(self.output0_dtype)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        print("Cleaning up...")