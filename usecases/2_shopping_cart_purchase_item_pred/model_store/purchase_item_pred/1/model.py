import io, json
import numpy as np
import pandas as pd
import triton_python_backend_utils as pb_utils
import sys
sys.path.append('/src')
from app.dataprep.data_prepare import DataPrep
from app.serve.serve import ServeApp

import pathlib
cur_path = pathlib.Path(__file__).parent.resolve()



class TritonPythonModel:

    def initialize(self, args):
        self.dataprep = DataPrep("/feature_store")
        self.dataprep.prepare_features()
        self.input_feature_names = self.dataprep.input_feature_names
        self.serve = ServeApp(model_dir=cur_path, feat_dir = "/feature_store", candidate_dir=cur_path)
        
    def prepare_input(self, request):
        in_0 = pb_utils.get_input_tensor_by_name(
            request, "input__0"
        ).as_numpy()
        print(in_0.shape)
        in_0 = [np.squeeze(in_0).tolist()] if in_0.shape==(1, 1) else np.squeeze(in_0).tolist()
        
        in_1 = pb_utils.get_input_tensor_by_name(
            request, "input__1"
        ).as_numpy()
        in_1 = [np.squeeze(in_1).tolist()] if in_1.shape==(1, 1) else np.squeeze(in_1).tolist()
        
        in_2 = pb_utils.get_input_tensor_by_name(
            request, "input__2"
        ).as_numpy()
        in_2 = [np.squeeze(in_2).tolist()] if in_2.shape==(1, 1) else np.squeeze(in_2).tolist()
        in_2 = [s.decode() for s in in_2]
        
        in_data = dict(zip(self.input_feature_names, [in_0, in_1, in_2]))
        in_df = pd.DataFrame(in_data)
        return in_df
    
    def generate_inference_response(self, postprocess_out):
        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(
                    "session_id",
                    np.array([i[0] for i in postprocess_out])
                ),
                pb_utils.Tensor(
                    "recommended_items",
                    np.array([i[1] for i in postprocess_out])
                )
            ]
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get INPUT0
            in_data = self.prepare_input(request)
            processed_data = self.dataprep.transform(in_data)
            out =  self.serve.predict(processed_data)            
            inference_response = self.generate_inference_response(out)
            responses.append(inference_response)
        return responses

    def finalize(self):
        print("Cleaning up...")