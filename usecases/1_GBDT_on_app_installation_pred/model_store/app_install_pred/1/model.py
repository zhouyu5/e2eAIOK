import io, json
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        
    def preprocess(self, request):
        in_0 = pb_utils.get_input_tensor_by_name(
            request, "input__0"
        )
        preprocess_request = pb_utils.InferenceRequest(
            model_name="preprocessing",
            requested_output_names=["output__0"],
            inputs=[in_0],
        )

        response = preprocess_request.exec()
        if response.has_error():
            raise pb_utils.TritonModelException(response.error().message())
        else:
            preprocess_feat = pb_utils.get_output_tensor_by_name(
                response, "output__0"
            )
            return preprocess_feat
                
    def postprocess(self, pred_result):
        in_0 = pb_utils.Tensor("input__0", pred_result.as_numpy())
        postprocess_request = pb_utils.InferenceRequest(
            model_name="postprocessing",
            requested_output_names=["output__0"],
            inputs=[in_0],
        )
        response = postprocess_request.exec()
        if response.has_error():
            raise pb_utils.TritonModelException(response.error().message())
        else:
            predict_label = pb_utils.get_output_tensor_by_name(
                response, "output__0"
            )
        
            return predict_label
    
    def predict(self, preprocess_out):
        in_0 = pb_utils.Tensor("input__0", preprocess_out.as_numpy(),)
        pred_request = pb_utils.InferenceRequest(
            model_name="prediction",
            requested_output_names=["output__0"],
            inputs=[in_0],
        )
        response = pred_request.exec()
        if response.has_error():
            raise pb_utils.TritonModelException(response.error().message())
        else:
            pred_result = pb_utils.get_output_tensor_by_name(
                response, "output__0"
            )
            return pred_result
    
    def generate_inference_response(self, postprocess_out):
        return pb_utils.InferenceResponse(
            output_tensors=[postprocess_out]
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            preprocess_out = self.preprocess(request)
            prediction = self.predict(preprocess_out)
            postprocess_out = self.postprocess(prediction)
            inference_response = self.generate_inference_response(postprocess_out)
            responses.append(inference_response)
        return responses

    def finalize(self):
        print("Cleaning up...")