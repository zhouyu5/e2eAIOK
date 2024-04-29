import io, json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "output__0"
        )

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):
        output0_dtype = self.output0_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(
                request, "input__0"
            )

            input0 = in_0.as_numpy()
            #output0 = [[f"identified as install, probability is {input0[i][1]}"] if input0[i][1] > 0.5 else [f"identified as not install, probability is {input0[i][1]}"] for i in range(input0.shape[0])]
            output0 = [[input0[i][1]] for i in range(input0.shape[0])]
            output0 = np.array(output0)

            out_tensor_0 = pb_utils.Tensor(
                "output__0", output0.astype(output0_dtype)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")