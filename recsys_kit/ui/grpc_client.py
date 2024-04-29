# Create GRPC client instance
import time
import sys
import numpy as np
import pandas as pd
import gradio as gr
import tritonclient.grpc as triton_grpc
from tritonclient import utils as triton_utils
from sklearn.metrics import log_loss, roc_auc_score
import time

# Check to see if server is live and model is loaded
def is_triton_ready(client, MODEL_NAME):
    TIMEOUT = 60
    server_start = time.time()
    while True:
        try:
            if client.is_server_ready() and client.is_model_ready(MODEL_NAME):
                return True
        except triton_utils.InferenceServerException:
            pass
        if time.time() - server_start > TIMEOUT:
            print('Server was not ready before given timeout. Check the logs below for possible issues.')
            return False
        time.sleep(1)


def triton_predict(client, model_name, arr):
    triton_input = triton_grpc.InferInput('input__0', arr.shape, 'FP32')
    triton_input.set_data_from_numpy(arr)
    response = client.infer(model_name, model_version='1', inputs=[triton_input])
    return response.as_numpy('output__0'), response.as_numpy('output__1'), \
        response.as_numpy('output__2'), response.as_numpy('output__3'), response.as_numpy('output__4')


def cal_metric(y_true, y_pred):
    y_true = y_true.T[0]
    y_pred = y_pred.values
    eval_loss = log_loss(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return eval_loss, auc


def infer(inputs_df):
    start = time.time()

    inputs = inputs_df.values
    inputs = np.array(inputs, dtype=np.float32)
    MODEL_NAME = 'lgbm_pipeline_bls'
    HOST = 'localhost'
    PORT = 8001
    client = triton_grpc.InferenceServerClient(url=f'{HOST}:{PORT}')
    if is_triton_ready(client, MODEL_NAME):
        triton_result = triton_predict(client, MODEL_NAME, inputs)

    label_dict = {1: "install", 0: "not install"}
    actual_label_string = np.vectorize(label_dict.get)(triton_result[3])
    display_prediction = np.concatenate((triton_result[4], triton_result[2], actual_label_string), axis=1)
    
    done = time.time()
    elapsed = done - start
    avg_time = round(elapsed / inputs_df.shape[0] * 1000, 1)

    model_pred_df = pd.DataFrame(triton_result[1], columns=model_pred_columns)

    if inputs_df.shape[0] < 10:
        eval_loss, auc = 0, 0
    else:
        eval_loss, auc = cal_metric(triton_result[3], model_pred_df["prob_install"])

    return (
        pd.DataFrame(triton_result[0], columns=output_feature_names), 
        model_pred_df, 
        pd.DataFrame(display_prediction, columns=result_columns), 
        pd.DataFrame(np.array([[avg_time, eval_loss, auc]]), columns=metric_columns)
    )


def start_gradio(df):
    print(' ########################## starting Gradio Server ########################## ')
    headers = df.columns.tolist()
    title = "RecSys LGBM model Serving Demo"
    description = "This model predicts the CTR in RecSys. Drag and drop any slice from dataset or edit values as you wish in below dataframe component."
    inputs = [gr.Dataframe(headers = headers, row_count = (1, "dynamic"), col_count=(len(headers),"fixed"), label="Input Data", interactive=True)]
    outputs = [
        gr.Dataframe(row_count = (1, "dynamic"), col_count=(len(output_feature_names), "fixed"), label="step 1: Process features online", headers=output_feature_names),
        gr.Dataframe(row_count = (1, "dynamic"), col_count=(len(model_pred_columns), "fixed"), label="step 2: Predict probabilities", headers=model_pred_columns),
        gr.Dataframe(row_count = (1, "dynamic"), col_count=(len(result_columns), "fixed"), label="step 3: Predict behaviour based on the probabilities", headers=result_columns),
        gr.Dataframe(row_count = (1, "dynamic"), col_count=(len(metric_columns), "fixed"), label="step 4: Calculate the metric (for loss and auc, only take effct while n > 10)", headers=metric_columns),
    ]
    gr.Interface(
        infer, inputs = inputs, outputs = outputs, title = title,
        description = description, examples=[df.head(1), df.head(3), df.head(256)], cache_examples=False
    ).launch(
        server_name="0.0.0.0",
        server_port=7861,
        debug=True
    )


if __name__ == '__main__':
    input_feature_names = [f'f_{i}' for i in range(80)] + ["is_clicked", "is_installed"]
    output_feature_names = 'dow,f_2,f_3,f_4,f_5,f_6,f_8,f_9,f_10,f_11,f_12,f_13,f_14,f_15,f_16,f_17,f_18,f_19,f_20,f_21,f_22,f_23,f_24,f_25,f_26,f_27,f_28,f_29,f_30,f_31,f_32,f_33,f_34,f_35,f_36,f_37,f_38,f_39,f_40,f_41,f_42,f_43,f_44,f_45,f_46,f_47,f_48,f_49,f_50,f_51,f_52,f_53,f_54,f_55,f_56,f_57,f_58,f_59,f_60,f_61,f_62,f_63,f_64,f_65,f_66,f_67,f_68,f_69,f_70,f_71,f_72,f_73,f_74,f_75,f_76,f_77,f_78,f_79,f_2_CE,f_4_CE,f_6_CE,f_13_CE,f_15_CE,f_18_CE,f_78_CE,f_75_CE,f_50_CE,f_20_CE,f_24_CE,f_3_idx,f_5_idx,f_7_idx,f_8_idx,f_9_idx,f_10_idx,f_11_idx,f_12_idx,f_13_idx,f_14_idx,f_16_idx,f_17_idx,f_18_idx,f_19_idx,f_20_idx,f_21_idx,f_22_idx'.split(',')
    target_label = 'is_installed'
    id_columns = 'f_0'

    model_pred_columns = ["prob_not_install", "prob_install"]
    result_columns = ['record_id', 'predict_install', 'actual_install']
    metric_columns = ['avg_latency (ms)', 'log_loss', 'auc']
    
    data_path = '/apps/dataset/valid/valid.csv'
    feat_sample_df = pd.read_csv(data_path, sep='\t').head(1000)

    start_gradio(feat_sample_df)

