import time
import sys
import numpy as np
import pandas as pd
import gradio as gr
import io
import subprocess

def infer(inputs_df):
    inputs_df.to_csv("tmp.csv")
    result = subprocess.run("curl http://127.0.0.1:8080/predictions/sihg4sr -T tmp.csv", shell=True, capture_output=True, text=True)
    output = result.stdout.replace("\n","").replace("[","").replace("],","\n")[:-2]
    df = pd.read_csv(io.StringIO(output), sep=",")
    return (df)


def start_gradio(df):
    print(' ########################## starting Gradio Server ########################## ')
    headers = df.columns.tolist()
    title = "RecSys SIHG4SR model Serving Demo"
    description = "This model predicts the top 5 purchased items when given a session in RecSys. Drag and drop any slice from dataset or edit values as you wish in below dataframe component."
    inputs = [gr.Dataframe(headers = headers, row_count = (df.shape[0], "dynamic"), col_count=(len(headers),"fixed"), label="Input Data", interactive=True)]
    outputs = [
        gr.Dataframe(row_count = (5, "dynamic"), col_count=(3, "fixed"), label="top 5 purchase item", headers=["session_id", "item_id", "rank"])
    ]
    gr.Interface(
        infer, inputs = inputs, outputs = outputs, title = title,
        description = description, examples=[df], cache_examples=False
    ).launch()


if __name__ == '__main__':
    data = pd.read_csv("mtstill_test.csv")[["session_id","item_id","date"]]
    start_gradio(data)

