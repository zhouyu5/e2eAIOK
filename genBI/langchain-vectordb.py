
import os
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
import logging
import time
import pickle

logger = logging.getLogger('langchain')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


######################################################################################
emb_model_name = "/root/qyao/gitspace/llm-on-ray/model_store/sqlcoder"
db_filename = 'retriever.db'

logger.info(f'embedding model: {emb_model_name}')

######################################################################################
start_time = time.time()

embedding = HuggingFaceEmbeddings(
    model_name = emb_model_name,
)

documents = JSONLoader(file_path='schema.jsonl', jq_schema='.meta_data', text_content=False, json_lines=True).load()
db = FAISS.from_documents(documents=documents, embedding=embedding)
retriever = db.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'lambda_mult': 1})


with open(db_filename, 'wb') as f:
    pickle.dump(retriever, f)

time_taken = time.time() - start_time
logger.info(f'index time takes: {time_taken}s')