
import os
import json
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback
import logging
import time
import pickle

logger = logging.getLogger('langchain')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

######################################################################################

os.environ['OPENAI_API_KEY'] = "EMPTY"
os.environ['OPENAI_API_BASE'] = "http://127.0.0.1:8000/v1/"
model = "sqlcoder-v2"
db_filename = 'retriever.db'
emb_model_name = "/root/qyao/gitspace/llm-on-ray/model_store/sqlcoder"
top_k_table = 1
query = "How many distinct actors last names are there?"
logger.info(f'query model:{model}, embedding model: {emb_model_name}, question: {query}')



llm = ChatOpenAI(
    model=model,
    max_tokens=256,
    temperature=0.5,
    model_kwargs={"top_p": 0.3},
)

with open(db_filename, 'rb') as f:
    db = pickle.load(f)
retriever = db.as_retriever(search_type='mmr', search_kwargs={'k': top_k_table, 'lambda_mult': 1})

######################################################################################
start_time = time.time()

matched_documents = retriever.get_relevant_documents(query=query)

matched_tables = []

for document in matched_documents:
    page_content = document.page_content
    matched_tables.append(page_content)


def generate_prompt(schema, question):
    messages = [] 

    template = """### Instructions:
    Your task is convert a question into a SQL query, given a MySQL database schema.
    Adhere to these rules:
    - **Deliberately go through the question and database schema word by word** to appropriately answer the question
    - **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
    - When creating a ratio, always cast the numerator as float
    - Use LIKE instead of ilike
    - Only generate the SQL query, no additional text is required
    - Generate SQL queries for MySQL database

    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    messages.append(system_message_prompt)

    human_template = """### Input:
    Generate a SQL query that answers the question `{question}`.
    This query will run on a database whose schema is represented in this string:
    {schema}

    ### Response:
    Based on your instructions, here is the SQL query I have generated to answer the question `{question}`:
    ```sql"""
    human_message = HumanMessagePromptTemplate.from_template(human_template)
    messages.append(human_message)

    chat_prompt = ChatPromptTemplate.from_messages(messages)

    return chat_prompt.format_prompt(
        question=question,
        schema=schema
    ).to_messages()

message = generate_prompt(matched_tables, query)

logger.info(f'message: {message}')

with get_openai_callback() as cb:
    response = llm.invoke(message)
    time_taken = time.time() - start_time
    logger.info(f'the matched schema: {matched_tables}')
    logger.info('tracking token usage')
    logger.info(cb)

logger.info(f'response: {response}')
logger.info(f'query time takes: {time_taken}s')