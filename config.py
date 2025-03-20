"""This module handles the configuration and data loading for the application."""
import json
import os
import shutil
import joblib
from rank_bm25 import BM25Okapi
import numpy as np
from dotenv import load_dotenv
from src.graph import TableGraph
# 加载 .env 文件
load_dotenv()

import llms # pylint: disable=C0413
import embeddings # pylint: disable=C0413

os.environ['DEBUG'] = '0'
os.environ['SHOW_LLM_INPUT_MSG'] = '1'
os.environ['ENABLE_TOKENIZER_COUNT'] = '0'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = ROOT_DIR + '/assets'
CACHE_DIR = ROOT_DIR + '/cache'
OUTPUT_DIR = ROOT_DIR + '/output'
SUBMIT_DIR = ROOT_DIR + '/submit'
SUBMIT_FILE = SUBMIT_DIR + '/Eva_Now_result.json'
QUESTION_FILE = ROOT_DIR + '/assets/金融复赛a榜.json'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(SUBMIT_DIR):
    os.makedirs(SUBMIT_DIR)

schema = []
if os.path.exists(CACHE_DIR + '/schema.json'):
    with open(CACHE_DIR + '/schema.json', 'r', encoding='utf-8') as file:
        schema = json.loads(file.read())

db_table = {}
if os.path.exists(CACHE_DIR + '/db_table.json'):
    with open(CACHE_DIR + '/db_table.json', 'r', encoding='utf-8') as file:
        db_table = json.loads(file.read())

table_relations = TableGraph()
if os.path.exists(CACHE_DIR + '/table_relations.json'):
    table_relations = TableGraph.load_from_file(CACHE_DIR + '/table_relations.json')

all_question = []
if not os.path.exists(OUTPUT_DIR + '/all_question.json'):
    if os.environ.get('QUESTION_FILE'):
        shutil.copy(os.environ['QUESTION_FILE'], OUTPUT_DIR + '/all_question.json')
    elif os.path.exists(QUESTION_FILE):
        shutil.copy(QUESTION_FILE, OUTPUT_DIR + '/all_question.json')
if os.path.exists(OUTPUT_DIR + '/all_question.json'):
    with open(OUTPUT_DIR + '/all_question.json', 'r', encoding='utf-8') as file:
        all_question = json.load(file)

sql_template = []
if os.path.exists(CACHE_DIR+"/sql_template.json"):
    with open(CACHE_DIR+"/sql_template.json", "r", encoding="utf-8") as f:
        tmp_list = json.load(f)
        sql_template = [f"- {item[0]}\n{item[1]}" for item in tmp_list if len(item) > 1]

sql_template_vectors = []
if os.path.exists(CACHE_DIR+"/sql_template_vectors.npy"):
    sql_template_vectors = np.load(CACHE_DIR+"/sql_template_vectors.npy")

column_vectors = []
column_vector_names = []
if os.path.exists(CACHE_DIR + '/column_vectors.npy'):
    column_vectors = np.load(CACHE_DIR + '/column_vectors.npy')
    for t in schema:
        table_name = t["table_name"]
        for c in t["columns"]:
            column_vector_names.append(table_name + "." + c["name"])
column_bm25: BM25Okapi | None = None
if os.path.exists(CACHE_DIR + '/column_bm25.pkl'):
    column_bm25: BM25Okapi = joblib.load(CACHE_DIR + '/column_bm25.pkl')

table_index = {}
for idx, t in enumerate(schema):
    table_index[t["table_name"]] = t

column_index = {}
for t in schema:
    table_name = t["table_name"]
    column_index[table_name] = {}
    for idx, c in enumerate(t["columns"]):
        column_index[table_name][c["name"]] = c

enum_columns = {}
for table_name, cols in column_index.items():
    for col_name, col in cols.items():
        if col["enum_desc"] != "":
            if table_name not in enum_columns:
                enum_columns[table_name] = {}
            enum_columns[table_name][col_name] = col["enum_desc"]

table_snippet = "有以下数据表:\n"
for t in schema:
    # table_snippet += f"数据表[{t['table_desc']}({t['table_name']})]: {t['table_remarks']}\n"
    table_snippet += f"{t['table_desc']};"

import_column_names = {
    # "InnerCode", "CompanyCode", "SecuCode",
    # "ChiNameAbbr",
    # "ChiSpelling",
    # "ConceptCode",
    # "IndustryCode",
    # "FirstIndustryCode", "SecondIndustryCode",
    # "ThirdIndustryCode", "FourthIndustryCode", "IndustryNum",
    # "IndexCode", "IndexInnerCode", "SecuInnerCode",
    # "FirstPublDate",
}

MAX_ITERATE_NUM = 15
MAX_SQL_RESULT_ROWS = 30
MAX_CONCURRENT_THREADS = 10 # if -1, then auto set to the number of cores

START_INDEX = [0, 0]  # 起始下标 [team_index, question_idx]
END_INDEX = [0, 0]
SAVE_FILE_SUBFIX = "_f2ad09dc360c4249ac273521a378104f_J64xUpwA4TAc_v3.1.3"

FLAG_IGNORE_CACHE = False # 是否忽略中间缓存的结果并重跑
ENABLE_LLM_SEARCH_DB = True # 是否启用LLM搜索数据库
ENABLE_VECTOR_SEARCH_DB = True # 是否启用向量搜索数据库
ENABLE_BATCH_MODE = False # 是否启用按问题组批量模式

llm_plus = llms.llm_glm_4_plus
embed = embeddings.glm_embedding_3
