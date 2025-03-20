"""
This module provides specific functions for Game Finglm
"""

import os
import copy
# import re
import json
import jieba
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.log import get_logger
from src.agent import Agent
from src.utils import extract_last_sql, extract_last_json
from src.workflow import COLUMN_LIST_MARK
import config
import time

def execute_sql_query(sql: str) -> str:
    """
    Executes an SQL query using the specified API endpoint and returns the result as a string.
    
    Args:
        sql (str): The SQL query to be executed.
    
    Returns:
        str: The result of the SQL query execution.
    """
    debug_mode = os.getenv("DEBUG", "0") == "1"
    sql = sql.replace('\\n', ' ')
    url = "https://comm.chatglm.cn/finglm2/api/query"
    access_token = os.getenv("ZHIPU_ACCESS_TOKEN", "")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    logger = get_logger()
    logger.info("\n>>>>> 查询sql:\n%s\n", sql)
    if debug_mode:
        print(f"\n>>>>> 查询ql:\n{sql}")
    
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.post(url, headers=headers, json={"sql": sql, "limit": config.MAX_SQL_RESULT_ROWS}, timeout=60)
            result = response.json()
            
            if "success" in result and result["success"] is True:
                data = json.dumps(result["data"], ensure_ascii=False)
                logger.info("查询结果:\n%s\n", data)
                if debug_mode:
                    print(f"查询结果:\n{data}")
                return data
            
            logger.info("查询失败: %s\n", result["detail"])
            if debug_mode:
                print("查询失败:" + result["detail"])
            
            if "Commands out of sync" in result["detail"]:
                raise SyntaxError("不能同时执行多组SQL: "+result["detail"])
            
            # 非超时错误直接抛出异常，不重试
            raise RuntimeError(result["detail"])
            
        except requests.exceptions.Timeout as exc:
            # 只有超时错误才重试
            retry_count += 1
            logger.info("请求超时，无法执行SQL查询，第 %d/%d 次重试", retry_count, max_retries)
            print(f"请求超时，无法执行SQL查询，第 {retry_count}/{max_retries} 次重试")
            
            if retry_count >= max_retries:
                raise RuntimeError("执行SQL查询超时，已重试最大次数，请优化SQL后重试。") from exc
            
            # 等待一段时间后重试
            wait_time = retry_count  # 1秒，2秒，3秒...
            time.sleep(wait_time)
    
    # 正常情况下不会执行到这里，但为了代码的完整性
    raise RuntimeError("执行SQL查询失败，已重试最大次数。")

def keep_db_column_info(agent: Agent, messages: dict) -> None:
    """Stores knowledge from messages into the agent."""
    for msg in messages:
        if COLUMN_LIST_MARK in msg["content"]:
            agent.add_system_prompt_kv(
                kv = {"Known Database Structure": msg["content"]}
            )

def extract_and_execute_sql(message: str) -> str:
    """
    Extracts SQL from a message and executes it, returning the result.
    
    Args:
        message (str): The message containing the SQL query.
    
    Returns:
        str: The result of the SQL query execution.
    """
    sql = extract_last_sql(
        query_string=message,
        block_mark="sql",
    )
    if sql is None:
        if 'SELECT' in message:
            raise RuntimeError("请把sql写到代码块```sql```中")
        else:
            return message
    result = execute_sql_query(sql=sql)
    return f"{message}\n执行SQL:\n{sql}查询结果是:\n{result}"

def get_constant_column_list(table_column: dict) -> list:
    """
    Retrieves a list of basic columns for constant tables based on the provided table column data.

    Args:
        table_column (dict): A dictionary containing table names as keys and their columns as values.

    Returns:
        list: A list of dictionaries, each containing the table name and its corresponding columns that are part of the constant tables.
    """
    constant_tables = {
        "constantdb.secumain": {
            "InnerCode", "CompanyCode", "SecuCode", "ChiName",
            "ChiNameAbbr", "EngName", "EngNameAbbr", "SecuAbbr",
        },
        "constantdb.hk_secumain": {
            "InnerCode", "CompanyCode", "SecuCode", "ChiName",
            "ChiNameAbbr", "EngName", "EngNameAbbr", "SecuAbbr",
            "FormerName",
        },
        "constantdb.us_secumain": {
            "InnerCode", "CompanyCode", "SecuCode", "ChiName",
            "EngName", "SecuAbbr",
        },
        "constantdb.ct_systemconst": {
            "LB", "LBMC", "MS", "DM"
        },
        "constantdb.lc_areacode": {
            "AreaInnerCode", "ParentNode", "IfEffected", "AreaChiName",
            "ParentName", "AreaEngName", "AreaEngNameAbbr", "FirstLevelCode",
            "SecondLevelCode"
        },
        "astockindustrydb.lc_conceptlist": {
            "ClassCode", "ClassName", "SubclassCode", "SubclassName",
            "ConceptCode", "ConceptName", "ConceptEngName",
        },
    }
    column_lists = []
    for table, cols in constant_tables.items():
        _, table_name = table.split('.')
        col_list = {
            "表名": table,
            "表字段": [],
        }
        for col in table_column[table_name]:
            if col["column"] in cols:
                col_list["表字段"].append(col)
        column_lists.append(col_list)
    return column_lists

def ajust_org_question(question: str) -> str:
    # if "合并报表调整后" in question:
    #     question = question.replace("合并报表调整后", "合并报表")
    return question

def query_company(name: str) -> str:
    if name in ["公司", "基金", "有限公司", "CN", "A股", "港股", "美股", "该公司", "上市公司", "下属公司", "公司股东", "银行", "股票", "证券公司"]:
        return "[]"
    if name == "":
        return "[]"
    sql = f"""SELECT 'constantdb.secumain' AS TableName, InnerCode, CompanyCode,
    ChiName, EngName, SecuCode, ChiNameAbbr, EngNameAbbr, SecuAbbr, ChiSpelling
FROM constantdb.secumain 
WHERE SecuCode = '{name}'
   OR ChiName LIKE '%{name}%'
   OR ChiNameAbbr LIKE '%{name}%'
   OR EngName LIKE '%{name}%'
   OR EngNameAbbr LIKE '%{name}%'
   OR SecuAbbr LIKE '%{name}%'
   OR ChiSpelling LIKE '%{name}%'
UNION ALL
SELECT 'constantdb.hk_secumain' AS TableName, InnerCode, CompanyCode,
ChiName, EngName, SecuCode, ChiNameAbbr, EngNameAbbr, SecuAbbr, ChiSpelling
FROM constantdb.hk_secumain 
WHERE SecuCode = '{name}'
   OR ChiName LIKE '%{name}%'
   OR ChiNameAbbr LIKE '%{name}%'
   OR EngName LIKE '%{name}%'
   OR EngNameAbbr LIKE '%{name}%'
   OR SecuAbbr LIKE '%{name}%'
   OR FormerName LIKE '%{name}%'
   OR ChiSpelling LIKE '%{name}%'
UNION ALL
SELECT 'constantdb.us_secumain' AS TableName, InnerCode, CompanyCode,
ChiName, EngName, SecuCode, null as ChiNameAbbr, null as EngNameAbbr, SecuAbbr, ChiSpelling
FROM constantdb.us_secumain 
WHERE SecuCode = '{name}'
   OR ChiName LIKE '%{name}%'
   OR EngName LIKE '%{name}%'
   OR SecuAbbr LIKE '%{name}%'
   OR ChiSpelling LIKE '%{name}%';"""
    result = execute_sql_query(sql)
    if result == "[]":
        sql = f"""SELECT 'astockbasicinfodb.lc_stockarchives' AS TableName, 
       CompanyCode, 
       ChiName, 
       NULL as EngName, 
       NULL as EngNameAbbr, 
       AShareAbbr, 
       AStockCode, 
       BShareAbbr, 
       BStockCode, 
       HShareAbbr, 
       HStockCode, 
       CDRShareAbbr, 
       CDRStockCode, 
       ExtendedAbbr
FROM astockbasicinfodb.lc_stockarchives
WHERE CompanyCode = '{name}'
   OR ChiName LIKE '%{name}%'
   OR AShareAbbr LIKE '%{name}%'
   OR BShareAbbr LIKE '%{name}%'
   OR HShareAbbr LIKE '%{name}%'
   OR CDRShareAbbr LIKE '%{name}%'
UNION ALL
SELECT 'hkstockdb.hk_stockarchives' AS TableName, 
       CompanyCode, 
       ChiName, 
       NULL as EngName, 
       NULL as EngNameAbbr, 
       NULL as AShareAbbr, 
       NULL as AStockCode, 
       NULL as BShareAbbr, 
       NULL as BStockCode, 
       NULL as HShareAbbr, 
       NULL as HStockCode, 
       NULL as CDRShareAbbr, 
       NULL as CDRStockCode, 
       NULL as ExtendedAbbr
FROM hkstockdb.hk_stockarchives
WHERE CompanyCode = '{name}'
   OR ChiName LIKE '%{name}%'
UNION ALL
SELECT 'usstockdb.us_companyinfo' AS TableName, 
       CompanyCode, 
       ChiName, 
       EngName, 
       EngNameAbbr, 
       NULL as AShareAbbr, 
       NULL as AStockCode, 
       NULL as BShareAbbr, 
       NULL as BStockCode, 
       NULL as HShareAbbr, 
       NULL as HStockCode, 
       NULL as CDRShareAbbr, 
       NULL as CDRStockCode, 
       NULL as ExtendedAbbr
FROM usstockdb.us_companyinfo
WHERE CompanyCode = '{name}'
   OR ChiName LIKE '%{name}%'
   OR EngName LIKE '%{name}%'
   OR EngNameAbbr LIKE '%{name}%';"""
        result = execute_sql_query(sql)
    return result

def seg_entities(entity: str) -> list[str]:
    stopwords = ['公司', '基金', '管理', '有限', '有限公司']
    seg_list = list(jieba.cut(entity, cut_all=False))
    filtered_seg_list = [word for word in seg_list if word not in stopwords]
    return filtered_seg_list

def extract_company_code(llm_answer: str) -> str:
    """Extracts company codes from the given LLM answer.

    Args:
        llm_answer (str): The answer from the LLM containing company information.

    Returns:
        str: A formatted string with extracted company codes.
    """
    debug_mode = os.getenv("DEBUG", "0") == "1"
    logger = get_logger()
    results = []
    try:
        names_json = extract_last_json(llm_answer)
        if names_json is not None:
            names = json.loads(names_json)
            if not isinstance(names, list):
                raise ValueError("names should be a list")
            for name in names:
                rows = json.loads(query_company(name))
                if len(rows) > 0:
                    info = f"{name}的关联信息有:[" if len(rows) == 1 else f"{name}关联信息有多组:["
                    for idx, row in enumerate(rows):
                        if idx >= 3: # 限制最多遍历三行 因为A股/美股/港股，同一个公司不会命中超过3个结果，超了的话说明关键词取错了
                            # break
                            return ""
                        col_chi = {}
                        if "TableName" in row:
                            col_chi = config.column_index[row["TableName"]]
                        for k, v in dict(row).items():
                            if k == "TableName":
                                info += f"所在数据表是{v};"
                                continue
                            if k in col_chi:
                                info += f"{k}({col_chi[k]['desc']})是{v};"
                            else:
                                info += f"{k}是{v};"
                        info += "]" if idx == len(rows) -1 else "],"
                    results.append(info)

    except Exception as e:
        print(f"extract_company_code::Exception：{str(e)}")
        logger.debug("extract_company_code::Exception：%s", str(e))
    return "\n".join(results)

def foreign_key_hub() -> dict:
    # return {
    #     "constantdb.secumain": {
    #         "InnerCode", "CompanyCode", "SecuCode", "SecuAbbr", "ChiNameAbbr"
    #     },
    #     "constantdb.hk_secumain": {
    #         "InnerCode", "CompanyCode", "SecuCode", "SecuAbbr", "ChiNameAbbr"
    #     },
    #     "constantdb.us_secumain": {
    #         "InnerCode", "CompanyCode", "SecuCode", "SecuAbbr"
    #     },
    # }
    return {}

def db_select_post_process(dbs: list[str]) -> list[str]:
    debug_mode = os.getenv("DEBUG", "0") == "1"
    logger = get_logger()
    required_dbs = {"astockbasicinfodb", "hkstockdb", "usstockdb"}
    present_dbs = set(dbs)
    missing_dbs = required_dbs - present_dbs
    # 确保所有必需的数据库都存在
    if len(missing_dbs) > 0 and len(missing_dbs) != len(required_dbs):
        if debug_mode:
            print("补充选择db: " + json.dumps(list(missing_dbs), ensure_ascii=False))
        logger.debug("补充选择db: %s", json.dumps(list(missing_dbs), ensure_ascii=False))
        dbs.extend(missing_dbs)

    return list(dbs)

def table_select_post_process(db_table_names: list[str]) -> list[str]:
    debug_mode = os.getenv("DEBUG", "0") == "1"
    logger = get_logger()

    required_tables_list = [
        {"astockbasicinfodb.lc_stockarchives", "hkstockdb.hk_stockarchives", "usstockdb.us_companyinfo", "constantdb.lc_areacode"},
        {"astockmarketquotesdb.qt_dailyquote", "hkstockdb.cs_hkstockperformance", "usstockdb.us_dailyquote"},
        {"astockmarketquotesdb.qt_stockperformance", "hkstockdb.cs_hkstockperformance"},
        {"publicfunddb.mf_fundprodname", "publicfunddb.mf_fundarchives"},
        {"astockmarketquotesdb.lc_suspendresumption", "constantdb.hk_secumain", "constantdb.us_secumain"},
        {"astockmarketquotesdb.qt_dailyquote", "astockmarketquotesdb.cs_stockpatterns"},
        {"astockshareholderdb.lc_sharestru", "astockshareholderdb.lc_mainshlistnew"},
    ]

    for required_tables in required_tables_list:
        present_tables = set(db_table_names)
        missing_tables = required_tables - present_tables
        # 确保所有必需的数据库都存在
        if len(missing_tables) > 0 and len(missing_tables) != len(required_tables):
            if debug_mode:
                print("\n补充选择table: " + json.dumps(list(missing_tables), ensure_ascii=False))
            logger.debug("\n补充选择table: %s", json.dumps(list(missing_tables), ensure_ascii=False))
            db_table_names.extend(missing_tables)
    return db_table_names

def find_similar_texts(search_query: str, vectors: list[np.ndarray], texts: list[str], top_p=5, threshold=0.1):
    """
    查找与搜索查询最相似的文本。
    
    Args:
        search_query (str): 要搜索的查询文本
        vectors (list[np.ndarray]): 文本向量列表
        texts (list[str]): 原始文本列表
        top_p (int, optional): 返回的最相似文本数量，默认为5。如果为-1则返回所有结果
        threshold (float, optional): 相似度阈值，默认为0.1。如果相似度小于该阈值，则不返回结果
    Returns:
        tuple: 包含相似度分数和对应文本的元组
    """
    debug_mode = os.getenv("DEBUG", "0") == "1"
    logger = get_logger()
    
    max_retries = 5
    retry_count = 0
    wait_time = 1
    
    while retry_count < max_retries:
        try:
            em, _ = config.embed.create(search_query)
            query_vector = np.array(em)
            # 计算查询文本与所有文本的余弦相似度
            similarities = cosine_similarity(query_vector.reshape(1, -1), vectors)
            # 找到最相似的文本
            if top_p == -1:
                # 如果top_p为-1，返回所有结果
                sorted_indices = np.argsort(similarities[0])[::-1]
            else:
                # 否则只返回前top_p个结果
                sorted_indices = np.argsort(similarities[0])[-top_p:][::-1]
            
            # 过滤掉相似度低于阈值的结果
            threshold_indices = [idx for idx in sorted_indices if similarities[0][idx] >= threshold]
            return similarities[0][threshold_indices], [texts[indice] for indice in threshold_indices]
        except Exception as e:
            retry_count += 1
            if debug_mode:
                print(f"向量嵌入计算超时，第 {retry_count}/{max_retries} 次重试，错误：{str(e)}")
            logger.info("向量嵌入计算超时，第 %d/%d 次重试，错误：%s", retry_count, max_retries, str(e))
            
            if retry_count >= max_retries:
                raise RuntimeError(f"向量嵌入计算失败，已重试最大次数：{str(e)}") from e
            
            # 等待时间增加，实现退避策略
            time.sleep(wait_time)
            wait_time *= 2  # 指数退避

def tokenize_text(text: str) -> list[str]:
    """
    将文本分词并返回分词结果。
    
    Args:
        text (str): 需要分词的文本
    
    Returns:
        list[str]: 分词结果
    """
    # 分词并过滤停用词
    stop_words = {
        '的', '了', '和', '与', '及', '或', '在', '是', '为', '以', '对', '等', '将', '由',
        # '年', '月', '日', '时', '分', '秒', '个', '名', '无', '前', '后', '上', '下', 
        # '左', '右', '中', '内', '外', '其他', '其它', '什么', '多少'
    }
    return [word for word in jieba.cut(text) if word not in stop_words and word.strip()]

def calculate_table_tf_idf_score(question: str, top_p: int = 1) -> list[tuple[str, float, float, float, int, dict]]:
    """
    基于TF-IDF(词频-逆文档频率)计算查询与数据表的相似度得分。
    
    Args:
        question (str): 用户查询文本
        top_p (int): 返回的最相关表的数量
    
    Returns:
        list[tuple[str, float, float, float, int, dict]]: 按加权分数排序的表名和分数元组列表，每个元组包含(表名, 加权分数, 最高分, 总分, 列数, 各列相似度)
    """
    doc_scores = config.column_bm25.get_scores(tokenize_text(question))
    # 对doc_scores直接排序，获取索引和分数
    sorted_indices = np.argsort(doc_scores)[::-1]  # 降序排列的索引
    column_question_scores = [(idx, doc_scores[idx]) for idx in sorted_indices]

    # 按数据表进行分组和加权
    table_scores = {}
    table_max_scores = {}  # 记录每个表的最高分数
    table_top_columns = {}  # 记录每个表的前N个高分列
    table_all_columns = {}  # 新增：存储每个表所有列的相似度
    
    for idx, score in column_question_scores:
        db_name, table_name, column_name = config.column_vector_names[idx].split('.')
        db_table_name = db_name + "." + table_name
        
        # 累加每个表的总分
        if db_table_name not in table_scores:
            table_scores[db_table_name] = 0
            table_max_scores[db_table_name] = score  # 直接使用第一次遇到的分数作为最高分
            table_top_columns[db_table_name] = []
            table_all_columns[db_table_name] = {}  # 初始化所有列相似度字典
        
        table_scores[db_table_name] += score
        table_all_columns[db_table_name][column_name] = score  # 记录每一列的相似度
        
        # 直接保存前3个高分列（因为column_question_scores已经按分数倒序排列）
        if len(table_top_columns[db_table_name]) < 3:
            table_top_columns[db_table_name].append((column_name, score))
    
    # 计算加权分数：结合最大分、前N列平均分和总分
    table_weighted_scores = []
    for db_table_name, total_score in table_scores.items():
        # 前N列的平均分
        top_avg = sum(score for _, score in table_top_columns[db_table_name]) / len(table_top_columns[db_table_name]) if table_top_columns[db_table_name] else 0
        # 加权组合分数
        weighted_score = (
            0.5 * table_max_scores[db_table_name] +  # 最高匹配列的分数
            0.3 * top_avg +                       # 前N列的平均分数
            0.2 * total_score / max(1, config.table_index[db_table_name]['column_count'])  # 整体平均分的小权重
        )
        table_weighted_scores.append((db_table_name, weighted_score))
    
    # 按加权分数排序结果
    sorted_weighted_tables = sorted(table_weighted_scores, key=lambda x: x[1], reverse=True)

    similarity_tables = []
    for db_table_name, weighted_score in sorted_weighted_tables[:top_p]:
        top_cols = [(col, score) for col, score in table_all_columns[db_table_name].items()]
        similarity_tables.append((
            db_table_name,
            weighted_score,
            table_max_scores[db_table_name],
            table_scores[db_table_name],
            config.table_index[db_table_name]['column_count'],
            top_cols,
        ))
    return similarity_tables

def calculate_table_similarity(question: str, top_p: int = 1) -> list[tuple[str, float, float, float, int, dict]]:
    """
    计算查询与数据表的相似度，并返回排序后的结果。
    
    Args:
        question: 查询文本
        top_p: 返回的最相关表的数量
        
    Returns:
        list: 按加权分数排序的表名和分数元组列表，每个元组包含(表名, 加权分数, 最高分, 总分, 列数, 各列相似度)
    """
    column_question_sim = find_similar_texts(
        search_query=question,
        vectors=config.column_vectors,
        texts=config.column_vector_names,
        top_p=-1
    )
    # 按数据表进行分组和加权
    table_scores = {}
    table_max_scores = {}
    table_top_columns = {}
    table_all_columns = {}  # 新增：存储每个表所有列的相似度

    for score, name in zip(*column_question_sim):
        db_name, table_name, column_name = name.split(".")
        db_table_name = db_name + "." + table_name

        # 累加每个表的总分
        if db_table_name not in table_scores:
            table_scores[db_table_name] = 0
            table_max_scores[db_table_name] = score  # 直接使用第一次遇到的分数作为最高分
            table_top_columns[db_table_name] = []
            table_all_columns[db_table_name] = {}  # 初始化所有列相似度字典

        table_scores[db_table_name] += score
        table_all_columns[db_table_name][column_name] = score  # 记录每一列的相似度

        # 直接保存前3个高分列（因为column_question_sim已经按分数倒序排列）
        if len(table_top_columns[db_table_name]) < 3:
            table_top_columns[db_table_name].append((column_name, score))

    # 计算加权分数：结合最大分、前N列平均分和总分
    table_weighted_scores = []
    for db_table_name, total_score in table_scores.items():
        # 前N列的平均分
        top_avg = sum(score for _, score in table_top_columns[db_table_name]) / len(table_top_columns[db_table_name]) if table_top_columns[db_table_name] else 0
        # 加权组合分数
        weighted_score = (
            0.5 * table_max_scores[db_table_name] +  # 最高匹配列的分数
            0.3 * top_avg +                       # 前N列的平均分数
            0.2 * total_score / max(1, config.table_index[db_table_name]['column_count'])  # 整体平均分的小权重
        )
        table_weighted_scores.append((db_table_name, weighted_score))

    # 按加权分数排序结果
    sorted_weighted_tables = sorted(table_weighted_scores, key=lambda x: x[1], reverse=True)

    # 构建结果列表，包含详细信息
    result = []
    for db_table_name, weighted_score in sorted_weighted_tables[:top_p]:
        top_cols = [(col, score) for col, score in table_all_columns[db_table_name].items()]
        # top_cols = table_top_columns[db_table_name]
        # 新增：将所有列的相似度作为结果的一部分返回
        result.append((
            db_table_name,
            weighted_score,
            table_max_scores[db_table_name],
            table_scores[db_table_name],
            config.table_index[db_table_name]['column_count'],
            top_cols,
        ))

    return result

def create_column_description(c):
    """
    创建字段描述
    """
    # return (
    #     f"{c['desc']};" +
    #     (f"枚举类型,枚举值:{c['enum_desc']};" if c['enum_desc'] else "") +
    #     (f"备注:{c['remarks']};" if c['remarks'] else "") +
    #     # ""
    #     (f"数据示例:{c['val']};" if c['enum_desc'] == "" and c['val'] != "" and c['val'] != "NULL" else "")
    # )
    return copy.deepcopy(c)

def print_table_column(table_column: dict) -> str:
    """
    打印表字段描述
    """
    info = f"<table>{table_column['表名']}(即: {table_column['表描述']}),{table_column['表备注']}。\n包含有以下字段:\n"
    for col_name, c in table_column['表字段'].items():
        info += f"<column>{col_name}(即: {c['desc']}):"
        if c['enum_desc'] != "":
            info += f"是枚举类型,枚举值包括:{c['enum_desc']};"
        if c['remarks'] != "":
            info += f"{c['remarks']};"
        # if c['val'] != "":
        #     info += f"字段值样例是:{c['val']};"
        info += "</column>\n"
    info += "</table>"
    return info

def get_relevant_table_columns(questions: list[str]) -> list[dict]:
    """
    获取跟问题最相关的表的列
    """
    debug_mode = os.getenv("DEBUG", "0") == "1"
    logger = get_logger()
    table_top_cols = {}
    save_column_num = 15
    if debug_mode:
        print("\nget_relevant_table_columns --- 向量搜索 ---")
    logger.debug("\nget_relevant_table_columns --- 向量搜索 ---\n")
    for question in questions:
        # 向量搜索
        similarity_tables = calculate_table_similarity(question, 3)
        if debug_mode:
            print(f"question: {question}")
            for t in similarity_tables:
                print(f"{t[0]}: {t[1]:.3f} (最高分: {t[2]:.3f}, 总分: {t[3]:.3f}, 字段数: {t[4]}, \n最佳列: {t[5][:save_column_num]})")
        logger.debug("question: %s\n", question)
        for t in similarity_tables:
            logger.debug("%s: %.3f (最高分: %.3f, 总分: %.3f, 字段数: %d, \n最佳列: %s)\n", t[0], t[1], t[2], t[3], t[4], t[5][:save_column_num])
        for table_name, _, _, _, _, top_cols in similarity_tables:
            if table_name not in table_top_cols:
                table_top_cols[table_name] = set()
            table_top_cols[table_name].update([col for col, _ in top_cols[:save_column_num]])

    if debug_mode:
        print("\nget_relevant_table_columns --- 词频逆文档频率 ---")
    logger.debug("\nget_relevant_table_columns --- 词频逆文档频率 ---\n")
    for question in questions:
        # 词频逆文档频率
        tf_idf_tables = calculate_table_tf_idf_score(question, 1)
        if debug_mode:
            print(f"question: {question}")
            for t in tf_idf_tables:
                print(f"{t[0]}: {t[1]:.3f} (最高分: {t[2]:.3f}, 总分: {t[3]:.3f}, 字段数: {t[4]}, \n最佳列: {t[5][:save_column_num]})")
        logger.debug("question: %s\n", question)
        for t in tf_idf_tables:
            logger.debug("%s: %.3f (最高分: %.3f, 总分: %.3f, 字段数: %d, \n最佳列: %s)\n", t[0], t[1], t[2], t[3], t[4], t[5][:save_column_num])
        for table_name, _, _, _, _, top_cols in tf_idf_tables:
            if table_name not in table_top_cols:
                table_top_cols[table_name] = set()
            table_top_cols[table_name].update([col for col, _ in top_cols[:save_column_num]])

    for db_table_name in table_top_cols:
        table_top_cols[db_table_name] = list(table_top_cols[db_table_name])
    if debug_mode:
        print("\ntable_top_cols:\n", json.dumps(table_top_cols, ensure_ascii=False))
    logger.debug("\ntable_top_cols:\n%s", json.dumps(table_top_cols, ensure_ascii=False))

    db_table_names = list(table_top_cols.keys())
    # db_table_names = table_select_post_process(db_table_names) # fixme 是否还需要这个步骤？

    table_columns = []
    for db_table_name in db_table_names:
        table_column = {
            "表名": db_table_name,
            "表描述": config.table_index[db_table_name]['table_desc'],
            "表备注": config.table_index[db_table_name]['table_remarks'],
            "表字段": {},
        }
        for col_name in table_top_cols[db_table_name]:
            c = config.column_index[db_table_name][col_name]
            table_column["表字段"][col_name] = create_column_description(c)
        table_columns.append(table_column)
    table_columns = fill_sibling_columns(table_columns, all=True)
    return table_columns

def fill_sibling_tables(table_columns: list[dict]) -> list[dict]:
    """
    填充兄弟表字段 - 当发现某个表在预定义的表组中时，补充该组中其他相关表及其相应字段
    
    Args:
        table_columns (list[dict]): 原始表字段信息列表
        
    Returns:
        list[dict]: 补充了兄弟表和字段后的表字段信息列表
    """
    debug_mode = os.getenv("DEBUG", "0") == "1"
    logger = get_logger()

    sibling_table_groups = [
        {"constantdb.secumain", "constantdb.hk_secumain", "constantdb.us_secumain"},
        {"astockbasicinfodb.lc_stockarchives", "hkstockdb.hk_stockarchives", "usstockdb.us_companyinfo"},
        {"astockmarketquotesdb.qt_dailyquote", "hkstockdb.cs_hkstockperformance", "usstockdb.us_dailyquote"},
        {"astockmarketquotesdb.qt_stockperformance", "hkstockdb.cs_hkstockperformance"},
        # {"constantdb.us_secumain", "usstockdb.us_companyinfo"},
    ]

    # 将表列表转换为字典，方便快速查找
    hash_table_columns = {table["表名"]: table for table in table_columns}
    
    # 记录原始表列表，避免在遍历过程中修改列表导致问题
    org_table_names = list(hash_table_columns.keys())
    
    # 遍历每个原始表
    for db_table_name in org_table_names:
        # 查找该表所在的兄弟表组
        for sibling_table_group in sibling_table_groups:
            if db_table_name in sibling_table_group:
                # 获取当前表的字段列表
                current_table_fields = hash_table_columns[db_table_name]["表字段"]
                
                # 处理同组的其他表
                for sibling_table in sibling_table_group:
                    # 跳过当前表自身
                    if sibling_table == db_table_name:
                        continue
                    
                    # 检查config中是否有该表的索引
                    if sibling_table not in config.column_index:
                        continue
                        
                    # 如果兄弟表不在结果中，创建新表条目
                    if sibling_table not in hash_table_columns:
                        table_column = {
                            "表名": sibling_table,
                            "表描述": config.table_index[sibling_table]['table_desc'],
                            "表备注": config.table_index[sibling_table]['table_remarks'],
                            "表字段": {},
                        }
                        
                        # 添加与当前表相同的字段（如果兄弟表中存在）
                        for col_name in current_table_fields.keys():
                            if col_name in config.column_index[sibling_table]:
                                c = config.column_index[sibling_table][col_name]
                                table_column["表字段"][col_name] = create_column_description(c)
                        
                        # 只有当有字段被添加时才保存新表
                        if table_column["表字段"]:
                            hash_table_columns[sibling_table] = table_column
                            table_columns.append(table_column)
                            if debug_mode:
                                print(f"\n补充表字段: {sibling_table} [字段: {list(table_column['表字段'].keys())}]")
                            logger.debug("\n补充表字段: %s [字段: %s]", sibling_table, list(table_column['表字段'].keys()))
                    
                    # 如果兄弟表已存在，检查是否需要补充字段
                    else:
                        table_column = hash_table_columns[sibling_table]
                        # 检查当前表中有而兄弟表中没有的字段
                        for col_name in current_table_fields.keys():
                            if (col_name not in table_column["表字段"] and 
                                col_name in config.column_index.get(sibling_table, {})):
                                c = config.column_index[sibling_table][col_name]
                                table_column["表字段"][col_name] = create_column_description(c)
                                if debug_mode:
                                    print(f"\n补充表字段: {sibling_table}.{col_name}")
                                logger.debug("\n补充表字段: %s.%s", sibling_table, col_name)
    
    return table_columns

def fill_sibling_columns(table_columns: list[dict], all: bool = False) -> list[dict]:
    """
    填充相关联的字段（当出现某个字段时，补充相关的字段）
    
    Args:
        table_columns (list[dict]): 表字段信息列表
        
    Returns:
        list[dict]: 补充后的表字段信息列表
    """
    debug_mode = os.getenv("DEBUG", "0") == "1"
    logger = get_logger()
    
    # 定义常见的关联字段组合
    sibling_column_groups = [
        {"InnerCode", "CompanyCode", "SecuCode"},  # 证券基本信息关联字段
        {"IfAdjusted", "IfMerged"},  # 合并报表相关字段
        {"DividendRatioBeforeTax", "ActualRatioAfterTax"},  # 分红相关字段
        {"ChiName", "ChiNameAbbr"},
        {"EngName", "EngNameAbbr"},
        {"ChangePCT", "ChangePCTRMSix"},
        {"InfoPublDate", "InitialInfoPublDate"},
    ]
    if all:
        sibling_column_groups.extend([
            {"IfHighestHPriceRW", "IfHighestHPriceRM", "IfHighestHPriceRMThree", "IfHighestHPriceRMSix", "IfHighestHPriceRY", "IfHighestHPriceSL"},
            {"IfHighestCPriceRW", "IfHighestCPriceRM", "IfHighestCPriceRMThree", "IfHighestCPriceRMSix", "IfHighestCPriceRY", "IfHighestCPriceSL"},
            {"IfHighestTVolumeRW", "IfHighestTVolumeRM", "IfHighestTVRMThree", "IfHighestTVolumeRMSix", "IfHighestTVolumeRY", "IfHighestTVolumeSL"},
            {"IfHighestTValueRW", "IfHighestTValueRM", "IfHighestTValueRMThree", "IfHighestTValueRMSix", "IfHighestTValueRY", "IfHighestTValueSL"},
            {"HighestHPTimesSL", "HighestHPTimesRW", "HighestHPTimesRM", "HighestHPTimesRMThree", "HighestHPTimesRMSix", "HighestHPTimesRY"},
            {"IfLowestLPriceRW", "IfLowestLPriceRM", "IfLowestLPRMThree", "IfLowestLPriceRMSix", "IfLowestLPriceRY", "IfLowestLPriceSL"},
            {"IfLowestClosePriceRW", "IfLowestClosePriceRM", "IfLowestCPriceRMThree", "IfLowestCPriceRMSix", "IfLowestClosePriceRY", "IfLowestClosePriceSL"},
            {"IfLowestTVolumeRW", "IfLowestTVolumeRM", "IfLowestTVolumeRMThree", "IfLowestVolumeRMSix", "IfLowestTVolumeRY", "IfLowestTVolumeSL"},
            {"IfLowestTValueRW", "IfLowestTValueRM", "IfLowestTValueRMThree", "IfLowestTValueRMSix", "IfLowestTValueRY", "IfLowestTValueSL"},
            {"LowestLowPriceTimesSL", "LowestLowPriceTimesRW", "LowestLowPriceTimesRM", "LowestLPTimesRMThree", "LowestLPTimesRMSix", "LowestLPTimesRY"},
            {"RisingUpDays", "FallingDownDays", "VolumeRisingUpDays", "VolumeFallingDownDays"},
            {"BreakingMAverageFive", "BreakingMAverageTen", "BreakingMAverageTwenty", "BreakingMAverageSixty"},
        ])
    
    # 遍历每个表
    for table_column in table_columns:
        db_table_name = table_column["表名"]
        table_fields = table_column["表字段"]
        
        # 检查每个字段组合
        for column_group in sibling_column_groups:
            # 查找当前表中是否有这个组合中的字段
            intersection = set(table_fields.keys()) & column_group
            
            # 如果有交集但不完全包含，则添加缺失的字段
            if intersection and intersection != column_group:
                for col_name in column_group - set(table_fields.keys()):
                    if col_name in config.column_index.get(db_table_name, {}):
                        c = config.column_index[db_table_name][col_name]
                        table_fields[col_name] = create_column_description(c)
                        if debug_mode:
                            print(f"\n补充关联字段: {db_table_name}.{col_name}")
                        logger.debug("\n补充关联字段: %s.%s", db_table_name, col_name)
    return table_columns

def fill_import_columns(table_columns: list[dict]) -> list[dict]:
    """
    补充重要字段 - 当特定表出现在结果中时，确保这些表包含预定义的重要字段
    
    Args:
        table_columns (list[dict]): 表字段信息列表
        
    Returns:
        list[dict]: 补充了重要字段后的表字段信息列表
    """
    debug_mode = os.getenv("DEBUG", "0") == "1"
    logger = get_logger()
    
    # 定义特定表及其重要字段的映射
    # important_columns = {
    #     "astockindustrydb.lc_industryvaluation": ["TradingDay", "IndustryCode"],
    #     "astockmarketquotesdb.cs_stockcapflowindex": ["TradingDay", "InnerCode"],
    #     "astockmarketquotesdb.cs_stockpatterns": ["TradingDay", "InnerCode"],
    #     "astockmarketquotesdb.cs_turnovervoltecindex": ["TradingDay", "InnerCode"],
    #     "astockmarketquotesdb.qt_dailyquote": ["TradingDay", "InnerCode"],
    #     "astockmarketquotesdb.qt_stockperformance": ["TradingDay", "InnerCode"],
    #     "astockshareholderdb.cs_foreignholdingst": ["TradingDay", "InnerCode"],
    #     "hkstockdb.cs_hkstockperformance": ["TradingDay", "InnerCode"],
    #     "usstockdb.us_dailyquote": ["TradingDay", "InnerCode"],
    # }
    important_columns = {}
    import_column_names = {
        "TradingDay", "InnerCode", "CompanyCode", "SecuCode", "InitialInfoPublDate", "EndDate", "BeginDate",
        "IndustryName", "FirstPublDate", "IniInfoPublDate", "InitialImpleDay", "State", "RegAbbr",
        "ChiName", "ChiNameAbbr", "SecuAbbr", "EngName", "EngNameAbbr", "PEOStatus", "InfoPublDate",
        "ConceptName", "SubclassName", "ClassName", "ConceptCode", "SubclassCode", "ClassCode",
        "RelationType", "InfoTypeCode", "IfEffected", "Level", "AreaChiName", "EffectiveDate",
    }
    for db_table_name, cols in config.column_index.items():
        for col_name in cols:
            if col_name in import_column_names:
                if db_table_name not in important_columns:
                    important_columns[db_table_name] = []
                important_columns[db_table_name].append(col_name)
    
    # 遍历每个表
    for table_column in table_columns:
        db_table_name = table_column["表名"]
        table_fields = table_column["表字段"]
        
        # 检查是否是预定义表
        if db_table_name in important_columns:
            # 添加预定义的重要字段（如果不存在）
            for col_name in important_columns[db_table_name]:
                if col_name not in table_fields and col_name in config.column_index.get(db_table_name, {}):
                    c = config.column_index[db_table_name][col_name]
                    table_fields[col_name] = create_column_description(c)
                    if debug_mode:
                        print(f"\n补充重要字段: {db_table_name}.{col_name}")
                    logger.debug("\n补充重要字段: %s.%s", db_table_name, col_name)
    
    return table_columns

def validate_column_filter(column_filter: dict) -> str:
    """
    验证column_filter里的表和字段是否存在
    """
    logger = get_logger()
    result = []
    for db_table_name, cols in column_filter.items():
        if db_table_name not in config.table_index:
            result.append(f"不存在表[{db_table_name}];")
            continue
        for col in cols:
            if col not in config.column_index[db_table_name]:
                result.append(f"表[{db_table_name}]中没有字段[{col}];")
    return "\n".join(result)

def filter_table_columns(column_filter: dict) -> tuple[list[dict], list[str]]:
    """
    筛选表字段
    """
    logger = get_logger()
    filtered_table_columns = []
    stored_column_names = set()
    stored_table_names = set()
    # for table in table_columns:
    #     db_table_name = table["表名"]
    #     if db_table_name not in column_filter:
    #         continue
    #     table_column ={
    #         "表名": db_table_name,
    #         "表描述": config.table_index[db_table_name]['table_desc'],
    #         "表备注": config.table_index[db_table_name]['table_remarks'],
    #         "表字段": {},
    #     }
    #     stored_table_names.add(db_table_name)
    #     for c in config.table_index[db_table_name]['columns']:
    #         if c["name"] in column_filter[db_table_name] or c["name"] in config.import_column_names:
    #             table_column["表字段"][c["name"]] = create_column_description(c)
    #             stored_column_names.add(f"{db_table_name}.{c['name']}")
    #     filtered_table_columns.append(table_column)

    # column_filter里可能包含table_columns里没有的表或字段，所以需要补充
    for db_table_name, cols in column_filter.items():
        if db_table_name not in config.table_index:
            logger.debug("表%s不存在，跳过", db_table_name)
            continue
            # raise ValueError(f"表{db_table_name}不存在，请仔细检查")
        for col in cols:
            if col not in config.column_index[db_table_name]:
                logger.debug("表%s不存在字段%s，跳过", db_table_name, col)
                continue
                # raise ValueError(f"表{db_table_name}不存在字段{col}，请仔细检查")
            col_full_name = f"{db_table_name}.{col}"
            if col_full_name not in stored_column_names:
                if db_table_name not in stored_table_names:
                    table_column = {
                        "表名": db_table_name,
                        "表描述": config.table_index[db_table_name]['table_desc'],
                        "表备注": config.table_index[db_table_name]['table_remarks'],
                        "表字段": {},
                    }
                    filtered_table_columns.append(table_column)
                    stored_table_names.add(db_table_name)
                    table_column["表字段"][col] = create_column_description(config.column_index[db_table_name][col])
                    stored_column_names.add(col_full_name)
                    for c in config.table_index[db_table_name]['columns']:
                        if c["name"] in config.import_column_names:
                            table_column["表字段"][c["name"]] = create_column_description(c)
                            stored_column_names.add(f"{db_table_name}.{c['name']}")
                else:
                    for tc in filtered_table_columns:
                        if tc["表名"] == db_table_name:
                            table_column = tc
                            break
                    table_column["表字段"][col] = create_column_description(config.column_index[db_table_name][col])
                    stored_column_names.add(col_full_name)

    for db_table_name, cols in foreign_key_hub().items():
        for col in cols:
            if f"{db_table_name}.{col}" not in stored_column_names:
                if db_table_name not in stored_table_names:
                    table_column = {
                        "表名": db_table_name,
                        "表描述": config.table_index[db_table_name]['table_desc'],
                        "表备注": config.table_index[db_table_name]['table_remarks'],
                        "表字段": {},
                    }
                    filtered_table_columns.append(table_column)
                else:
                    for tc in filtered_table_columns:
                        if tc["表名"] == db_table_name:
                            table_column = tc
                for c in config.table_index[db_table_name]['columns']:
                    if c["name"] in cols or c["name"] in config.import_column_names:
                        table_column["表字段"][c["name"]] = create_column_description(c)
                        stored_column_names.add(f"{db_table_name}.{c['name']}")

    filtered_table_columns = fill_sibling_tables(filtered_table_columns)
    filtered_table_columns = fill_sibling_columns(filtered_table_columns, all=False)
    filtered_table_columns = fill_import_columns(filtered_table_columns)

    # 找出表之间的最短关联
    db_table_names = [t["表名"] for t in filtered_table_columns]
    table_relations = []
    a_table_set = {"constantdb.secumain", "astockbasicinfodb.lc_stockarchives", "astockmarketquotesdb.qt_dailyquote", "astockmarketquotesdb.qt_stockperformance"}
    hk_table_set = {"constantdb.hk_secumain", "hkstockdb.hk_stockarchives", "hkstockdb.cs_hkstockperformance"}
    us_table_set = {"constantdb.us_secumain", "usstockdb.us_companyinfo", "usstockdb.us_dailyquote"}
    
    # 定义不同市场组之间不应建立关联的规则
    no_relation_tables = [
        (a_table_set, hk_table_set),  # A股表与港股表不关联
        (a_table_set, us_table_set),  # A股表与美股表不关联
        (hk_table_set, us_table_set)  # 港股表与美股表不关联
    ]
    
    # pylint: disable=C0200
    for i in range(len(db_table_names)):
        from_table = db_table_names[i]
        for j in range(i+1, len(db_table_names)):
            to_table = db_table_names[j]
            
            # 检查这两个表是否属于不同市场组且不应关联
            skip_relation = False
            for set1, set2 in no_relation_tables:
                if (from_table in set1 and to_table in set2) or (from_table in set2 and to_table in set1):
                    skip_relation = True
                    break
            
            # 如果它们属于不应关联的不同市场组，跳过路径查找
            if skip_relation:
                continue
                
            rl_path = config.table_relations.find_shortest_path(from_table, to_table)
            if rl_path:
                path_tables = set()
                for rl in rl_path:
                    path_tables.add(rl[0])
                    path_tables.add(rl[1])
                # 这里再过一遍no_relation_tables
                for set1, set2 in no_relation_tables:
                    # 检查路径中的表是否跨越了不同市场组
                    set1_tables = path_tables.intersection(set1)
                    set2_tables = path_tables.intersection(set2)
                    if set1_tables and set2_tables:
                        skip_relation = True
                        break
                if skip_relation:
                    continue
                table_relations.append(config.table_relations.print_path(rl_path))

    return filtered_table_columns, table_relations

def get_db_info() -> str:
    """
    获取数据库信息
    """
    db_info = {}
    for db_name, db_data in config.db_table.items():
        db_info[db_name] = db_data.get('desc', '')
    
    return (
        json.dumps(db_info, ensure_ascii=False)+
        "\n"
    )

def get_table_list(dbs: list[str]) -> str:
    """
    Retrieves a list of tables for each specified database.

    Parameters:
    dbs (list[str]): A list of database names.

    Returns:
    str: A formatted string containing the table information for each database.
    """
    table_list = []
    for db_name in dbs:
        if db_name not in config.db_table:
            raise KeyError(f"发生异常: 数据库名`{db_name}`不存在")
        for table_name, table_info in config.db_table[db_name]['tables'].items():
            table_list.append({
                "表名": f"{db_name}."+table_name,
                "说明": table_info["cols_summary"]
            })
    result = (
        "数据库表信息如下:\n" +
        json.dumps(table_list, ensure_ascii=False)+
        "\n"
    )
    return result

def get_column_list(tables: list[str]) -> str:
    """
    tables: list of table names, format is database_name.table_name
    """
    column_lists = []
    for table in tables:
        if '.' not in table or table.count('.') != 1:
            raise ValueError(f"发生异常: 表名`{table}`格式不正确，应该为database_name.table_name")
        if table not in config.table_index:
            raise ValueError(f"发生异常: 表名`{table}`不存在")
        t = config.table_index[table]
        column_lists.append({
            "表名": table,
            "表描述": t['table_desc'],
            "表备注": t['table_remarks'],
            "表字段": {c['name']: create_column_description(c) for c in t['columns']},
        })
    result = (
        f"已取得可用的{COLUMN_LIST_MARK}:\n" +
        # json.dumps(column_lists, ensure_ascii=False)+
        "\n---\n".join([print_table_column(c) for c in column_lists])+
        "\n"
    )
    return result
