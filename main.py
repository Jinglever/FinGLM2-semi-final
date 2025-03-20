"""主程序入口"""

import os
import sys
import threading
import json
import copy
import logging
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.log import setup_logger, get_logger
import config
from agents import agent_summary_answer, agent_extract_company, agent_rewrite_question
from workflows import sql_query, check_db_structure
from utils import ajust_org_question, find_similar_texts

# 创建一个全局锁
file_write_lock = threading.Lock()

def process_single_question(question_team: dict, q_idx: int) -> None:
    """
    处理单个问题，提取实体、重写问题并生成答案
    """
    debug_mode = os.getenv("DEBUG", "0") == "1"
    ag_rewrite_question = agent_rewrite_question.clone()
    ag_extract_company = agent_extract_company.clone()
    wf_sql_query = sql_query.clone()
    wf_check_db_structure = check_db_structure.clone()
    ag_summary_answer = agent_summary_answer.clone()
    facts = []
    qas = []
    sql_results = []
    question_item = question_team["team"][q_idx]
    qid = question_item["id"].strip()
    if not config.FLAG_IGNORE_CACHE and "answer" in question_item and question_item["answer"] != "":
        print(f"\n>>>>> {qid} 已存在答案，跳过...\n")
        return

    question = ajust_org_question(question_item["question"])
    for idx in range(q_idx):
        qas.append([
            {"role": "user", "content": ajust_org_question(question_team["team"][idx]["question"])},
            {"role": "assistant", "content": question_team["team"][idx]["answer"] if "answer" in question_team["team"][idx] else ""},
        ])
        if "facts" in question_team["team"][idx]:
            facts = copy.deepcopy(question_team["team"][idx]["facts"])
        if "sql_results" in question_team["team"][idx]:
            sql_results.append(copy.deepcopy(question_team["team"][idx]["sql_results"]))
        else:
            sql_results.append([])

    start_time = time.time()
    log_file_path = config.OUTPUT_DIR + f"/{qid}.log"
    open(log_file_path, 'w', encoding='utf-8').close()
    setup_logger(
        log_file=log_file_path,
        log_level=logging.DEBUG,
    )
    logger = get_logger()

    logger.debug("\n>>>>> Original Question: %s\n", question_item["question"])

    # 获取实体内部代码
    # ag_extract_company.clear_history()
    answer, _ = ag_extract_company.answer((
        '''提取下面这段文字中的实体（如公司名、股票代码、拼音缩写等），如果识别结果是空，那么就回复No Entities.'''
        f'''"{question}"'''
    ))
    answer = answer.strip()
    if answer != "" and answer not in facts:
        facts.append(answer)

    # rewrite question
    # ag_rewrite_question.clear_history()
    qas_content = [
        f"{qa[0]['content']} (无需查询，已有答案: {qa[1]['content']})"
        for qa in qas
    ]
    new_question = (
        ("\n".join(qas_content) + "\n" if len(qas_content) > 0 else "") +
        question
    )
    # new_question, _ = ag_rewrite_question.answer(
    #     ("历史问答:无。\n" if len(qas_content) == 0 else "下面是顺序的历史问答:\n'''\n" + "\n".join(qas_content) + "\n'''\n") +
    #     "现在用户继续提问，请根据已知信息，理解当前这个问题的完整含义，并重写这个问题使得单独拿出来看仍然能够正确理解。用户的问题是：\n" +
    #     question
    # )

    # 注入已知事实
    key_facts = "已知信息"
    if len(facts) > 0:
        kv = {key_facts: "\n".join(facts)}
        wf_sql_query.agent_master.add_system_prompt_kv(kv)
        # wf_check_db_structure.agent_decode_question.add_system_prompt_kv(kv)
        wf_check_db_structure.agent_column_selector.add_system_prompt_kv(kv)
    else:
        wf_sql_query.agent_master.del_system_prompt_kv(key_facts)
        # wf_check_db_structure.agent_decode_question.del_system_prompt_kv(key_facts)
        wf_check_db_structure.agent_column_selector.del_system_prompt_kv(key_facts)
    logger.debug("\n>>>>> %s:\n%s", key_facts, "\n---\n".join(facts))

    # 注入历史对话以及支撑它的SQL查询
    key_qas = "历史对话"
    if len(qas_content) > 0:
        val_qas = ""
        for qa_idx, qa_content in enumerate(qas_content):
            if qa_idx > 0:
                val_qas += "---\n"
            val_qas += f"{qa_content}\n"
            if sql_results[qa_idx] != []:
                val_qas += (
                    "用到以下SQL查询（供后续问答理解本问题的答案如何得来，后续问答可参考）：\n" +
                    "\n".join(sql_results[qa_idx])
                )
        kv = {key_qas: val_qas}
        wf_sql_query.agent_master.add_system_prompt_kv(kv)
        wf_check_db_structure.agent_column_selector.add_system_prompt_kv(kv)
    else:
        wf_sql_query.agent_master.del_system_prompt_kv(key_qas)
        wf_check_db_structure.agent_column_selector.del_system_prompt_kv(key_qas)

    # 注入sql模板
    key_sql_template = "SQL参考样例"
    sql_template_sim = find_similar_texts(
        search_query=question,
        vectors=config.sql_template_vectors,
        texts=config.sql_template,
        top_p=3,
        threshold=0.65
    )
    if len(sql_template_sim[0]) > 0:
        logger.debug("\n>>>>> %s:\n%s", key_sql_template, "\n".join(sql_template_sim[1]))
        kv = {key_sql_template: "\n".join(sql_template_sim[1])}
        wf_sql_query.agent_master.add_system_prompt_kv(kv)
        wf_check_db_structure.agent_column_selector.add_system_prompt_kv(kv)
    else:
        wf_sql_query.agent_master.del_system_prompt_kv(key_sql_template)
        wf_check_db_structure.agent_column_selector.del_system_prompt_kv(key_sql_template)
        
    # 搜索相关数据库结构
    # wf_check_db_structure.clear_history()
    res = wf_check_db_structure.run(inputs={"messages":[
        {"role": "user", "content": new_question}
    ]})
    db_info = res["content"]

    # wf_sql_query.clear_history()
    if db_info != "":

        if debug_mode:
            print(f"\n>>>>> db_info:\n{db_info}")
        logger.debug("\n>>>>> db_info:\n%s", db_info)

        # 查询数据库回答用户问题
        res = wf_sql_query.run(inputs={"messages":[
            {"role": "assistant", "content": db_info},
            {"role": "user", "content": new_question},
        ]})
        answer, _ = ag_summary_answer.answer(
            f'''{res["content"]}\n充分尊重前面给出的结论，回答问题:\n<question>{question_item["question"]}</question>'''
        )
        question_item["answer"] = answer

        # Caching
        qas.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": question_item["answer"]},
        ])
    else:
        print(f"\n>>>>> {qid} db_info is empty, skip this question\n")
        logger.debug("\n>>>>> db_info is empty, skip this question: %s", qid)

    elapsed_time = time.time() - start_time
    question_item["usage_tokens"] = {
        agent_extract_company.cfg.name: ag_extract_company.usage_tokens,
        agent_rewrite_question.cfg.name: ag_rewrite_question.usage_tokens,
        check_db_structure.name: wf_check_db_structure.usage_tokens,
        sql_query.name: wf_sql_query.usage_tokens,
    }
    minutes, seconds = divmod(elapsed_time, 60)
    question_item["use_time"] = f"{int(minutes)}m {int(seconds)}s"
    question_item["facts"] = copy.deepcopy(facts)
    question_item["rewrited_question"] = new_question
    question_item["sql_results"] = copy.deepcopy(wf_sql_query.history_facts)

    print((
        f">>>>> id: {qid}\n" +
        f">>>>> Original Question: {question_item["question"]}\n" +
        f">>>>> Rewrited Question: {new_question}\n" +
        f">>>>> Answer: {question_item["answer"]}\n" +
        f">>>>> Used Time: {int(minutes)}m {int(seconds)}s\n"
    ))


def process_team_question(question_team: dict, team_idx: int) -> None:
    """
    处理一组问题
    """
    debug_mode = os.getenv("DEBUG", "0") == "1"
    tid = question_team["tid"].strip()
    questions = []
    qids = []
    for question_item in question_team["team"]:
        questions.append(question_item["question"])
        qids.append(question_item["id"].strip())
    ag_summary_answer = agent_summary_answer.clone()
    ag_extract_company = agent_extract_company.clone()
    wf_sql_query = sql_query.clone()
    wf_check_db_structure = check_db_structure.clone()
    facts = []
    if not config.FLAG_IGNORE_CACHE and "done" in question_team and question_team["done"]:
        print(f"\n>>>>> {team_idx} 已存在答案，跳过...\n")
        return

    for i in range(len(questions)):
        questions[i] = ajust_org_question(questions[i])

    start_time = time.time()
    log_file_path = config.OUTPUT_DIR + f"/{tid}.log"
    open(log_file_path, 'w', encoding='utf-8').close()
    setup_logger(
        log_file=log_file_path,
        log_level=logging.DEBUG,
    )
    logger = get_logger()

    # 获取实体内部代码
    # ag_extract_company.clear_history()
    answer, _ = ag_extract_company.answer((
        '''提取下面这段文字中的实体（如公司名、股票代码、拼音缩写等），如果识别结果是空，那么就回复No Entities.'''
        f'''<questions>{"\n".join(questions)}</questions>'''
    ))
    answer = answer.strip()
    if answer != "" and answer not in facts:
        facts.append(answer)

    # 注入已知事实
    key_facts = "已知信息"
    if len(facts) > 0:
        kv = {key_facts: "\n".join(facts)}
        wf_sql_query.agent_master.add_system_prompt_kv(kv)
        # wf_check_db_structure.agent_decode_question.add_system_prompt_kv(kv)
        wf_check_db_structure.agent_column_selector.add_system_prompt_kv(kv)
    else:
        wf_sql_query.agent_master.del_system_prompt_kv(key_facts)
        # wf_check_db_structure.agent_decode_question.del_system_prompt_kv(key_facts)
        wf_check_db_structure.agent_column_selector.del_system_prompt_kv(key_facts)
    # if debug_mode:
    #     print(f"\n>>>>> {key_facts}:\n" + "\n---\n".join(facts))
    logger.debug("\n>>>>> %s:\n%s", key_facts, "\n---\n".join(facts))

    # 注入sql模板
    key_sql_template = "SQL参考样例"
    sql_template_sim = set()
    for question in questions:
        sim = find_similar_texts(
            search_query=question,
            vectors=config.sql_template_vectors,
            texts=config.sql_template,
            top_p=3,
            threshold=0.65
        )
        sql_template_sim.update(sim[1])
    if len(sql_template_sim) > 0:
        logger.debug("\n>>>>> %s:\n%s", key_sql_template, "\n".join(sql_template_sim))
        kv = {key_sql_template: "\n".join(sql_template_sim)}
        wf_sql_query.agent_master.add_system_prompt_kv(kv)
        wf_check_db_structure.agent_column_selector.add_system_prompt_kv(kv)
    else:
        wf_sql_query.agent_master.del_system_prompt_kv(key_sql_template)
        wf_check_db_structure.agent_column_selector.del_system_prompt_kv(key_sql_template)

    # 搜索相关数据库结构
    res = wf_check_db_structure.run(inputs={"messages":[
        {"role": "user", "content": " ".join(questions)}
    ]})
    db_info = res["content"]

    if db_info != "":
        if debug_mode:
            print(f"\n>>>>> db_info:\n{db_info}")
        logger.debug("\n>>>>> db_info:\n%s", db_info)

        # 查询数据库回答用户问题
        res = wf_sql_query.run(inputs={"messages":[
            {"role": "assistant", "content": db_info},
            {"role": "user", "content": " ".join(questions)},
        ]})

        # 逐条回答原问题
        for question_item in question_team["team"]:
            answer, _ = ag_summary_answer.answer(
                f'''{res["content"]}\n充分尊重前面给出的结论，回答问题:\n<question>{question_item["question"]}</question>'''
            )
            question_item["answer"] = answer
    else:
        print(f"\n>>>>> {tid} db_info is empty, skip this question\n")
        logger.debug("\n>>>>> db_info is empty, skip this question: %s", tid)

    elapsed_time = time.time() - start_time
    question_team["usage_tokens"] = {
        agent_extract_company.cfg.name: ag_extract_company.usage_tokens,
        check_db_structure.name: wf_check_db_structure.usage_tokens,
        sql_query.name: wf_sql_query.usage_tokens,
    }
    minutes, seconds = divmod(elapsed_time, 60)
    question_team["use_time"] = f"{int(minutes)}m {int(seconds)}s"
    question_team["facts"] = copy.deepcopy(facts)
    question_team["sql_results"] = copy.deepcopy(wf_sql_query.history_facts)
    question_team["done"] = True

    msg = "\n"
    for question_item in question_team["team"]:
        msg += (
            f">>>>> id: {question_item["id"]}\n" +
            f">>>>> Question: {question_item["question"]}\n" +
            f">>>>> Answer: {question_item["answer"]}\n"
        )
    msg += f">>>>> Used Time: {int(minutes)}m {int(seconds)}s\n"
    print(msg)

# 执行任务
def process_question(question_team: dict, team_idx: int) -> None:
    """
    Processes a team of questions, extracting facts and generating answers.
    
    Args:
        question_team (dict): A dictionary containing a list of questions to process.
        team_idx(int): index of team
    """
    print(f"----- Processing Team Index {team_idx} ... -----\n")
    if config.ENABLE_BATCH_MODE:
        if team_idx >= config.START_INDEX[0] and team_idx <= config.END_INDEX[0]:
            process_team_question(question_team, team_idx)
            # 使用锁来保证多线程写入文件时不会出现问题
            with file_write_lock:
                with open(config.OUTPUT_DIR+"/all_question.json", 'w', encoding='utf-8') as f1:
                    json.dump(config.all_question, f1, ensure_ascii=False, indent=4)  # 添加缩进以便于阅读
                with open(config.SUBMIT_DIR+f"/all_question{config.SAVE_FILE_SUBFIX}.json", 'w', encoding='utf-8') as f2:
                    json.dump(config.all_question, f2, ensure_ascii=False, indent=4)  # 添加缩进以便于阅读
    else:
        for q_idx, _ in enumerate(question_team["team"]):
            if team_idx == config.START_INDEX[0] and q_idx < config.START_INDEX[1]:
                continue
            if team_idx == config.END_INDEX[0] and q_idx > config.END_INDEX[1]:
                return question_team
            process_single_question(question_team, q_idx)
            # 使用锁来保证多线程写入文件时不会出现问题
            with file_write_lock:
                with open(config.OUTPUT_DIR+"/all_question.json", 'w', encoding='utf-8') as f1:
                    json.dump(config.all_question, f1, ensure_ascii=False, indent=4)  # 添加缩进以便于阅读
                with open(config.SUBMIT_DIR+f"/all_question{config.SAVE_FILE_SUBFIX}.json", 'w', encoding='utf-8') as f2:
                    json.dump(config.all_question, f2, ensure_ascii=False, indent=4)  # 添加缩进以便于阅读

    print(f"----- Completed Team Index {team_idx} -----\n")
    return

if __name__ == "__main__":
    if len(sys.argv) > 2:
        question_file = sys.argv[1]
        submit_file = sys.argv[2]
        config.SUBMIT_DIR = os.path.dirname(submit_file)
        config.SUBMIT_FILE = submit_file
        config.QUESTION_FILE = question_file
    cache_file = config.SUBMIT_DIR + f"/all_question{config.SAVE_FILE_SUBFIX}.json"
    print("cache_file: ", cache_file) # fortest
    if os.path.exists(cache_file) and not config.FLAG_IGNORE_CACHE:
        print("cache_file exists") # fortest
        with open(cache_file, 'r', encoding='utf-8') as file:
            config.all_question = json.load(file)
    else:
        with open(config.QUESTION_FILE, 'r', encoding='utf-8') as file:
            config.all_question = json.load(file)
    config.START_INDEX = [0, 0]
    if len(config.all_question) > 0:
        config.END_INDEX = [len(config.all_question)-1, len(config.all_question[-1]["team"])-1]
    else:
        config.END_INDEX = [0, 0]

    # 使用切片来限制处理的范围
    selected_questions = config.all_question[config.START_INDEX[0]:config.END_INDEX[0]+1]
    # 使用线程池并发处理
    if config.MAX_CONCURRENT_THREADS == -1:
        config.MAX_CONCURRENT_THREADS = os.cpu_count()
    with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_THREADS) as executor:
        futures = {
            executor.submit(process_question, q_team, i + config.START_INDEX[0]): i + config.START_INDEX[0]
            for i, q_team in enumerate(selected_questions)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing JSON data in range"
        ):
            i = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"\n***** Team Index {i} generated an exception: {exc} *****\n")

    # 计算tokens数
    total_usage_tokens = {
        agent_extract_company.cfg.name: 0,
        agent_rewrite_question.cfg.name: 0,
        check_db_structure.name: 0,
        sql_query.name: 0,
    }

    for q_team in config.all_question:
        if "usage_tokens" in q_team:
            for key in q_team["usage_tokens"]:
                if key in total_usage_tokens:
                    total_usage_tokens[key] += q_team["usage_tokens"][key]
        for q_item in q_team["team"]:
            if "usage_tokens" in q_item:
                for key in q_item["usage_tokens"]:
                    if key in total_usage_tokens:
                        total_usage_tokens[key] += q_item["usage_tokens"][key]


    print(json.dumps(total_usage_tokens, ensure_ascii=False, indent=4))

    total_tokens = sum(total_usage_tokens.values())
    print(f"所有tokens数: {total_tokens}")

    for q_team in config.all_question:
        if "usage_tokens" in q_team:
            del q_team["usage_tokens"]
        if "use_time" in q_team:
            del q_team["use_time"]
        if "facts" in q_team:
            del q_team["facts"]
        if "sql_results" in q_team:
            del q_team["sql_results"]
        if "done" in q_team:
            del q_team["done"]
        for q_item in q_team["team"]:
            if "usage_tokens" in q_item:
                del q_item["usage_tokens"]
            if "use_time" in q_item:
                del q_item["use_time"]
            if "iterate_num" in q_item:
                del q_item["iterate_num"]
            if "facts" in q_item:
                del q_item["facts"]
            if "rewrited_question" in q_item:
                del q_item["rewrited_question"]
            if "sql_results" in q_item:
                del q_item["sql_results"]


    with open(config.OUTPUT_DIR+"/"+os.path.basename(config.SUBMIT_FILE), 'w', encoding='utf-8') as f1:
        json.dump(config.all_question, f1, ensure_ascii=False, indent=4)  # 添加缩进以便于阅读
    with open(config.SUBMIT_FILE, 'w', encoding='utf-8') as f2:
        json.dump(config.all_question, f2, ensure_ascii=False, indent=4)  # 添加缩进以便于阅读
