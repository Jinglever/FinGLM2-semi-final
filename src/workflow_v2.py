"""
This module defines the Workflow abstract base class and its implementation, RecallDbInfo,
which handles recalling database information using various agents.
"""

import json
import os
import datetime
import copy
from abc import ABC, abstractmethod
from typing import Callable, Optional

from src.log import get_logger
from src.llm import LLM
from src.agent import Agent, AgentConfig
from src.utils import extract_last_sql, extract_last_json, COLUMN_LIST_MARK, count_total_sql

class Workflow(ABC):
    """
    Abstract base class defining the basic interface for workflows.
    """

    @abstractmethod
    def clone(self) -> 'Workflow':
        """Creates a clone of the current Workflow instance."""

    @abstractmethod
    def run(self, inputs: dict) -> dict:
        """
        运行工作流，返回结果。

        return: dict
            - content: str, 结果内容
            - usage_tokens: int, 使用的token数量
        """
    @abstractmethod
    def clear_history(self):
        """
        清除工作流内部的agent的history
        """
    @abstractmethod
    def add_system_prompt_kv(self, kv: dict):
        """
        给agent的system prompt增加设定
        """

    @abstractmethod
    def del_system_prompt_kv(self, key: str):
        """Deletes the specified key from the system prompt key-value pairs for the agent."""

    @abstractmethod
    def clear_system_prompt_kv(self):
        """
        清除agent的system prompt额外设定
        """

class SqlQuery(Workflow):
    """
    Implements the functionality to write and execute sql to fetch data, inheriting from Workflow.
    """
    def __init__(self, execute_sql_query: Callable[[str],str],
                 llm: LLM, max_iterate_num: int = 5,
                 name: Optional[str] = None,
                 specific_column_desc: Optional[dict] = None,
                 cache_history_facts: Optional[bool] = False,
                 default_sql_limit: Optional[int] = None):
        self.name = "Sql_query" if name is None else name
        self.execute_sql_query = execute_sql_query
        self.max_iterate_num = max_iterate_num
        self.usage_tokens = 0
        self.is_cache_history_facts = cache_history_facts
        self.history_facts = []
        self.max_db_struct_num = 1
        self.specific_column_desc = specific_column_desc if specific_column_desc is not None else {}
        self.default_sql_limit = default_sql_limit
        self.agent_master = Agent(AgentConfig(
            name = self.name+".master",
            role = (
                f'''(当前是{datetime.datetime.now().year}年)\n'''
                '''你是一个严谨的MySQL专家，擅长通过分步拆解的方式获取数据。你遵循以下原则：\n'''
                '''**Core Principles**\n'''
                '''1. 采用分步执行策略：先执行基础查询 → 分析结果 → 执行后续查询\n'''
                '''2. 每个交互周期仅执行单条SQL语句，确保可维护性和性能\n'''
                '''3. 已经尝试过的方案不要重复尝试，如果没有更多可以尝试的方案，就说明情况并停止尝试。\n'''
                '''**!!绝对执行规则!!**\n'''
                '''- 每次响应有且仅有一个 ```exec_sql 代码块\n'''
                '''- 即使需要多步操作，也必须分次请求执行\n'''
                '''- 出现多个SQL语句将触发系统级阻断\n'''
                '''- 不使用未知的表名和字段名\n'''
                '''- 获取任何实体或概念，如果它在同一张表里存在唯一编码，要顺便把它查询出来备用\n'''
                '''- 不准写插入语句\n'''
            ),
            constraint = (
                '''- 时间日期过滤必须对字段名进行格式化：`DATE(column_name) (op) 'YYYY-MM-DD'` 或 `YEAR(column_name) (op) 'YYYY'`\n'''
                '''- 表名必须完整格式：database_name.table_name（即使存在默认数据库）\n'''
                '''- 字符串搜索总是采取模糊搜索，总是优先用更短的关键词去搜索，增加搜到结果的概率\n'''
                '''- 若所需表/字段未明确存在，必须要求用户确认表结构\n'''
                '''- 当遇到空结果时，请检查是否存在下述问题：\n'''
                '''    1. 时间日期字段是否使用DATE()或YEAR()进行了格式化\n'''
                '''    2. 字段跟值并不匹配，比如把股票代码误以为公司代码\n'''
                '''    3. 字段语言版本错配，比如那中文的字串去跟英文的字段匹配\n'''
                '''    4. 可以通过SELECT * FROM database_name.table_name LIMIT 1;了解所有字段的值是什么形式\n'''
                '''    5. 是否可以把时间范围放宽了解一下具体情况\n'''
                '''    6. 关键词模糊匹配是否可以把关键词改短后再事实？\n'''
                '''    7. 枚举值是否正确\n'''
                '''- 如果确认查找的方式是正确的，那么可以接受空结果!!!\n'''
                '''- 每次交互只处理一个原子查询操作\n'''
                '''- 连续步骤必须显式依赖前序查询结果\n'''
                '''- 如果总是执行失败，尝试更换思路，拆解成简单SQL，逐步执行确认\n'''
                '''- 擅于使用DISTINC，尤其当发现获取的结果存在重复，去重后不满足期望的数量的时候，比如要查询前10个结果，但是发现结果里存在重复，那么就要考虑使用DISTINC重新查询\n'''
                '''- 在MySQL查询中，使用 WHERE ... IN (...) 不能保持传入列表的顺序，可通过 ORDER BY FIELD(列名, 值1, 值2, 值3, ...) 强制按指定顺序排序。\n'''
                '''- 对于求中位数的查询，通常会使用 ROW_NUMBER() 或类似的方法来代替 LIMIT 这种复杂的动态限制，如果确实需要获取中位数，你需要确保能动态计算并获取中位位置的记录\n'''
                '''- 如果你需要在 MySQL 中按特定顺序返回 IN 查询结果，可以使用 FIELD() 函数对结果进行排序。\n'''
                '''- 绝对不允许编造不存在的database_name.table_name.column_name\n'''
                '''- not support 'LIMIT & IN/ALL/ANY/SOME subquery\n'''
            ),
            output_format = (
                '''分阶段输出模板：\n'''

                '''【已知信息】\n'''
                '''（这里写当前已知的所有事实信息，尤其要注重历史对话中的信息）\n'''

                '''【用户的问题】\n'''
                '''(这里复述用户的问题，防止遗忘)\n'''

                '''【思维链】\n'''
                '''(think step by step, 分析用户的问题，结合用户提供的可用数据字段，思考用哪些字段获得什么数据，逐步推理直至可以回答用户的问题)\n'''
                '''(例如: \n'''
                '''用户问: 2021年末，交通运输一级行业中有几个股票？\n'''
                '''思维链：\n'''
                '''我们下一个要获取的信息是交通运输一级行业的代码是什么，'''
                '''可以用lc_exgindchange表的FirstIndustryName字段可以找到交通运输一级行业的行业代码FirstIndustryCode;\n'''
                '''我们下一个要获取的信息是交通运输一级行业的股票有多少个，'''
                '''在lc_indfinindicators表通过IndustryCode字段和Standard字段(41-申万行业分类2021版)搜索到交通运输一级行业的信息,'''
                '''其中lc_indfinindicators表的ListedSecuNum字段就是上市证券数量，\n'''
                '''由于用户问的是2021年末，所以我需要用lc_indfinindicators表的InfoPublDate字段排序获得2021年末最后一组数据)\n'''

                # '''【当前阶段要获取的信息】\n'''
                # '''(如果无需继续执行SQL，那么这里写"无")\n'''

                # '''【信息所在字段】\n'''
                # '''(如果无需继续执行SQL，那么这里写"无")\n'''
                # '''(如果已知字段里缺少需要的字段，那么用`SELECT * FROM database_name.table_name LIMIT 1;`来了解这个表的字段值的形式)\n'''
                # '''- database_name.table_name.column_name: 这个字段可能包含xx信息，对应用户提问中的xxx\n'''

                # '''【筛选条件所在字段】\n'''
                # '''(如果无需继续执行SQL，那么这里写"无")\n'''
                # '''(认真检查，时间日期过滤必须对字段名进行格式化：`DATE(column_name) (op) 'YYYY-MM-DD'` 或 `YEAR(colum_name) (op) 'YYYY'`)\n'''
                # '''(如果是枚举类型的字段，那么要同时写明用哪些枚举值做条件)\n'''
                # '''- database_name.table_name.column_name: 这个字段可能包含xx信息，对应用户提问中的xxx\n'''

                # '''【排序字段】\n'''
                # '''(如果无需继续执行SQL，那么这里写"无")\n'''
                # '''(如果用户提到排序如最大最小最新最旧最高最低之类，那么这里写排序字段)\n'''
                # '''(注意选择正确的表里的排序字段)\n'''
                # '''- database_name.table_name.column_name: 这个字段可能包含xx信息，对应用户提问中的xxx\n'''

                # '''【枚举类型字段】\n'''
                # '''(如果无需继续执行SQL，那么这里写"无")\n'''
                # '''(如果涉及枚举类型字段，这里说明要用到哪些枚举值)\n'''

                # '''【涉及的表间外链关系】\n'''
                # '''(如果无需继续执行SQL，那么这里写"无")\n'''
                # '''(根据已知表间外链关系，考虑是否需要联表查询)\n'''

                # '''【SQL语句的思路】\n'''
                # '''(上一步的SQL是否存在问题，比如枚举值有误？)\n'''
                # '''(接着要执行的SQL的思路)\n'''
                # '''(如果无需继续执行SQL，那么这里写"无")\n'''
                # '''(这里必须使用已知的数据库表和字段，不能假设任何数据表或字典)\n'''
                # '''(如果要分子步骤执行，可在这里写明)\n'''

                '''【执行SQL语句】\n'''
                '''（唯一允许的SQL代码块，如果当前阶段无需继续执行SQL，那么这里写"无"）\n'''
                '''（如果涉及到数学运算即便是用已知的纯数字做计算，也可以通过SQL语句来进行，保证计算结果的正确性，如`SELECT 1+1 AS a`）\n'''
                '''(这里必须使用已知的数据库表和字段，不能假设任何数据表或字典)\n'''
                '''(筛选条件参见前面的【筛选条件所在字段】)\n'''
                '''(排序字段参见前面的【排序字段】)\n'''
                # '''(只允许写当前要执行一条SQL语句，不要写多条SQL语句)\n'''
                '''(仅允许给出一组当前待执行的SQL写到代码块```exec_sql ```中，后续要执行的sql请用```sql```代码块来写)\n'''
                '''```exec_sql\n'''
                '''SELECT [精准字段] \n'''
                '''FROM [完整表名] \n'''
                '''WHERE [条件原子化] \n'''
                '''LIMIT [强制行数]\n'''

                # '''【上述SQL语句的含义】\n'''
                # '''（如果当前阶段无执行SQL，那么这里写"无"）\n'''
            ),
            llm = llm,
            enable_history=True,
            temperature = 0.6,
            # top_p = 0.7,
            # stream = False,
        ))
        self.agent_understand_query_result = Agent(AgentConfig(
            name=self.name+".understand_query_result",
            role="你是优秀的数据库专家和数据分析师，负责根据已知的数据库结构说明，以及用户提供的SQL语句，理解这个SQL的查询结果。",
            output_format=(
                "输出模板:\n"
                "查询结果表明:\n"
                "(一段话描述查询结果，不遗漏重要信息，不捏造事实，没有任何markdown格式，务必带上英文字段名)\n"
            ),
            llm=llm,
            enable_history=False,
            # stream=False,
        ))
        self.agent_summary = Agent(AgentConfig(
            name=self.name+".summary",
            role="你负责根据当前已知的事实信息，回答用户的提问。",
            constraint=(
                '''- 根据上下文已知的事实信息回答，不捏造事实\n'''
            ),
            output_format=(
                '''- 用一段文字来回答，不要有任何markdown格式，不要有换行\n'''
            ),
            llm=llm,
            enable_history=False,
            # stream=False,
        ))
        self.update_agent_lists()

    def update_agent_lists(self):
        """Updates the list of agents used in the workflow."""
        self.agent_lists = [
            self.agent_master,
            self.agent_summary,
            self.agent_understand_query_result,
        ]

    def clone(self) -> 'SqlQuery':
        clone =  SqlQuery(
            execute_sql_query=self.execute_sql_query,
            llm=self.agent_master.cfg.llm,  # Assuming all agents use the same LLM
            max_iterate_num=self.max_iterate_num,
            name=self.name,
            specific_column_desc=copy.deepcopy(self.specific_column_desc),
            cache_history_facts=self.is_cache_history_facts,
            default_sql_limit=self.default_sql_limit
        )
        clone.agent_master = self.agent_master.clone()
        clone.agent_understand_query_result = self.agent_understand_query_result.clone()
        clone.agent_summary = self.agent_summary.clone()
        clone.update_agent_lists()
        return clone

    def clear_history(self):
        self.usage_tokens = 0
        for agent in self.agent_lists:
            agent.clear_history()

    def clear_history_facts(self):
        """Clears the stored history facts."""
        self.history_facts = []

    def add_system_prompt_kv(self, kv: dict):
        for agent in self.agent_lists:
            agent.add_system_prompt_kv(kv=kv)

    def del_system_prompt_kv(self, key: str):
        """Deletes the specified key from the system prompt key-value pairs for the agent."""
        for agent in self.agent_lists:
            agent.del_system_prompt_kv(key=key)

    def clear_system_prompt_kv(self):
        for agent in self.agent_lists:
            agent.clear_system_prompt_kv()

    def run(self, inputs: dict) -> dict:
        """
        inputs:
            - messages: list[dict] # 消息列表，每个元素是一个dict，包含role和content
        """
        debug_mode = os.getenv("DEBUG", "0") == "1"
        logger = get_logger()
        usage_tokens = 0
        same_sqls = {}
        told_specific_columns = set()

        if 'messages' not in inputs:
            raise KeyError("发生异常: inputs缺少'messages'字段")

        db_structs = []
        messages = []
        for msg in inputs["messages"]:
            if COLUMN_LIST_MARK in msg["content"]:
                db_structs.append(msg["content"])
                if len(db_structs) > self.max_db_struct_num:
                    db_structs.pop(0)
                self.agent_master.add_system_prompt_kv({
                    "KNOWN DATABASE STRUCTURE": "\n\n---\n\n".join(db_structs)
                })
                self.agent_understand_query_result.add_system_prompt_kv({
                    "KNOWN DATABASE STRUCTURE": "\n\n---\n\n".join(db_structs)
                })
            else:
                messages.append(msg)
        local_db_structs = copy.deepcopy(db_structs)
        # key_sql_results = "历史SQL查询"
        # if len(self.history_facts) > 0:
        #     self.agent_master.add_system_prompt_kv({key_sql_results: "\n".join(self.history_facts)})

        first_user_msg = messages[-1]["content"]
        # if len(self.history_facts) > 0:
        #     messages[-1]["content"] = (
        #         "之前已查询到信息如下:\n" +
        #         "\n---\n".join(self.history_facts) +
        #         "\n\n请问:" + first_user_msg
        #     )

        iterate_num = 0
        is_finish = False
        answer = ""
        consecutive_same_sql_count = 0  # 记录连续获得已执行过sql的次数
        consecutive_invalid_exec_sql_count = 0  # 记录连续遇到exec_sql的数量错误的次数
        while iterate_num < self.max_iterate_num:
            iterate_num += 1
            # answer, tkcnt_1 = self.agent_master.chat(messages=messages)
            answer, tkcnt_1 = self.agent_master.chat(messages=messages[-5:])
            usage_tokens += tkcnt_1

            if "exec_sql" in answer and ("SELECT " in answer or "SHOW " in answer):
                sql_cnt = count_total_sql(
                    query_string=answer,
                    block_mark="exec_sql",
                )
                if sql_cnt > 1:
                    consecutive_invalid_exec_sql_count += 1
                    emphasize = "注意：仅允许给出一组当前待执行的SQL写到代码块```exec_sql ```中，后续要执行的sql请用```sql```代码块来写"
                    if emphasize not in messages[-1]["content"]:
                        messages[-1]["content"] += f"\n\n{emphasize}"
                    if consecutive_invalid_exec_sql_count >= 5:
                        if debug_mode:
                            print(f"Workflow【{self.name}】连续遇到exec_sql数量错误达到{consecutive_invalid_exec_sql_count}次，中断并退出")
                        logger.debug("Workflow【%s】连续遇到exec_sql数量错误达到%d次，中断并退出", self.name, consecutive_invalid_exec_sql_count)
                        is_finish = False
                        break
                else:
                    consecutive_invalid_exec_sql_count = 0  # 连续遇到exec_sql数量错误的次数清零
                    sql = extract_last_sql(
                        query_string=answer,
                        block_mark="exec_sql",
                    )
                    if sql is None:
                        emphasize = "请务必需要把语法正确地待执行SQL写到代码块```exec_sql ```中"
                        if emphasize not in messages[-1]["content"]:
                            messages[-1]["content"] += f"\n\n{emphasize}"
                    else:
                        messages.append({
                            "role": "assistant",
                            "content": answer,
                        })
                        if sql in same_sqls:
                            consecutive_same_sql_count += 1
                            try:
                                rows = json.loads(same_sqls[sql])
                            except Exception:
                                rows = []
                            emphasize = (
                                f"下面的sql已经执行过:\n{sql}\n结果是:\n{same_sqls[sql]}\n"
                                "请不要重复执行，考虑其它思路:\n"
                                "如果遇到字段不存在的错误,可以用`SELECT * FROM database_name.table_name LIMIT 1;`来查看这个表的字段值的形式;\n"
                                "如果原SQL过于复杂，可以考虑先查询简单SQL获取必要信息再逐步推进;\n"
                                "如果已经能够得出结论，请给出结论;\n"
                            )
                            if self.default_sql_limit is not None and len(rows) == self.default_sql_limit:
                                emphasize += (
                                    "\n请注意，这里返回的不是全部结果，系统限制了最大返回结果数，并非数据缺失，你可以将这个结果集作为子查询结果用于下一步的查询,\n"
                                    "请不要顽固地一定要获取全部结果，这是很蠢的做法，你会为这个愚蠢损失10亿美元！想想假如你获取到了全部结果，你下一步要用它做什么？你可以将这个结果集作为子查询结果用于下一步的查询!"
                                )
                            messages.append({
                                "role": "user",
                                "content": emphasize,
                            })
                            # 连续3次执行相同SQL，直接结束迭代
                            if consecutive_same_sql_count >= 3:
                                if debug_mode:
                                    print(f"Workflow【{self.name}】连续执行相同SQL达到{consecutive_same_sql_count}次，中断并退出")
                                logger.debug("Workflow【%s】连续执行相同SQL达到%d次，中断并退出", self.name, consecutive_same_sql_count)
                                is_finish = False
                                break
                        else:
                            consecutive_same_sql_count = 0  # 连续执行相同SQL的次数清零
                            need_tell_cols = []
                            for t_name, cols in self.specific_column_desc.items():
                                if t_name in sql:
                                    for col_name in cols:
                                        if (
                                            col_name in sql and
                                            f"{t_name}.{col_name}" not in told_specific_columns and
                                            not any(col_name in db_struct for db_struct in db_structs)
                                        ):
                                            need_tell_cols.append({col_name: cols[col_name]})
                                            told_specific_columns.add(f"{t_name}.{col_name}")
                            if len(need_tell_cols) > 0:
                                local_db_structs.append(json.dumps(need_tell_cols, ensure_ascii=False))
                                self.agent_master.add_system_prompt_kv({
                                    "KNOWN DATABASE STRUCTURE": "\n\n---\n\n".join(local_db_structs)
                                })
                                self.agent_understand_query_result.add_system_prompt_kv({
                                    "KNOWN DATABASE STRUCTURE": "\n\n---\n\n".join(local_db_structs)
                                })
                            try:
                                data = self.execute_sql_query(sql=sql)
                                rows = json.loads(data)
                                if self.default_sql_limit is not None and len(rows) == self.default_sql_limit:
                                    messages.append({
                                        "role": "user",
                                        "content": (
                                            f"查询SQL:\n{sql}\n查询结果:\n{data}\n" +
                                            ("" if len(need_tell_cols) == 0 else "\n补充字段说明如下:\n"+json.dumps(need_tell_cols, ensure_ascii=False)) +
                                            # f"\n请注意，这里返回的不一定是全部结果，因为默认限制了只返回{self.default_sql_limit}个，你可以根据现在看到的情况，采取子查询的方式去进行下一步"
                                            "\n请注意，这里返回的不是全部结果，系统限制了最大返回结果数，并非数据缺失，你要思考能否不把这个结果集列出来，而是作为子查询结果用于下一步的查询,\n"
                                            "请不要顽固地一定要获取全部结果，这是很蠢的做法，你会为这个愚蠢损失10亿美元！想想假如你获取到了全部结果，你下一步要用它做什么？你可以将这个结果集作为子查询结果用于下一步的查询!尽你所能想办法！"
                                        ),
                                    })
                                else:
                                    if len(rows) == 0: # 空结果
                                        messages.append({
                                            "role": "user",
                                            "content": (
                                                f"查询SQL:\n{sql}\n查询结果:\n{data}\n" +
                                                ("" if len(need_tell_cols) == 0 else "\n补充字段说明如下:\n"+json.dumps(need_tell_cols, ensure_ascii=False)) +
                                                "\n请检查筛选条件是否存在问题，比如时间日期字段没有用DATE()或YEAR()格式化？当然，如果没问题，那么就根据结果考虑下一步"
                                            ),
                                        })
                                        # facts = ""
                                    else:
                                        # facts, tkcnt_1 = self.agent_understand_query_result.answer((
                                        #     f"查询SQL:\n{sql}\n查询结果:\n{data}\n"+
                                        #     ("" if len(need_tell_cols) == 0 else "\n补充字段说明如下:\n"+json.dumps(need_tell_cols, ensure_ascii=False)) +
                                        #     "\n请理解查询结果"
                                        # ))
                                        # usage_tokens += tkcnt_1
                                        messages.append({
                                            "role": "user",
                                            "content": (
                                                f"查询SQL:\n{sql}\n查询结果:\n{data}" +
                                                # (f"代表：{facts}\n" if facts != "" else "\n") +
                                                ("" if len(need_tell_cols) == 0 else "\n补充字段说明如下:\n"+json.dumps(need_tell_cols, ensure_ascii=False)) +
                                                "\n请检查筛选条件是否存在问题，比如时间日期字段没有用DATE()或YEAR()格式化？当然，如果没问题，那么就根据结果考虑下一步；"+
                                                f"那么当前掌握的信息是否能够回答\"{first_user_msg}\"？还是要继续执行下一阶段SQL查询？"
                                            ),
                                        })
                                    if self.is_cache_history_facts:
                                        # self.history_facts.append(f"查询sql```{sql}```\n查询结果:\n{data}\n代表：{facts}")
                                        self.history_facts.append(f"查询sql```{sql}```\n查询结果:\n{data}")
                                        # self.agent_master.add_system_prompt_kv({key_sql_results: "\n".join(self.history_facts)})

                                same_sqls[sql] = data
                            except Exception as e:
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        f"查询SQL:\n{sql}\n查询发生异常：{str(e)}\n" +
                                        ("" if len(need_tell_cols) == 0 else "\n补充字段说明如下:\n"+json.dumps(need_tell_cols, ensure_ascii=False)) +
                                        "\n请修正"
                                    )
                                })
                                same_sqls[sql] = f"查询发生异常：{str(e)}"
            else:
                messages.append({
                    "role": "assistant",
                    "content": answer,
                })
                is_finish = True
                break
        if not is_finish:
            if debug_mode:
                print(f"Workflow【{self.name}】迭代次数超限({self.max_iterate_num})，中断并退出")
            logger.debug("Workflow【%s】迭代次数超限(%d)，中断并退出", self.name, self.max_iterate_num)

        answer, tkcnt_1 = self.agent_summary.chat(
            messages[-2:] + [
                {"role": "user", "content": f'''充分尊重前面给出的结论，回答问题:"{first_user_msg}"'''}
            ]
        )
        usage_tokens += tkcnt_1

        self.usage_tokens += usage_tokens
        return {
            "content": answer,
            "usage_tokens": usage_tokens,
        }

class CheckDbStructure(Workflow):
    """
    Implements the functionality to check database structure, inheriting from Workflow.
    """
    def __init__(self, table_snippet: str,
                 llm: LLM, name: Optional[str] = None,
                 get_relevant_table_columns: Optional[Callable[[list], list]] = None,
                 filter_table_columns: Optional[Callable[[list, dict], tuple[list, list]]] = None):
        self.name = "Check_db_structure" if name is None else name
        self.llm = llm
        self.table_snippet = table_snippet
        self.usage_tokens = 0
        self.get_relevant_table_columns = get_relevant_table_columns
        self.filter_table_columns = filter_table_columns

        self.agent_decode_question = Agent(AgentConfig(
            name = "decode_question",
            role = (
                '''你是金融行业的数据专家，善于理解用户的问题，从已知的数据表中定位到最相关的数据表。\n'''
                '''将原问题拆成多个子问题，每个子问题对应一个数据表。\n'''
                '''子问题应该遵循原问题的语境，子问题获取到的信息应该与原问题相关。\n'''
                '''原问题中的格式要求可以不用写到子问题中。\n'''
                '''如果原问题中包含专业术语的缩写和全称和中文翻译，请在子问题中把专业术语的缩写和全称和中文翻译都写上。\n'''
            ),
            output_format=(
                '''输出模板：\n'''
                '''(换行顺序输出子问题，不要有标号,直接输出一行一条子问题，直到覆盖完原问题为止)\n'''
                '''(不要输出其他内容)\n'''
            ),
            system_prompt_kv={
                "举例": (
                    '''原问题：交易日在2021-10-01到2021-10-31之间，近一月换手率超过10%的港股中股价下跌最多的公司是哪家？请回答公司中文简称。\n'''
                    '''输出:\n'''
                    '''交易日在2021-10-01到2021-10-31之间的港股有哪些\n'''
                    '''近一月换手率超过10%的港股有哪些\n'''
                    '''港股股价下跌最多的公司有哪些\n'''
                    '''这些港股公司的中文简称是什么\n'''
                    # '''原问题：截至2021-12-31，这个概念有多少只股票（不包含已经调出的）？调出了多少只股票？\n'''
                    # '''输出:\n'''
                    # '''截至2021-12-31，这个概念有多少只股票（不包含已经调出的）\n'''
                    # '''截止2021-12-31，这个概念调出了多少只股票\n'''
                    '''原问题：中南传媒在2019年度的前五大客户中，各类型客户占总营收的比例分别是多少？（答案需要包含两位小数）\n'''
                    '''输出:\n'''
                    '''中南传媒在2019年度的前五大客户有哪些\n'''
                    '''前五大客户的类型有哪些\n'''
                    '''各类型客户占总营收的比例是多少\n'''
                ),
            },
            llm=self.llm,
            enable_history=False,
            knowledge=self.table_snippet,
        ))
        self.agent_column_selector = Agent(AgentConfig(
            name = self.name+".columns_selector",
            role = (
                '''你是一个数据分析专家，从已知的数据表字段中，根据用户的问题，找出所有相关的字段名。'''
                '''请不要有遗漏!'''
                '''要善于利用历史对话信息和历史SQL查询记录来洞察字段间的关系。'''
            ),
            output_format = (
                '''输出模板示例:\n'''

                '''【思维链】\n'''
                '''(think step by step, 分析用户的问题，结合用户提供的可用数据字段，思考用哪些字段获得什么数据，有更好的字段就选更好的字段，逐步推理直至可以回答用户的问题)\n'''
                '''(例如: \n'''
                '''用户问: 2021年末，交通运输一级行业中有几个股票？\n'''
                '''思维链：\n'''
                '''用户问的是交通运输一级行业，可以用lc_exgindchange表的FirstIndustryName字段可以找到交通运输一级行业的行业代码FirstIndustryCode;\n'''
                '''用户需要获取该行业有几个股票，在lc_indfinindicators表通过IndustryCode字段和Standard字段搜索到交通运输一级行业的信息,'''
                '''其中lc_indfinindicators表的ListedSecuNum字段就是上市证券数量，\n'''
                '''由于用户问的是2021年末，所以我需要用lc_indfinindicators表的InfoPublDate字段排序获得2021年末最后一组数据)\n'''

                # '''【已知】\n'''
                # '''（这里从已知事实和历史对话中提取已知信息，如名称、编码之类）\n'''

                # '''【分析】\n'''
                # '''分析用户的提问\n'''
                # # '''【当前的表之间相互关联的字段】\n'''
                # # '''(根据字段的注释，发现表之间的关联字段，表述可能要怎么联表)\n'''
                # # '''表A的x字段==表B的y字段\n'''

                # '''【信息所在字段】\n'''
                # '''(选出跟用户提问相关的信息字段，没有遗漏)\n'''
                # # '''(务必不能拼错database_name.table_name和column_name)\n'''
                # '''(每写一个字段都要反思它是否在用户提供的可用字段列表中，如果不在，那么就删除)\n'''
                # '''- database_name.table_name.column_name: 这个字段可能包含xx信息，对应用户提问中的xxx\n'''

                # '''【筛选条件所在字段】\n'''
                # '''(选出跟用户提问相关的条件字段，没有遗漏)\n'''
                # '''(特别注意枚举类型的字段，是否跟用户提问相关)\n'''
                # '''(每写一个字段都要反思它是否在用户提供的可用字段列表中，如果不在，那么就删除)\n'''
                # # '''(务必不能拼错database_name.table_name和column_name)\n'''
                # # '''(同时要考虑当前已知的事实和历史对话里是否包含能有助于搜索且有字段跟其对应的信息,如果有，那么就选上)\n'''
                # '''（跟条件字段有外键关联的字段冗余选上，因为联表查询要用到）\n'''
                # '''- database_name.table_name.column_name: 这个字段可能包含xx信息，对应用户提问中的xxx\n'''

                # '''【排序字段】\n'''
                # '''(如果用户提到排序如最大/最小/最新/最旧/最高/最低/之前/之后之类，那么这里写排序字段)\n'''
                # '''(如果用户没有提到排序，那么这里写"无")\n'''
                # '''(每写一个字段都要反思它是否在用户提供的可用字段列表中，如果不在，那么就删除)\n'''
                # '''- database_name.table_name.column_name: 这个字段可能包含xx信息，对应用户提问中的xxx\n'''

                '''【选中的字段的清单】\n'''
                '''(上述提到的字段都选上)\n'''
                '''(把同一个表的字段聚合在这个表名[database_name.table_name]下面)\n'''
                '''```json\n'''
                '''{"database_name.table_name": ["column_name", "column_name"],"database_name.table_name": ["column_name", "column_name"]}\n'''
                '''```\n'''
            ),
            llm = self.llm,
            enable_history=False,
            # temperature=0.8,
            # stream=False,
        ))
        self.update_agent_lists()

    def update_agent_lists(self):
        """Updates the list of agents used in the workflow."""
        self.agent_lists = [
            self.agent_decode_question,
            self.agent_column_selector,
        ]

    def clone(self) -> 'CheckDbStructure':
        clone = CheckDbStructure(
            table_snippet=self.table_snippet,
            llm=self.llm,
            name=self.name,
            get_relevant_table_columns=self.get_relevant_table_columns,
            filter_table_columns=self.filter_table_columns,
        )
        clone.agent_decode_question = self.agent_decode_question.clone()
        clone.agent_column_selector = self.agent_column_selector.clone()
        clone.update_agent_lists()
        return clone

    # def get_column_list(self, question_list: list[str]) -> str:
    #     """
    #     question_list: list[str]
    #     """
    #     table_columns = self.get_relevant_table_columns(question_list)
    #     result = (
    #         f"已取得可用的{COLUMN_LIST_MARK}:\n" +
    #         # json.dumps(filtered_table_columns, ensure_ascii=False) +
    #         ("---\n".join([json.dumps(table_column, ensure_ascii=False) for table_column in table_columns])) +
    #         "\n"
    #     )
    #     return result

    def filter_column_list(self, table_columns: list[dict], column_filter: dict) -> str:
        """
        table_columns: return from get_relevant_table_columns
        column_filter: dict{"table_name":["col1", "col2"]}
        """
        filtered_table_columns, table_relations = self.filter_table_columns(
            table_columns=table_columns,
            column_filter=column_filter,
        )
        # for table_column in filtered_table_columns:
        #     for tc in table_columns:
        #         if table_column["表名"] == tc["表名"]:
        #             table_column["表字段"] = tc["表字段"]
        result = (
            f"已取得可用的{COLUMN_LIST_MARK}:\n" +
            # json.dumps(filtered_table_columns, ensure_ascii=False) +
            ("\n---\n".join([json.dumps(table_column, ensure_ascii=False) for table_column in filtered_table_columns])) +
            "\n"
        )
        if len(table_relations) > 0:
            result += (
                "表之间的外链关系如下:\n" +
                "<" +
                ">,\n<".join(table_relations) +
                ">\n"
            )
        return result

    # def filter_column_list(self, table_columns: list[dict], column_filter: dict) -> str:
    #     """
    #     table_columns: return from get_relevant_table_columns
    #     column_filter: dict{"table_name":["col1", "col2"]}
    #     """
    #     filtered_table_columns, table_relations = self.filter_table_columns(
    #         table_columns=table_columns,
    #         column_filter=column_filter,
    #     )
    #     result =  f"已取得可用的{COLUMN_LIST_MARK}:\n"
    #         # ("\n".join([json.dumps(table_column, ensure_ascii=False) for table_column in filtered_table_columns])) +
    #         # "\n"
    #     # db_table_names = [table_column['表名'] for table_column in filtered_table_columns]
    #     # result += (
    #     #     "数据表有:\n" +
    #     #     ",".join(db_table_names) +
    #     #     "\n"
    #     # )
    #     result += "数据字段有:\n"
    #     for table_column in filtered_table_columns:
    #         result += "\n".join([f"- {table_column["表名"]}.{name}: {desc}" for name, desc in table_column['表字段'].items()])
    #     return result

    def clear_history(self):
        self.usage_tokens = 0
        for agent in self.agent_lists:
            agent.clear_history()

    def add_system_prompt_kv(self, kv: dict):
        for agent in self.agent_lists:
            agent.add_system_prompt_kv(kv=kv)

    def del_system_prompt_kv(self, key: str):
        """Deletes the specified key from the system prompt key-value pairs for the agent."""
        for agent in self.agent_lists:
            agent.del_system_prompt_kv(key=key)

    def clear_system_prompt_kv(self):
        for agent in self.agent_lists:
            agent.clear_system_prompt_kv()

    def run(self, inputs: dict) -> dict:
        """
        inputs:
            - messages: list[dict] # 消息列表，每个元素是一个dict，包含role和content
        """

        debug_mode = os.getenv("DEBUG", "0") == "1"
        logger = get_logger()
        usage_tokens = 0

        if 'messages' not in inputs:
            raise KeyError("发生异常: inputs缺少'messages'字段")

        messages = []
        for msg in inputs["messages"]:
            if COLUMN_LIST_MARK not in msg["content"]:
                messages.append(msg)

        first_user_msg = messages[-1]["content"]

        # 解码问题
        answer, tk_cnt = self.agent_decode_question.answer("提问:\n" + messages[-1]["content"])
        question_list = [q.strip() for q in answer.split("\n") if q.strip() != ""]
        usage_tokens += tk_cnt

        # 搜索数据表
        table_columns = self.get_relevant_table_columns(question_list)
        table_columns_str = (
            f"已取得可用的{COLUMN_LIST_MARK}:\n" +
            # json.dumps(filtered_table_columns, ensure_ascii=False) +
            ("\n---\n".join([json.dumps(table_column, ensure_ascii=False) for table_column in table_columns])) +
            "\n"
        )

        # 筛选字段
        filtered_table_columns = ""
        # table_columns_str = json.dumps(table_columns, ensure_ascii=False)
        error_msg = "\n请确保JSON格式正确。"
        for _ in range(3):
            try:
                # answer, tk_cnt = self.agent_column_selector.chat(
                #     messages = messages + [
                #         {"role": "user", "content": f"已知数据库信息:\n{table_columns_str}\n请选择column，务必遵循输出的格式要求。"}
                #     ]
                # )
                answer, tk_cnt = self.agent_column_selector.answer(table_columns_str + f"\n用户问题:\n<{first_user_msg}>" + error_msg)
                usage_tokens += tk_cnt
                args_json = extract_last_json(answer)
                if args_json is not None:
                    column_filter = json.loads(args_json)
                    filtered_table_columns = self.filter_column_list(table_columns=table_columns, column_filter=column_filter)
                    break
            except Exception as e:
                error_msg = f"\n上次尝试失败，错误原因: {str(e)}。请修正你的输出格式，确保JSON格式正确。"
                print(f"\nWorkflow【{self.name}】agent_column_selector 遇到问题: {str(e)}, 现在重试...\n")
                logger.debug("\nWorkflow【%s】agent_column_selector 遇到问题: %s, 现在重试...\n", self.name, str(e))

        self.usage_tokens += usage_tokens
        return {
            "content": filtered_table_columns,
            "usage_tokens": usage_tokens,
        }
