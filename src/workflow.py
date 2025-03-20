"""
This module defines the Workflow abstract base class and its implementation, RecallDbInfo,
which handles recalling database information using various agents.
"""

import json
import os
import copy
from abc import ABC, abstractmethod
from typing import Callable, Optional

from src.log import get_logger
from src.llm import LLM
from src.agent import Agent, AgentConfig
from src.utils import generate_markdown_table, extract_last_sql, extract_last_json, COLUMN_LIST_MARK, count_total_sql

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
                '''你是一个严谨的数据库专家，擅长通过分步拆解的方式获取数据。你遵循以下原则：\n'''
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
                '''- 如果确认查找的方式是正确的，那么可以接受空结果!!!\n'''
                '''- 每次交互只处理一个原子查询操作\n'''
                '''- 连续步骤必须显式依赖前序查询结果\n'''
                '''- 如果总是执行失败，尝试更换思路，拆解成简单SQL，逐步执行确认\n'''
                '''- 擅于使用DISTINC，尤其当发现获取的结果存在重复，去重后不满足期望的数量的时候，比如要查询前10个结果，但是发现结果里存在重复，那么就要考虑使用DISTINC重新查询\n'''
                '''- 在MySQL查询中，使用 WHERE ... IN (...) 不能保持传入列表的顺序，可通过 ORDER BY FIELD(列名, 值1, 值2, 值3, ...) 强制按指定顺序排序。\n'''
                '''- 对于求中位数的查询，通常会使用 ROW_NUMBER() 或类似的方法来代替 LIMIT 这种复杂的动态限制，如果确实需要获取中位数，你需要确保能动态计算并获取中位位置的记录\n'''
            ),
            output_format = (
                '''分阶段输出模板：\n'''
                '''【已知信息】\n'''
                '''（这里写当前已知的所有事实信息）\n'''
                '''【当前阶段要获取的信息】\n'''
                '''(如果无需继续执行SQL，那么这里写"无")\n'''
                '''【信息所在字段】\n'''
                '''(如果无需继续执行SQL，那么这里写"无")\n'''
                '''(如果已知字段里缺少需要的字段，那么用`SELECT * FROM database_name.table_name LIMIT 1;`来了解这个表的字段值的形式)\n'''
                '''【筛选条件所在字段】\n'''
                '''(如果无需继续执行SQL，那么这里写"无")\n'''
                '''（认真检查，时间日期过滤必须对字段名进行格式化：`DATE(column_name) (op) 'YYYY-MM-DD'` 或 `YEAR(colum_name) (op) 'YYYY'`）\n'''
                '''【SQL语句的思路】\n'''
                '''(接着要执行的SQL的思路)\n'''
                '''(如果无需继续执行SQL，那么这里写"无")\n'''
                '''（这里必须使用已知的数据库表和字段，不能假设任何数据表或字典）\n'''
                '''【执行SQL语句】\n'''
                '''（唯一允许的SQL代码块，如果当前阶段无需继续执行SQL，那么这里写"无"）\n'''
                '''（如果涉及到数学运算即便是用已知的纯数字做计算，也可以通过SQL语句来进行，保证计算结果的正确性，如`SELECT 1+1 AS a`）\n'''
                '''（这里必须使用已知的数据库表和字段，不能假设任何数据表或字典）\n'''
                '''```exec_sql\n'''
                '''SELECT [精准字段] \n'''
                '''FROM [完整表名] \n'''
                '''WHERE [条件原子化] \n'''
                '''LIMIT [强制行数]\n'''
                '''【上述SQL语句的含义】\n'''
                '''（如果当前阶段无执行SQL，那么这里写"无"）\n'''
            ),
            llm = llm,
            enable_history=True,
            # temperature = 0.8,
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

        first_user_msg = messages[-1]["content"]
        if len(self.history_facts) > 0:
            messages[-1]["content"] = (
                "之前已查询到信息如下:\n" +
                "\n---\n".join(self.history_facts) +
                "\n\n请问:" + first_user_msg
            )

        iterate_num = 0
        is_finish = False
        answer = ""
        while iterate_num < self.max_iterate_num:
            iterate_num += 1
            answer, tkcnt_1 = self.agent_master.chat(messages=messages)
            usage_tokens += tkcnt_1

            if "```exec_sql" in answer and ("SELECT " in answer or "SHOW " in answer):
                sql_cnt = count_total_sql(
                    query_string=answer,
                    block_mark="exec_sql",
                )
                if sql_cnt > 1:
                    emphasize = "一次仅允许给出一组待执行的SQL写到代码块```exec_sql ```中"
                    if emphasize not in messages[-1]["content"]:
                        messages[-1]["content"] += f"\n\n{emphasize}"
                else:
                    sql = extract_last_sql(
                        query_string=answer,
                        block_mark="exec_sql",
                    )
                    if sql is None:
                        emphasize = "请务必需要把待执行的SQL写到代码块```exec_sql ```中"
                        if emphasize not in messages[-1]["content"]:
                            messages[-1]["content"] += f"\n\n{emphasize}"
                    else:
                        messages.append({
                            "role": "assistant",
                            "content": answer,
                        })
                        if sql in same_sqls:
                            emphasize = (
                                f"下面的sql已经执行过:\n{sql}\n结果是:\n{same_sqls[sql]}\n"
                                "请不要重复执行，考虑其它思路:\n"
                                "如果遇到字段不存在的错误,可以用`SELECT * FROM database_name.table_name LIMIT 1;`来查看这个表的字段值的形式;\n"
                                "如果原SQL过于复杂，可以考虑先查询简单SQL获取必要信息再逐步推进;\n"
                                "如果已经能够得出结论，请给出结论;\n"
                            )
                            messages.append({
                                "role": "user",
                                "content": emphasize,
                            })
                        else:
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
                                if len(rows) == 0: # 空结果
                                    messages.append({
                                        "role": "user",
                                        "content": (
                                            f"查询SQL:\n{sql}\n查询结果:\n{data}\n" +
                                            ("" if len(need_tell_cols) == 0 else "\n补充字段说明如下:\n"+json.dumps(need_tell_cols, ensure_ascii=False)) +
                                            "\n请检查筛选条件是否存在问题，比如时间日期字段没有用DATE()或YEAR()格式化？当然，如果没问题，那么就根据结果考虑下一步"
                                        ),
                                    })
                                elif self.default_sql_limit is not None and len(rows) == self.default_sql_limit:
                                    messages.append({
                                        "role": "user",
                                        "content": (
                                            f"查询SQL:\n{sql}\n查询结果:\n{data}\n" +
                                            ("" if len(need_tell_cols) == 0 else "\n补充字段说明如下:\n"+json.dumps(need_tell_cols, ensure_ascii=False)) +
                                            f"\n请注意，这里返回的不一定是全部结果，因为默认限制了只返回{self.default_sql_limit}个，你可以根据现在看到的情况，采取子查询的方式去进行下一步"
                                        ),
                                    })
                                else:
                                    facts, tkcnt_1 = self.agent_understand_query_result.answer((
                                        f"查询SQL:\n{sql}\n查询结果:\n{data}\n"+
                                        ("" if len(need_tell_cols) == 0 else "\n补充字段说明如下:\n"+json.dumps(need_tell_cols, ensure_ascii=False)) +
                                        "\n请理解查询结果"
                                    ))
                                    if self.is_cache_history_facts:
                                        self.history_facts.append(facts)
                                    usage_tokens += tkcnt_1
                                    messages.append({
                                        "role": "user",
                                        "content": (
                                            f"查询SQL:\n{sql}\n查询结果:\n{data}\n" +
                                            ("" if len(need_tell_cols) == 0 else "\n补充字段说明如下:\n"+json.dumps(need_tell_cols, ensure_ascii=False)) +
                                            (f"\n{facts}\n" if facts != "" else "\n") +
                                            "\n请检查筛选条件是否存在问题，比如时间日期字段没有用DATE()或YEAR()格式化？当然，如果没问题，那么就根据结果考虑下一步；"+
                                            f"那么当前掌握的信息是否能够回答\"{first_user_msg}\"？还是要继续执行下一阶段SQL查询？"
                                        ),
                                    })

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
    def __init__(self, dbs_info: list[dict], db_table: dict, table_column: dict,
                 db_selector_llm: LLM, table_selector_llm: LLM,
                 column_selector_llm: LLM,  name: Optional[str] = None,
                 db_select_post_process: Optional[Callable[[list], list]] = None,
                 table_select_post_process: Optional[Callable[[list], list]] = None,
                 import_column_names: Optional[set] = None,
                 foreign_key_hub: Optional[dict] = None):
        self.name = "Check_db_structure" if name is None else name
        self.dbs_info = dbs_info
        self.db_table = db_table
        self.table_column = table_column
        self.usage_tokens = 0
        self.import_column_names = import_column_names if import_column_names is not None else {}
        self.db_select_post_process = db_select_post_process
        self.table_select_post_process = table_select_post_process
        self.foreign_key_hub = foreign_key_hub if foreign_key_hub is not None else {}

        self.agent_db_selector = Agent(AgentConfig(
            name = self.name+".db_selector",
            role = (
                '''你是一个数据分析专家。根据用户的提问，从已知的数据库中，选出一个或多个数据库名，'''
                '''判断可以从这些库中获取到用户所需要的信息。'''
                '''请选择能最快获取到用户所需信息的数据库名，不要舍近求远。只需要说明思考过程并给出数据库名即可。'''
            ),
            output_format = (
                '''输出模板示例:\n'''
                '''【分析】\n'''
                '''分析用户的提问\n'''
                '''【选中的数据库】\n'''
                '''（选出必要的数据库，不是越多越好）\n'''
                '''- database_name: 这个数据库包含哪些会被用到的信息\n'''
                '''【选中的数据库的清单】\n'''
                '''```json\n'''
                '''["database_name", "database_name"]\n'''
                '''```\n'''
            ),
            llm = db_selector_llm,
            knowledge=json.dumps(self.dbs_info, ensure_ascii=False),
            enable_history=False,
            # stream=False,
        ))
        self.agent_table_selector = Agent(AgentConfig(
            name = self.name+".table_selector",
            role = (
                '''你是一个数据分析专家，从已知的数据表中，根据需要选出一个或多个表名。'''
                '''请尽可能选择能最合适的表名。'''
            ),
            output_format = (
                '''输出模板示例:\n'''
                '''【分析】\n'''
                '''分析用户的提问\n'''
                '''【选中的数据表】\n'''
                '''（选出必要的数据表，不是越多越好）\n'''
                '''- database_name.table_name: 这个数据表包含哪些会被用到的信息\n'''
                '''【选中的数据库表的清单】\n'''
                '''```json\n'''
                '''["database_name.table_name", "database_name.table.name"]\n'''
                '''```\n'''
                '''给出的表名应该是库名和表名的组合(database_name.table_name)'''
            ),
            llm = table_selector_llm,
            enable_history=False,
            # stream=False,
        ))
        self.agent_column_selector = Agent(AgentConfig(
            name = self.name+".columns_selector",
            role = (
                '''你是一个数据分析专家，从已知的数据表字段中，根据用户的问题，找出所有相关的字段名。'''
                '''请不要有遗漏!'''
            ),
            output_format = (
                '''输出模板示例:\n'''
                '''【分析】\n'''
                '''分析用户的提问\n'''
                '''【当前的表之间相互关联的字段】\n'''
                '''（考虑表之间的关联，把关联的字段选出来）\n'''
                '''表A和表B之间: ...\n'''
                '''表A和表C之间: ...\n'''
                '''【信息所在字段】\n'''
                '''（选出跟用户提问相关的信息字段，没有遗漏）\n'''
                '''- database_name.table_name.column_name: 这个字段可能包含xx信息，对应用户提问中的xxx\n'''
                '''【筛选条件所在字段】\n'''
                '''（选出跟用户提问相关的条件字段，没有遗漏）\n'''
                '''（跟条件字段有外键关联的字段冗余选上，因为联表查询要用到）\n'''
                '''- database_name.table_name.column_name: 这个字段可能包含xx信息，对应用户提问中的xxx\n'''
                '''【选中的字段的清单】\n'''
                '''（把同一个表的字段聚合在这个表名[database_name.table_name]下面）\n'''
                '''```json\n'''
                '''{"database_name.table_name": ["column_name", "column_name"],"database_name.table_name": ["column_name", "column_name"]}\n'''
                '''```\n'''
            ),
            llm = column_selector_llm,
            enable_history=False,
            # stream=False,
        ))
        self.update_agent_lists()

    def update_agent_lists(self):
        """Updates the list of agents used in the workflow."""
        self.agent_lists = [
            self.agent_db_selector,
            self.agent_table_selector,
            self.agent_column_selector,
        ]

    def clone(self) -> 'CheckDbStructure':
        clone = CheckDbStructure(
            dbs_info=self.dbs_info,
            db_table=self.db_table,
            table_column=self.table_column,
            db_selector_llm=self.agent_db_selector.cfg.llm,
            table_selector_llm=self.agent_table_selector.cfg.llm,
            column_selector_llm=self.agent_column_selector.cfg.llm,
            name=self.name,
            db_select_post_process=self.db_select_post_process,
            table_select_post_process=self.table_select_post_process,
            import_column_names=self.import_column_names,
            foreign_key_hub=self.foreign_key_hub,
        )
        clone.agent_db_selector = self.agent_db_selector.clone()
        clone.agent_table_selector = self.agent_table_selector.clone()
        clone.agent_column_selector = self.agent_column_selector.clone()
        clone.update_agent_lists()
        return clone

    def get_table_list(self, dbs: list[str]) -> str:
        """
        Retrieves a list of tables for each specified database.

        Parameters:
        dbs (list[str]): A list of database names.

        Returns:
        str: A formatted string containing the table information for each database.
        """
        # table_lists = []
        # for db in dbs:
        #     if db not in self.db_table:
        #         raise KeyError(f"发生异常: 数据库名`{db}`不存在")
        #     table_list = generate_markdown_table(
        #         self.db_table[db]['表'],
        #         {'表英文': '表名', '表中文': '含义', 'cols_summary': '注释'},
        #     )
        #     table_list = f"数据库名: {db}\n" + table_list
        #     table_lists.append(table_list)
        # result = "数据库表信息如下:\n" + "\n".join(table_lists)

        table_list = []
        for db_name in dbs:
            if db_name not in self.db_table:
                raise KeyError(f"发生异常: 数据库名`{db_name}`不存在")
            for table in self.db_table[db_name]['表']:
                table_list.append({
                    "表名": f"{db_name}."+table["表英文"],
                    "说明": table["cols_summary"]
                })
        result = (
            "数据库表信息如下:\n" +
            json.dumps(table_list, ensure_ascii=False)+
            "\n"
        )
        return result

    def get_column_list(self, tables: list[str]) -> str:
        """
        tables: list of table names, format is database_name.table_name
        """
        # key_title_map = {'column': '字段名', 'desc': '注释'}
        # column_lists = []
        # for table in tables:
        #     if '.' not in table or table.count('.') != 1:
        #         raise ValueError(f"发生异常: 表名`{table}`格式不正确，应该为`database_name.table_name`")
        #     db_name, table_name = table.split('.')
        #     if db_name not in self.db_table:
        #         raise KeyError(f"发生异常: 数据库名`{db_name}`不存在")
        #     if any(t['表英文'] == table_name for t in self.db_table[db_name]['表']):
        #         column_lists.append(f"数据表:\n{db_name}.{table_name}\n"+
        #             generate_markdown_table(self.table_column[table_name], key_title_map))
        # result = f"已取得可用的{COLUMN_LIST_MARK}:\n" + "\n---\n".join(column_lists)

        column_lists = []
        for table in tables:
            if '.' not in table or table.count('.') != 1:
                raise ValueError(f"发生异常: 表名`{table}`格式不正确，应该为database_name.table_name")
            db_name, table_name = table.split('.')
            if db_name not in self.db_table:
                raise KeyError(f"发生异常: 数据库名`{db_name}`不存在")
            if any(t['表英文'] == table_name for t in self.db_table[db_name]['表']):
                column_lists.append({
                    "表名": table,
                    "表字段": self.table_column[table_name],
                })
        result = (
            f"已取得可用的{COLUMN_LIST_MARK}:\n" +
            json.dumps(column_lists, ensure_ascii=False)+
            "\n"
        )
        return result

    def filter_column_list(self, tables: list[str], column_filter: dict) -> str:
        """
        tables: list of table names, format is database_name.table_name
        column_filter: dict{"table_name":["col1", "col2"]}
        """
        column_lists = []
        for table in tables:
            if '.' not in table or table.count('.') != 1:
                raise ValueError(f"发生异常: 表名`{table}`格式不正确，应该为database_name.table_name")
            if table not in column_filter:
                continue
            db_name, table_name = table.split('.')
            if db_name not in self.db_table:
                raise KeyError(f"发生异常: 数据库名`{db_name}`不存在")
            if any(t['表英文'] == table_name for t in self.db_table[db_name]['表']):
                column_list = {
                    "表名": table,
                    "表字段": [],
                }
                for col in self.table_column[table_name]:
                    if col["column"] in column_filter[table] or col["column"] in self.import_column_names:
                        column_list["表字段"].append(col)
                column_lists.append(column_list)
        for table, cols in self.foreign_key_hub.items():
            if table not in tables:
                column_list = {
                    "表名": table,
                    "表字段": [],
                }
                db_name, table_name = table.split('.')
                for col in self.table_column[table_name]:
                    if col["column"] in cols or col["column"] in self.import_column_names:
                        column_list["表字段"].append(col)
                column_lists.append(column_list)
        result = (
            f"已取得可用的{COLUMN_LIST_MARK}:\n" +
            json.dumps(column_lists, ensure_ascii=False)+
            "\n"
        )
        return result

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

        # 选择数据库
        args_json = None
        table_list = ""
        for _ in range(3):
            try:
                answer, tk_cnt = self.agent_db_selector.chat(
                    messages = messages + [
                        {"role": "user", "content": "请选择db，务必遵循输出的格式要求。"}
                    ]
                )
                usage_tokens += tk_cnt
                args_json = extract_last_json(answer)
                if args_json is not None:
                    dbs = json.loads(args_json)
                    if self.db_select_post_process is not None:
                        dbs = self.db_select_post_process(dbs)
                    table_list = self.get_table_list(dbs=dbs)
                    break
            except Exception as e:
                print(f"\nWorkflow【{self.name}】agent_db_selector 遇到问题: {str(e)}, 现在重试...\n")
                logger.debug("\nWorkflow【%s】agent_db_selector 遇到问题: %s, 现在重试...\n", self.name, str(e))
        if table_list == "":
            logger.debug("\n Fail to get table_list, Skip\n")
            return {
                "content": "",
                "usage_tokens": usage_tokens,
            }

        # 选择数据表
        column_list = ""
        for _ in range(3):
            try:
                answer, tk_cnt = self.agent_table_selector.chat(
                    messages = messages + [
                        {"role": "user", "content": f"{table_list}\n请选择table，务必遵循输出的格式要求。"}
                    ]
                )
                usage_tokens += tk_cnt
                args_json = extract_last_json(answer)
                if args_json is not None:
                    tables = json.loads(args_json)
                    if self.table_select_post_process is not None:
                        tables = self.table_select_post_process(tables)
                    column_list = self.get_column_list(tables=tables)
                    break
            except Exception as e:
                print(f"\nWorkflow【{self.name}】agent_table_selector 遇到问题: {str(e)}, 现在重试...\n")
                logger.debug("\nWorkflow【%s】agent_table_selector 遇到问题: %s, 现在重试...\n", self.name, str(e))
        if column_list == "":
            logger.debug("\n Fail to get column_list, Skip\n")
            return {
                "content": "",
                "usage_tokens": usage_tokens,
            }

        # 筛选字段
        filtered_column_list = ""
        for _ in range(3):
            try:
                answer, tk_cnt = self.agent_column_selector.chat(
                    messages = messages + [
                        {"role": "user", "content": f"{column_list}\n请选择column，务必遵循输出的格式要求。"}
                    ]
                )
                usage_tokens += tk_cnt
                args_json = extract_last_json(answer)
                if args_json is not None:
                    column_filter = json.loads(args_json)
                    filtered_column_list = self.filter_column_list(tables=tables, column_filter=column_filter)
                    break
            except Exception as e:
                print(f"\nWorkflow【{self.name}】agent_column_selector 遇到问题: {str(e)}, 现在重试...\n")
                logger.debug("\nWorkflow【%s】agent_column_selector 遇到问题: %s, 现在重试...\n", self.name, str(e))

        self.usage_tokens += usage_tokens
        return {
            "content": filtered_column_list,
            "usage_tokens": usage_tokens,
        }
