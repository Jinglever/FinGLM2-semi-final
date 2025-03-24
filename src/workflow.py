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
import concurrent.futures

from src.log import get_logger
from src.llm import LLM
from src.agent import Agent, AgentConfig
from src.utils import extract_last_sql, extract_last_json, COLUMN_LIST_MARK, count_total_sql, extract_all_sqls, extract_tables_and_columns

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
                # '''2. 每个交互周期仅执行单条SQL语句，确保可维护性和性能\n'''
                '''3. 已经尝试过的方案不要重复尝试，如果没有更多可以尝试的方案，就说明情况并停止尝试。\n'''
                '''**!!绝对执行规则!!**\n'''
                # '''- 每次响应有且仅有一个 ```exec_sql 代码块\n'''
                '''- 即使需要多步操作，也必须分次请求执行\n'''
                # '''- 出现多个SQL语句将触发系统级阻断\n'''
                '''- 不使用未知的表名和字段名\n'''
                '''- 获取任何实体或概念，如果它在同一张表里存在唯一编码，要顺便把它查询出来备用\n'''
                '''- 不准写插入语句\n'''
                '''- 所有SQL语句必须使用别名(单个字母表示），否则无法区分字段属于哪个表\n'''
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
                '''    8. 联表查询用到的字段是否正确，检查字段的数据示例\n'''
                '''- 如果确认查找的方式是正确的，那么可以接受空结果!!!\n'''
                # '''- 每次交互只处理一个原子查询操作\n'''
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

                # '''【已知信息】\n'''
                # '''（这里写当前已知的所有事实信息，尤其要注重历史对话中的信息）\n'''

                # '''【用户的问题】\n'''
                # '''(这里复述用户的问题，防止遗忘)\n'''

                '''【思维链】\n'''
                '''(如果已知信息已经可以回答用户的问题，那么不要继续思考，这里可写"已知信息已经可以回答用户的问题")\n'''
                '''(think step by step, 分析用户的问题，结合用户提供的可用数据字段，思考用哪些字段获得什么数据，遵循用户的指令，逐步推理直至可以回答用户的问题)\n'''
                # '''(例如: \n'''
                # '''用户问: 2021年末，交通运输一级行业中有几个股票？\n'''
                # '''思维链：\n'''
                # '''我们下一个要获取的信息是交通运输一级行业的代码是什么，'''
                # '''可以用lc_exgindchange表的FirstIndustryName字段可以找到交通运输一级行业的行业代码FirstIndustryCode;\n'''
                # '''我们下一个要获取的信息是交通运输一级行业的股票有多少个，'''
                # '''在lc_indfinindicators表通过IndustryCode字段和Standard字段(41-申万行业分类2021版)搜索到交通运输一级行业的信息,'''
                # '''其中lc_indfinindicators表的ListedSecuNum字段就是上市证券数量，\n'''
                # '''由于用户问的是2021年末，所以我需要用lc_indfinindicators表的InfoPublDate字段排序获得2021年末最后一组数据)\n'''

                '''【本阶段执行的SQL语句】\n'''
                # '''（唯一允许的SQL代码块，如果当前阶段无需继续执行SQL，那么这里写"无"）\n'''
                '''（如果已知信息已经可以回答用户的问题，那么这里写"无"）\n'''
                '''（如果涉及到数学运算即便是用已知的纯数字做计算，也可以通过SQL语句来进行，保证计算结果的正确性，如`SELECT 1+1 AS a`）\n'''
                '''(这里必须使用已知的数据库表和字段，不能假设任何数据表或字典)\n'''
                '''(所有表名都要赋予别名(单个字母表示），即使最简单的SQL，也要赋予别名，从而可以清晰地知道字段属于哪个表)\n'''
                # '''(当前待执行的一条或多条SQL写到代码块```exec_sql ```中，依赖当前执行结果的后续执行的sql请写到代码块```sql```中)\n'''
                '''(当前待执行的一条或多条SQL写到代码块```exec_sql ```中)\n'''
                '''```exec_sql\n'''
                '''SELECT [精准字段] \n'''
                '''FROM [完整表名] as [表别名] \n'''
                '''WHERE [条件原子化] \n'''
                '''LIMIT [强制行数]\n'''
                '''```\n'''
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
            # output_format=(
            #     '''- 用一段文字来回答，不要有任何markdown格式，不要有换行\n'''
            # ),
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
        first_user_msg = messages[-1]["content"]

        iterate_num = 0
        is_finish = False
        answer = ""
        consecutive_same_sql_count = 0  # 记录连续获得已执行过sql的次数
        while iterate_num < self.max_iterate_num:
            iterate_num += 1
            answer, tkcnt_1 = self.agent_master.chat(messages=messages)
            # answer, tkcnt_1 = self.agent_master.chat(messages=messages[-3:])
            usage_tokens += tkcnt_1

            messages.append({
                "role": "assistant",
                "content": answer,
            })
            if ("exec_sql" in answer or "```sql" in answer) and ("SELECT " in answer or "SHOW " in answer):
                sqls = extract_all_sqls(
                    query_string=answer,
                    block_mark="exec_sql",
                )
                if len(sqls) == 0:
                    emphasize = "请务必把要当前阶段要执行的SQL用正确的语法写到代码块```exec_sql ```中"
                    messages.append({
                        "role": "user",
                        "content": emphasize,
                    })
                else:
                    sqls = list(set(sqls))
                    success_sql_results = []
                    failed_sql_results = []
                    repeated_sql_results = []
                    has_same_sql = False
                    need_tell_cols = set()
                    for sql in sqls:
                        tables_and_columns = extract_tables_and_columns(sql)
                        for table, columns in tables_and_columns["table_to_columns"].items():
                            if table in self.specific_column_desc:
                                for column in columns:
                                    if column in self.specific_column_desc[table]:
                                        need_tell_cols.add(f"{table}.{column}的枚举值包括：{self.specific_column_desc[table][column]};")
                                for column in tables_and_columns["unassigned_columns"]:
                                    if column in self.specific_column_desc[table]:
                                        need_tell_cols.add(f"{table}.{column}的枚举值包括：{self.specific_column_desc[table][column]};")
                        if sql in same_sqls:
                            has_same_sql = True
                            emphasize = same_sqls[sql]
                            repeated_sql_results.append(emphasize)
                        else:
                            try:
                                data = self.execute_sql_query(sql=sql)
                                rows = json.loads(data)
                                if self.default_sql_limit is not None and len(rows) == self.default_sql_limit:
                                    emphasize = (
                                        f"查询SQL:\n{sql}\n查询结果:\n{data}\n" +
                                        # (
                                        #     "" if len(need_tell_cols) == 0 else
                                        #     "\n枚举字段说明以下面的为准，请务必再次检查取值是否正确，用错会损失100亿美元:\n"+
                                        #     "\n".join(need_tell_cols)
                                        # ) +
                                        "\n请注意，这里返回的不是全部结果，系统限制了最大返回结果数，并非数据缺失，你要思考能否不把这个结果集列出来，而是作为子查询结果用于下一步的查询,\n"
                                        "请不要顽固地一定要获取全部结果，这是很蠢的做法，你会为这个愚蠢损失10亿美元！想想假如你获取到了全部结果，你下一步要用它做什么？你可以将这个结果集作为子查询结果用于下一步的查询!尽你所能想办法！"
                                    )
                                else:
                                    emphasize = (
                                        f"查询SQL:\n{sql}\n查询结果:\n{data}"
                                        # (
                                        #     "" if len(need_tell_cols) == 0 else
                                        #     "\n枚举字段说明以下面的为准，请务必再次检查取值是否正确，用错会损失100亿美元:\n"+
                                        #     "\n".join(need_tell_cols)
                                        # )
                                    )
                                    if self.is_cache_history_facts:
                                        self.history_facts.append(f"查询sql```{sql}```\n查询结果:\n{data}")

                                success_sql_results.append(emphasize)
                                same_sqls[sql] = emphasize
                            except Exception as e:
                                emphasize = (
                                    f"查询SQL:\n{sql}\n查询发生异常：{str(e)}\n"
                                    # (
                                    #     "" if len(need_tell_cols) == 0 else
                                    #     "\n枚举字段说明以下面的为准，请务必再次检查取值是否正确，用错会损失100亿美元:\n"+
                                    #     "\n".join(need_tell_cols)
                                    # )
                                )
                                failed_sql_results.append(emphasize)
                                same_sqls[sql] = emphasize
                    if has_same_sql:
                        consecutive_same_sql_count += 1
                        # 连续3次执行相同SQL，直接结束迭代
                        if consecutive_same_sql_count >= 3:
                            if debug_mode:
                                print(f"Workflow【{self.name}】连续执行相同SQL达到{consecutive_same_sql_count}次，中断并退出")
                            logger.debug("Workflow【%s】连续执行相同SQL达到%d次，中断并退出", self.name, consecutive_same_sql_count)
                            is_finish = False
                            break
                    else:
                        consecutive_same_sql_count = 0
                    messages.append({
                        "role": "user",
                        "content": (
                            (
                                "\n下面是查询成功的SQL:\n" +
                                "<success_sql_results>\n" +
                                "\n---\n".join(success_sql_results) +
                                "\n</success_sql_results>\n"
                                if len(success_sql_results) > 0 else ""
                            ) +
                            (
                                "\n下面是已查询过的SQL，请不要再请求执行，考虑其它思路:\n" +
                                "<repeated_sql_results>\n" +
                                "\n---\n".join(repeated_sql_results) +
                                "\n</repeated_sql_results>\n"
                                if len(repeated_sql_results) > 0 else ""
                            ) +
                            (
                                "\n下面是查询失败的SQL，请检查和修正SQL语句(如果遇到字段不存在的错误,可以用`SELECT * FROM database_name.table_name LIMIT 1;`来查看这个表的字段值的形式):\n" +
                                "<failed_sql_results>\n" +
                                "\n---\n".join(failed_sql_results) +
                                "\n</failed_sql_results>\n"
                                if len(failed_sql_results) > 0 else ""
                            ) +
                            (
                                "" if len(need_tell_cols) == 0 else
                                "\n枚举字段说明以下面的为准，请务必再次检查取值是否正确，用错会损失100亿美元:\n"+
                                "\n".join(need_tell_cols)
                            ) +
                            "\n请检查筛选条件是否存在问题，比如时间日期字段没有用DATE()或YEAR()格式化？是否用SUM()的同时取了一个错误的日期范围(如<=some_date)？当然，如果没问题，那么就根据结果考虑下一步；"+
                            # f"\n那么当前掌握的信息是否能够回答\"{first_user_msg}\"？\n还是要继续执行下一阶段SQL查询？"
                            f"\n那么当前掌握的信息是否能够回答下面的问题了呢：\n<{first_user_msg}>"
                        ),
                    })
            else:
                is_finish = True
                break
        if not is_finish:
            if debug_mode:
                print(f"Workflow【{self.name}】迭代次数超限({self.max_iterate_num})，中断并退出")
            logger.debug("Workflow【%s】迭代次数超限(%d)，中断并退出", self.name, self.max_iterate_num)

        answer, tkcnt_1 = self.agent_summary.chat(
            messages + [
                {"role": "user", "content": f'''充分尊重前面给出的结论，回答问题:\n<{first_user_msg}>'''},
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
                 filter_table_columns: Optional[Callable[[dict], tuple[list, list]]] = None,
                 get_db_info: Optional[Callable[[], str]] = None,
                 get_table_list: Optional[Callable[[list], str]] = None,
                 get_column_list: Optional[Callable[[list], str]] = None,
                 validate_column_filter: Optional[Callable[[dict], str]] = None,
                 use_concurrency: bool = False,
                 print_table_column: Optional[Callable[[dict], str]] = None,
                 enable_llm_search: bool = True,
                 enable_vector_search: bool = True):
        self.name = "Check_db_structure" if name is None else name
        self.llm = llm
        self.table_snippet = table_snippet
        self.usage_tokens = 0
        self.get_relevant_table_columns = get_relevant_table_columns
        self.filter_table_columns = filter_table_columns
        self.get_db_info = get_db_info
        self.get_table_list = get_table_list
        self.get_column_list = get_column_list
        self.validate_column_filter = validate_column_filter
        self.use_concurrency = use_concurrency
        self.print_table_column = print_table_column
        self.enable_llm_search = enable_llm_search
        self.enable_vector_search = enable_vector_search
        self.agent_fix_column_selection = Agent(AgentConfig(
            name = "fix_column_selection",
            role = (
                '''你是金融数据库专家，负责审核和修正其他agent选择的数据表和字段。\n'''
                '''你的主要任务是确保所有表名和字段名的准确性，以及它们之间的正确关联。\n'''
                '''你需要仔细检查以下几点：\n'''
                '''1. 字段名是否拼写正确 - 如果发现错误，请提供正确的字段名\n'''
                '''2. 表名与字段名的关联是否正确 - 确保字段确实属于指定的表\n'''
                '''3. 表之间的关联键是否正确 - 检查JOIN条件中使用的字段是否合适\n'''
                '''4. 数据类型是否匹配 - 确保查询条件中的数据类型与字段类型一致\n'''
                '''5. 是否遗漏了重要的表或字段 - 根据用户问题补充可能有用的信息\n'''
                '''请基于已知的数据库结构信息，对其他agent的选择进行修正，确保最终使用的表和字段能够准确回答用户的问题。\n'''
                '''如果发现多个可能的修正方案，请选择最可能正确的一个，并简要说明理由。\n'''
            ),
            output_format = (
                '''输出模板示例:\n'''
                '''【选中的字段的清单】\n'''
                '''(把同一个表的字段聚合在这个表名[database_name.table_name]下面)\n'''
                '''(注意表名和字段名都是英文的)\n'''
                '''```json\n'''
                '''{"database_name.table_name": ["column_name", "column_name"],"database_name.table_name": ["column_name", "column_name"]}\n'''
                '''```\n'''
            ),
            llm=self.llm,
            enable_history=False,
            # stream=False,
        ))
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
            # stream=False,
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
                '''(可以用sql模拟一下，看流程是否合理)\n'''
                # '''(例如: \n'''
                # '''用户问: 2021年末，交通运输一级行业中有几个股票？\n'''
                # '''思维链：\n'''
                # '''用户问的是交通运输一级行业，可以用lc_exgindchange表的FirstIndustryName字段可以找到交通运输一级行业的行业代码FirstIndustryCode;\n'''
                # '''用户需要获取该行业有几个股票，在lc_indfinindicators表通过IndustryCode字段和Standard字段搜索到交通运输一级行业的信息,'''
                # '''其中lc_indfinindicators表的ListedSecuNum字段就是上市证券数量，\n'''
                # '''由于用户问的是2021年末，所以我需要用lc_indfinindicators表的InfoPublDate字段排序获得2021年末最后一组数据)\n'''

                '''【选中的字段的清单】\n'''
                '''(上述提到的字段都选上)\n'''
                '''(把同一个表的字段聚合在这个表名[database_name.table_name]下面)\n'''
                '''(注意表名和字段名都是英文的)\n'''
                '''```json\n'''
                '''{"database_name.table_name": ["column_name", "column_name"],"database_name.table_name": ["column_name", "column_name"]}\n'''
                '''```\n'''
            ),
            llm = self.llm,
            enable_history=False,
            # temperature=0.8,
            # stream=False,
        ))
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
            llm = self.llm,
            knowledge=self.get_db_info(),
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
            llm = self.llm,
            enable_history=False,
            # stream=False,
        ))
        self.agent_column_selector_old = Agent(AgentConfig(
            name = self.name+".columns_selector_old",
            role = (
                '''你是一个数据分析专家，从已知的数据表字段中，根据用户的问题，找出所有相关的字段名。'''
                '''请不要有遗漏!'''
            ),
            output_format = (
                '''输出模板示例:\n'''
                '''【分析】\n'''
                '''分析用户的提问\n'''
                # '''【当前的表之间相互关联的字段】\n'''
                # '''（考虑表之间的关联，把关联的字段选出来）\n'''
                # '''表A和表B之间: ...\n'''
                # '''表A和表C之间: ...\n'''
                '''【信息所在字段】\n'''
                '''（选出跟用户提问相关的信息字段，没有遗漏）\n'''
                '''- database_name.table_name.column_name: 这个字段可能包含xx信息，对应用户提问中的xxx\n'''
                '''【筛选条件所在字段】\n'''
                '''（选出跟用户提问相关的条件字段，没有遗漏）\n'''
                # '''（跟条件字段有外键关联的字段冗余选上，因为联表查询要用到）\n'''
                '''- database_name.table_name.column_name: 这个字段可能包含xx信息，对应用户提问中的xxx\n'''
                '''【选中的字段的清单】\n'''
                '''（把同一个表的字段聚合在这个表名[database_name.table_name]下面）\n'''
                '''(注意表名和字段名都是英文的)\n'''
                '''```json\n'''
                '''{"database_name.table_name": ["column_name", "column_name"],"database_name.table_name": ["column_name", "column_name"]}\n'''
                '''```\n'''
            ),
            llm = self.llm,
            enable_history=False,
            # stream=False,
        ))
        self.update_agent_lists()

    def update_agent_lists(self):
        """Updates the list of agents used in the workflow."""
        self.agent_lists = [
            self.agent_decode_question,
            self.agent_column_selector,
            self.agent_db_selector,
            self.agent_table_selector,
            self.agent_column_selector_old,
        ]

    def clone(self) -> 'CheckDbStructure':
        clone = CheckDbStructure(
            table_snippet=self.table_snippet,
            llm=self.llm,
            name=self.name,
            get_relevant_table_columns=self.get_relevant_table_columns,
            filter_table_columns=self.filter_table_columns,
            get_db_info=self.get_db_info,
            get_table_list=self.get_table_list,
            get_column_list=self.get_column_list,
            validate_column_filter=self.validate_column_filter,
            use_concurrency=self.use_concurrency,
            print_table_column=self.print_table_column,
            enable_llm_search=self.enable_llm_search,
            enable_vector_search=self.enable_vector_search,
        )
        clone.agent_decode_question = self.agent_decode_question.clone()
        clone.agent_column_selector = self.agent_column_selector.clone()
        clone.agent_db_selector = self.agent_db_selector.clone()
        clone.agent_table_selector = self.agent_table_selector.clone()
        clone.agent_column_selector_old = self.agent_column_selector_old.clone()
        clone.update_agent_lists()
        return clone

    def filter_column_list(self, column_filter: dict) -> str:
        """
        column_filter: dict{"table_name":["col1", "col2"]}
        """
        filtered_table_columns, table_relations = self.filter_table_columns(
            column_filter=column_filter,
        )
        result = (
            f"已取得可用的{COLUMN_LIST_MARK}:\n" +
            "\n---\n".join([self.print_table_column(table_column) for table_column in filtered_table_columns])
        )
        if len(table_relations) > 0:
            result += (
                "\n---\n" +
                "表之间的外链关系如下:\n" +
                "\n".join(table_relations) +
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

        first_user_msg = messages[-1]["content"]

        # 定义两个搜索方法为独立函数，以便并发执行
        def llm_search():
            """全llm逐层搜索"""
            local_usage_tokens = 0
            column_filter_result = {}
            if not self.enable_llm_search:
                return column_filter_result, local_usage_tokens
            # 选择数据库
            args_json = None
            table_list = ""
            error_msg = "\n请选择db，确保JSON格式正确。"
            for _ in range(5):
                try:
                    answer, tk_cnt = self.agent_db_selector.answer(f"用户问题:\n<{first_user_msg}>\n{error_msg}")
                    local_usage_tokens += tk_cnt
                    args_json = extract_last_json(answer)
                    if args_json is not None:
                        dbs = json.loads(args_json)
                        table_list = self.get_table_list(dbs=dbs)
                        break
                except Exception as e:
                    error_msg = f"\n注意: {str(e)}。请选择db，确保JSON格式正确。"
                    print(f"\n用户问题:\n<{first_user_msg}>\nWorkflow【{self.name}】agent_db_selector 遇到问题: {str(e)}, 现在重试...\n")
                    logger.debug("\n用户问题:\n<%s>\nWorkflow【%s】agent_db_selector 遇到问题: %s, 现在重试...\n", first_user_msg, self.name, str(e))
            if table_list != "":
                # 选择数据表
                column_list = ""
                error_msg = "\n请选择table，确保JSON格式正确。"
                for _ in range(5):
                    try:
                        answer, tk_cnt = self.agent_table_selector.answer(f"{table_list}\n用户问题:\n<{first_user_msg}>\n{error_msg}")
                        local_usage_tokens += tk_cnt
                        args_json = extract_last_json(answer)
                        if args_json is not None:
                            tables = json.loads(args_json)
                            column_list = self.get_column_list(tables=tables)
                            break
                    except Exception as e:
                        error_msg = f"\n注意: {str(e)}。请选择table，确保JSON格式正确。"
                        print(f"\n用户问题:\n<{first_user_msg}>\nWorkflow【{self.name}】agent_table_selector 遇到问题: {str(e)}, 现在重试...\n")
                        logger.debug("\n用户问题:\n<%s>\nWorkflow【%s】agent_table_selector 遇到问题: %s, 现在重试...\n", first_user_msg, self.name, str(e))
                if column_list != "":
                    # 筛选字段
                    error_msgs = []
                    org_answer = ""
                    for _ in range(5):
                        try:
                            if len(error_msgs) == 0 or len(column_filter_result) == 0:
                                answer, tk_cnt = self.agent_column_selector_old.answer((
                                    f"{column_list}\n用户问题:\n<{first_user_msg}>" +
                                    ("\n请注意:\n" + "\n".join(error_msgs) if len(error_msgs) > 0 else "") +
                                    "\n请从已知的表字段信息中选择column，确保正确地表字段关系，确保JSON格式正确。"
                                ))
                                org_answer = answer
                            else:
                                answer, tk_cnt = self.agent_fix_column_selection.answer((
                                    f"{column_list}\n用户问题:\n<{first_user_msg}>\n" +
                                    f"原agent的输出:\n'''\n{org_answer}\n'''\n" +
                                    ("\n请注意:\n" + "\n".join(error_msgs) if len(error_msgs) > 0 else "") +
                                    "\n请修正，确保正确的表字段关系，确保JSON格式正确。"
                                ))
                            local_usage_tokens += tk_cnt
                            args_json = extract_last_json(answer)
                            if args_json is not None:
                                tmp_column_filter = json.loads(args_json)
                                column_filter_result = tmp_column_filter
                                error_msg = self.validate_column_filter(column_filter_result)
                                if error_msg != "":
                                    raise Exception(error_msg)
                                break
                        except Exception as e:
                            error_msgs.append(str(e))
                            print(f"\n用户问题:\n<{first_user_msg}>\nWorkflow【{self.name}】agent_column_selector_old 遇到问题: {str(e)}, 现在重试...\n")
                            logger.debug("\n用户问题:\n<%s>\nWorkflow【%s】agent_column_selector_old 遇到问题: %s, 现在重试...\n", first_user_msg, self.name, str(e))
                else:
                    logger.debug("\n Fail to get column_list, Skip\n")
            else:
                logger.debug("\n Fail to get table_list, Skip\n")
            
            return column_filter_result, local_usage_tokens
            
        def vector_search():
            """向量+词频搜索"""
            local_usage_tokens = 0
            column_filter = {}
            if not self.enable_vector_search:
                return column_filter, local_usage_tokens

            # 解码问题
            answer, tk_cnt = self.agent_decode_question.answer("提问:\n" + messages[-1]["content"])
            question_list = [q.strip() for q in answer.split("\n") if q.strip() != ""]
            local_usage_tokens += tk_cnt
            
            # 搜索数据字段
            table_columns = self.get_relevant_table_columns(question_list)
            table_columns_str = (
                f"已取得可用的{COLUMN_LIST_MARK}:\n" +
                "\n---\n".join([self.print_table_column(table_column) for table_column in table_columns]) +
                "\n"
            )
            
            # 筛选字段
            error_msgs = []
            org_answer = ""
            for _ in range(5):
                try:
                    if len(error_msgs) == 0 or len(column_filter) == 0:
                        answer, tk_cnt = self.agent_column_selector.answer((
                            table_columns_str +
                            f"\n用户问题:\n<{first_user_msg}>" +
                        ("\n请注意:\n" + "\n".join(error_msgs) if len(error_msgs) > 0 else "") +
                            "\n请从已知的表字段信息中选择column，确保正确地表字段关系，确保JSON格式正确。"
                        ))
                        org_answer = answer
                    else:
                        answer, tk_cnt = self.agent_fix_column_selection.answer((
                            table_columns_str +
                            f"\n用户问题:\n<{first_user_msg}>\n" +
                            f"原agent的输出:\n'''\n{org_answer}\n'''\n" +
                            ("\n请注意:\n" + "\n".join(error_msgs) if len(error_msgs) > 0 else "") +
                            "\n请修正，确保正确的表字段关系，确保JSON格式正确。"
                        ))
                    local_usage_tokens += tk_cnt
                    args_json = extract_last_json(answer)
                    if args_json is not None:
                        tmp_column_filter = json.loads(args_json)
                        column_filter = tmp_column_filter
                        error_msg = self.validate_column_filter(column_filter)
                        if error_msg != "":
                            raise Exception(error_msg)
                        break
                except Exception as e:
                    error_msgs.append(str(e))
                    print(f"\n用户问题:\n<{first_user_msg}>\nWorkflow【{self.name}】agent_column_selector 遇到问题: {str(e)}, 现在重试...\n")
                    logger.debug("\n用户问题:\n<%s>\nWorkflow【%s】agent_column_selector 遇到问题: %s, 现在重试...\n", first_user_msg, self.name, str(e))
            
            return column_filter, local_usage_tokens

        # 根据开关决定是否使用并发
        if self.use_concurrency:
            # 使用线程池并发执行两个搜索方法
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                llm_future = executor.submit(llm_search)
                vector_future = executor.submit(vector_search)
                
                # 等待两个任务完成并获取结果
                column_filter_1, tokens_1 = llm_future.result()
                column_filter_2, tokens_2 = vector_future.result()
        else:
            # 顺序执行搜索方法
            column_filter_1, tokens_1 = llm_search()
            column_filter_2, tokens_2 = vector_search()
        
        # 累加tokens
        usage_tokens = tokens_1 + tokens_2
        
        # 合并两种搜索的结果
        # 打印两个搜索方法获取的column_filter结果
        logger.debug("LLM搜索获取的column_filter: %s\n", json.dumps(column_filter_1, ensure_ascii=False))
        logger.debug("向量搜索获取的column_filter: %s\n", json.dumps(column_filter_2, ensure_ascii=False))
        for key, values in column_filter_1.items():
            if key in column_filter_2:
                # 合并列表并去重
                column_filter_2[key] = list(set(column_filter_2[key] + values))
            else:
                column_filter_2[key] = values
                
        # 使用合并后的结果生成最终的表字段列表
        filtered_table_columns = self.filter_column_list(column_filter=column_filter_2)

        self.usage_tokens += usage_tokens
        return {
            "content": filtered_table_columns,
            "usage_tokens": usage_tokens,
        }
