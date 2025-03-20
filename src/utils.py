"""
This module provides utility functions.
"""

import re
import json
from typing import Optional
import sqlparse
import sqlglot

COLUMN_LIST_MARK = "数据表的字段信息如下"

def generate_markdown_table(data_list, key_title_map):
    """
    根据输入的数据列表和键标题映射生成 Markdown 表格。

    :param data_list: 包含字典的列表，每个字典代表一行数据。
    :param key_title_map: 字典，键为数据字典中的键，值为表格标题。
    :return: 生成的 Markdown 表格字符串。
    """
    # 创建表头
    headers = "| " + " | ".join(key_title_map.values()) + " |\n"
    separators = "| " + " | ".join("---" for _ in key_title_map) + " |\n"
    markdown_table = headers + separators

    # 填充表格内容
    for item in data_list:
        row = "| " + " | ".join(item[key].replace('\n', '\\n') for key in key_title_map) + " |\n"
        markdown_table += row

    return markdown_table 

def get_column_list(db_table, table_column, tables: list[str]) -> str:
    """
    tables: list of table names, format is database_name.table_name
    """
    # key_title_map = {'column': '字段名', 'desc': '注释'}
    # column_lists = []
    # for table in tables:
    #     if '.' not in table or table.count('.') != 1:
    #         raise ValueError(f"发生异常: 表名`{table}`格式不正确，应该为`database_name.table_name`")
    #     db_name, table_name = table.split('.')
    #     if db_name not in db_table:
    #         raise KeyError(f"发生异常: 数据库名`{db_name}`不存在")
    #     if any(t['表英文'] == table_name for t in db_table[db_name]['表']):
    #         column_lists.append(f"数据表:\n{db_name}.{table_name}\n"+
    #             generate_markdown_table(table_column[table_name], key_title_map))
    # result = (
    #     f"已取得可用的{COLUMN_LIST_MARK}:\n" +
    #     "\n---\n".join(column_lists)
    # )

    column_lists = []
    for table in tables:
        if '.' not in table or table.count('.') != 1:
            raise ValueError(f"发生异常: 表名`{table}`格式不正确，应该为`database_name.table_name`")
        db_name, table_name = table.split('.')
        if db_name not in db_table:
            raise KeyError(f"发生异常: 数据库名`{db_name}`不存在")
        if any(t['表英文'] == table_name for t in db_table[db_name]['表']):
            column_lists.append({
                "表名": table,
                "表字段": table_column[table_name],
            })
    result = (
        f"已取得可用的{COLUMN_LIST_MARK}:\n" +
        json.dumps(column_lists, ensure_ascii=False)+
        "\n"
    )
    return result

def extract_last_sql(query_string: str, block_mark: str) -> Optional[str]:
    """
    从给定的字符串中提取最后一组 SQL 语句，并去掉注释。

    :param query_string: 包含 SQL 语句的字符串。
    :param block_mark: SQL 代码块的标记。
    :return: 最后一组 SQL 语句。
    """
    # 使用正则表达式匹配 SQL 语句块
    sql_pattern = re.compile(rf"(?s)```{re.escape(block_mark)}\s+(.*?)\s+```")
    matches = sql_pattern.findall(query_string)
    if matches:
        # 提取最后一个 SQL 代码块
        last_sql_block = matches[-1].strip()
        # 去掉注释但保留分号
        last_sql_block = re.sub(r"--.*(?=\n)|--.*$", "", last_sql_block)
        # 分割 SQL 语句
        sql_statements = [stmt.strip() for stmt in last_sql_block.split(';') if stmt.strip()]
        # 返回最后一个非空 SQL 语句
        sql = sql_statements[-1] + ';' if sql_statements else None
        if sql:
            try:
                sql = sqlparse.format(sql, reindent=False, keyword_case='upper', strip_comments=True)
                sql = re.sub(r'\s+', ' ', sql).strip()
            except Exception:
                return None
        return sql
    return None

def count_total_sql(query_string: str, block_mark: str) -> int:
    """
    从给定的字符串中提取所有 SQL 语句的总数。

    :param query_string: 包含 SQL 语句的字符串。
    :param block_mark: SQL 代码块的标记。
    :return: SQL 语句的总数。
    """
    # 使用正则表达式匹配 SQL 语句块
    sql_pattern = re.compile(rf"(?s)```{re.escape(block_mark)}\s+(.*?)\s+```")
    matches = sql_pattern.findall(query_string)
    total_sql_count = 0
    for sql_block in matches:
        # 去掉注释但保留分号
        sql_block = re.sub(r"--.*(?=\n)|--.*$", "", sql_block)
        # 分割 SQL 语句并计数
        sql_statements = [stmt.strip() for stmt in sql_block.split(';') if stmt.strip()]
        total_sql_count += len(sql_statements)
    return total_sql_count

def extract_last_json(text: str) -> Optional[str]:
    """
    从给定文本中提取最后一个```json和```之间的内容。

    Args:
        text (str): 包含JSON内容的文本。

    Returns:
        Optional[str]: 提取的JSON字符串，如果未找到则返回None。
    """
    pattern = r'```json(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1].strip() if matches else None

def show(obj):
    """
    打印对象的 JSON 表示。
    """
    if isinstance(obj, dict):
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    elif isinstance(obj, list):
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    elif isinstance(obj, str):
        if str(obj).startswith(('{', '[')):
            try:
                o = json.loads(str)
                print(json.dumps(o, ensure_ascii=False, indent=2))
            except Exception:
                print(obj)
        else:
            print(obj)
    elif isinstance(obj, (int, float)):
        print(obj)
    else:
        print(obj)

def extract_all_sqls(query_string: str, block_mark: str) -> list[str]:
    """
    从给定的字符串中提取所有 SQL 语句，并去掉注释。

    :param query_string: 包含 SQL 语句的字符串。
    :param block_mark: SQL 代码块的标记。
    :return: 所有处理后的 SQL 语句列表。
    """
    # 使用正则表达式匹配 SQL 语句块
    sql_pattern = re.compile(rf"(?s)```{re.escape(block_mark)}\s+(.*?)\s+```")
    matches = sql_pattern.findall(query_string)
    
    result_sqls = []
    for sql_block in matches:
        # 去掉注释但保留分号
        sql_block = re.sub(r"--.*(?=\n)|--.*$", "", sql_block)
        # 分割 SQL 语句
        sql_statements = [stmt.strip() for stmt in sql_block.split(';') if stmt.strip()]
        
        for sql in sql_statements:
            sql = sql + ';'
            try:
                formatted_sql = sqlparse.format(sql, reindent=False, keyword_case='upper', strip_comments=True)
                formatted_sql = re.sub(r'\s+', ' ', formatted_sql).strip()
                # formatted_sql = sqlparse.format(sql, reindent=False, keyword_case='upper', strip_comments=True)
                result_sqls.append(formatted_sql)
            except Exception:
                continue
                
    return result_sqls

def extract_tables_and_columns(sql_query):
    """
    从SQL查询中提取表名和列名，并将列映射到它们所属的表。
    
    :param sql_query: SQL查询字符串
    :return: 包含表、列和它们关系的字典
    """
    try:
        # 解析SQL
        parsed = sqlglot.parse_one(sql_query)
        
        # 提取表名和别名映射
        tables = []
        table_aliases = {}

        for table in parsed.find_all(sqlglot.exp.Table):
            qualified_name = f"{table.db}.{table.name}" if table.db else table.name
            tables.append(qualified_name)
            # 如果表有别名，记录下来
            if hasattr(table, 'alias') and table.alias:
                table_aliases[table.alias] = qualified_name
        
        # 提取字段名并映射到表
        all_columns = []
        table_to_columns = {table: [] for table in tables}
        unassigned_columns = []
        
        for column in parsed.find_all(sqlglot.exp.Column):
            col_name = column.name
            all_columns.append(col_name)
            
            # 尝试确定列所属的表
            table_ref = None
            if hasattr(column, 'table') and column.table:
                # 列有明确的表引用
                table_name = column.table
                # 检查是否是别名
                if table_name in table_aliases:
                    table_ref = table_aliases[table_name]
                else:
                    # 查找匹配的表名
                    for t in tables:
                        if t.endswith('.' + table_name) or t == table_name:
                            table_ref = t
                            break
            
            if table_ref and table_ref in table_to_columns:
                table_to_columns[table_ref].append(col_name)
            else:
                unassigned_columns.append(col_name)
        
        return {
            'table_to_columns': table_to_columns,
            'unassigned_columns': unassigned_columns
        }
    except Exception as e:
        # 解析失败时返回简单结构
        return {
            'table_to_columns': {},
            'unassigned_columns': [],
            'error': str(e)
        }
