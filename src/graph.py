"""
数据库表关系图模块，提供表示和分析数据库表之间关系的功能。
支持添加表关系、查找表之间路径以及导出为可视化格式。
"""

import json

class TableGraph:
    """表关系图类，用于表示和分析数据库表之间的关系结构"""

    def __init__(self):
        """初始化表关系图"""
        self.graph = {}  # 存储表之间的关系

    def add_table(self, table_name):
        """添加表到图中"""
        if table_name not in self.graph:
            self.graph[table_name] = {}  # 使用字典存储关系，提高查找效率

    def add_relation(self, table1, table2, relation_name=None, col1=None, col2=None, infer_transitive=False):
        """
        添加表之间的关系（无向）
        - table1, table2: 相关联的两个表
        - relation_name: 关系名称
        - col1: table1的关联列
        - col2: table2的关联列
        - infer_transitive: 是否自动推断传递关系
        """
        # 检查这个关系是否已经存在
        relation_exists = False
        
        self.add_table(table1)
        self.add_table(table2)

        # 创建关系对象
        relation = {
            "name": relation_name,
            "columns": f"{col1}-{col2}" if col1 and col2 else None
        }
        
        # 无向图 - 双向添加相同的关系
        if table2 not in self.graph[table1]:
            self.graph[table1][table2] = []
        else:
            # 检查是否已存在相同关系
            for existing_relation in self.graph[table1][table2]:
                if existing_relation["columns"] == relation["columns"]:
                    relation_exists = True
                    break
        
        if not relation_exists:
            self.graph[table1][table2].append(relation)

            if table1 not in self.graph[table2]:
                self.graph[table2][table1] = []
                
            # 创建反向关系
            reverse_relation = {
                "name": relation_name,
                "columns": f"{col2}-{col1}" if col2 and col1 else None
            }
            self.graph[table2][table1].append(reverse_relation)
            
            # 如果启用了传递关系推断，并且包含列信息
            if infer_transitive and col1 and col2:
                self._infer_from_new_relation(table1, table2, relation_name, col1, col2)
                
        return not relation_exists  # 返回是否添加了新关系

    def _infer_from_new_relation(self, table_a, table_b, relation_name, col_a, col_b):
        """
        基于新添加的关系推断可能的传递关系
        """
        # 情况1：A-B新关系，寻找B-C关系，推断A-C
        for table_c, relations_bc in self.graph[table_b].items():
            if table_c != table_a:  # 避免回到A
                for relation_bc in relations_bc:
                    if relation_bc["columns"]:
                        col_b2, col_c = relation_bc["columns"].split("-")
                        if col_b == col_b2:  # B表的同一列连接A和C
                            inferred_name = f"推断({relation_name}-{relation_bc['name']})"
                            self.add_relation(table_a, table_c, inferred_name, col_a, col_c, False)
        
        # 情况2：A-B新关系，寻找C-A关系，推断C-B
        for table_c, relations_ca in self.graph[table_a].items():
            if table_c != table_b:  # 避免回到B
                for relation_ca in relations_ca:
                    if relation_ca["columns"]:
                        col_c, col_a2 = relation_ca["columns"].split("-")
                        if col_a == col_a2:  # A表的同一列连接B和C
                            inferred_name = f"推断({relation_ca['name']}-{relation_name})"
                            self.add_relation(table_c, table_b, inferred_name, col_c, col_b, False)

    def get_neighbors(self, table_name):
        """获取与指定表直接相连的所有表及关系信息"""
        if table_name not in self.graph:
            return []

        neighbors = []
        for target_table, relations in self.graph[table_name].items():
            for relation in relations:
                neighbors.append((target_table, relation))
        return neighbors

    def find_shortest_path(self, start_table, end_table):
        """
        使用BFS查找从start_table到end_table的最短路径
        返回路径列表，包含表名和关系信息
        处理环路问题
        """
        if start_table not in self.graph or end_table not in self.graph:
            return None

        # BFS队列
        queue = [(start_table, [])]  # (当前表, 路径)
        visited = {start_table}  # 使用集合跟踪已访问节点，避免环路
        shortest_path = None
        shortest_length = float('inf')

        while queue:
            current, pth = queue.pop(0)

            # 找到目标表
            if current == end_table:
                # 检查是否是更短路径
                if len(pth) < shortest_length:
                    shortest_path = pth
                    shortest_length = len(pth)
                continue  # 继续搜索其他可能的路径

            # 如果当前路径已经超过已知的最短路径，则跳过
            if shortest_path and len(pth) >= shortest_length:
                continue

            # 探索相邻表
            for neighbor, relations in self.graph[current].items():
                if neighbor not in visited:  # 避免环路
                    visited.add(neighbor)

                    # 使用第一个关系信息
                    relation = relations[0]
                    relation_info = (
                        relation["name"] if relation["name"] else "关联",
                        relation["columns"] if relation["columns"] else ""
                    )

                    new_path = pth + [(current, neighbor, relation_info)]
                    queue.append((neighbor, new_path))

        # 返回最短路径
        return shortest_path

    def print_path(self, path_to_print):
        """以SQL JOIN格式美化输出路径信息"""
        if not path_to_print:
            return "没有找到路径"
        
        # 初始化SQL语句组件
        tables = []
        joins = []
        
        # 获取第一个表名作为起点
        start_table = path_to_print[0][0]
        tables.append(f"FROM {start_table} a")
        
        # 使用字母表示表别名 (a, b, c, ...)
        table_aliases = {start_table: 'a'}
        current_alias_index = 1
        
        # 处理路径中的每一步
        for src, dst, relation in path_to_print:
            # 为目标表分配别名
            if dst not in table_aliases:
                alias = chr(ord('a') + current_alias_index)
                table_aliases[dst] = alias
                current_alias_index += 1
            
            # 从关系信息中提取连接条件
            src_alias = table_aliases[src]
            dst_alias = table_aliases[dst]
            
            # 处理关系列信息（单行格式）
            if relation[1]:  # 如果有列信息
                src_col, dst_col = relation[1].split("-")
                join_clause = f"JOIN {dst} {dst_alias} ON {src_alias}.{src_col} = {dst_alias}.{dst_col}"
            else:
                # 无列信息时使用默认关联说明（单行格式）
                join_clause = f"JOIN {dst} {dst_alias} ON {src_alias}.id = {dst_alias}.{src.replace('表', '')}_id  -- 关系: {relation[0]}"
            
            joins.append(join_clause)
        
        # 组合SQL查询
        sql_query = [
            f"-- 从{start_table}到{path_to_print[-1][1]}的完整连接路径示例:\n",
            "SELECT *"
        ]
        sql_query.extend(tables)
        sql_query.extend(joins)
        sql_query.append(";")
        
        return " ".join(sql_query)

    def all_paths(self, max_length=None):
        """计算所有表之间的最短路径（可选限制最大长度）"""
        all_tables = list(self.graph.keys())
        paths = {}

        for start in all_tables:
            for end in all_tables:
                if start != end:
                    shortest_path = self.find_shortest_path(start, end)
                    if shortest_path and (max_length is None or len(shortest_path) <= max_length):
                        paths[(start, end)] = shortest_path

        return paths

    def export_dot(self, filename="table_relations.dot"):
        """导出为DOT格式，可用Graphviz可视化"""
        with open(filename, "w", encoding="utf-8") as f:
            f.write("graph TableRelations {\n")  # 注意这里用无向图graph而不是digraph
            f.write("  rankdir=LR;\n")  # 从左到右布局
            f.write("  node [shape=box, style=filled, fillcolor=lightblue];\n")

            # 添加边 - 无向图只需添加一次
            added_edges = set()  # 跟踪已添加的边，避免重复

            for src_table, neighbors in self.graph.items():
                for dst_table, relations in neighbors.items():
                    # 创建一个唯一标识这条边的键（按字母顺序排序表名）
                    edge_key = tuple(sorted([src_table, dst_table]))

                    if edge_key not in added_edges:
                        for relation in relations:
                            label = relation["name"] if relation["name"] else "关联"
                            if relation["columns"]:
                                label += f"\\n{relation['columns']}"
                            f.write(f'  "{src_table}" -- "{dst_table}" [label="{label}"];\n')
                        added_edges.add(edge_key)

            f.write("}\n")
        print(f"图已导出到 {filename}")

    def find_all_paths(self, start_table, end_table, max_length=3):
        """
        使用DFS查找从start_table到end_table的所有路径
        - start_table: 起始表
        - end_table: 目标表
        - max_length: 最大路径长度限制，默认为3防止过度搜索
        返回所有可能路径的列表
        """
        if start_table not in self.graph or end_table not in self.graph:
            return []

        all_paths = []
        visited = {start_table}  # 使用集合跟踪当前路径中已访问节点

        def dfs(current, pth, depth=0):
            # 达到最大长度限制，停止继续搜索
            if max_length is not None and depth >= max_length:
                return
            
            # 找到目标表
            if current == end_table:
                all_paths.append(pth[:])  # 添加当前路径的副本
                return

            # 探索相邻表
            for neighbor, relations in self.graph[current].items():
                if neighbor not in visited:  # 避免环路
                    visited.add(neighbor)

                    # 考虑所有关系
                    for relation in relations:
                        relation_info = (
                            relation["name"] if relation["name"] else "关联",
                            relation["columns"] if relation["columns"] else ""
                        )

                        # 递归搜索
                        dfs(neighbor, pth + [(current, neighbor, relation_info)], depth + 1)

                    # 回溯
                    visited.remove(neighbor)

        # 开始DFS搜索
        dfs(start_table, [], 0)
        return all_paths

    def print_all_paths(self, paths):
        """以SQL JOIN格式美化输出多条路径信息"""
        if not paths:
            return "没有找到路径"
        
        result = []
        for i, pth in enumerate(paths, 1):
            path_sql = self.print_path(pth)
            result.append(f"-- 路径 {i}:\n{path_sql}")
        
        return "\n\n".join(result)

    def save_to_file(self, filename="table_graph.json"):
        """
        将表关系图保存到JSON文件
        - filename: 保存的文件名
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.graph, f, ensure_ascii=False, indent=2)

        print(f"表关系图已保存到 {filename}")

    @classmethod
    def load_from_file(cls, filename="table_graph.json"):
        """
        从JSON文件加载表关系图
        - filename: 要加载的文件名
        返回加载的TableGraph对象
        """
        graph = cls()
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                graph.graph = json.load(f)
            print(f"TableGraph: 已从 {filename} 加载表关系图")
        except FileNotFoundError:
            print(f"TableGraph: 文件 {filename} 不存在")
        except json.JSONDecodeError:
            print(f"TableGraph: 文件 {filename} 格式错误，无法解析")

        return graph

if __name__ == "__main__":
    # 创建表关系图
    db_graph = TableGraph()

    # 添加表之间的关系
    db_graph.add_relation("用户表", "订单表", "用户-订单", "id", "user_id")
    db_graph.add_relation("订单表", "订单详情表", "订单-详情", "id", "order_id")
    db_graph.add_relation("产品表", "订单详情表", "产品-详情", "id", "product_id")
    db_graph.add_relation("产品表", "库存表", "产品-库存", "id", "product_id")
    db_graph.add_relation("供应商表", "产品表", "供应商-产品", "id", "supplier_id")
    db_graph.add_relation("供应商表", "用户表", "产品-产品", "user_id", "id")

    # 保存表关系图
    db_graph.save_to_file()

    # 加载表关系图示例
    loaded_graph = TableGraph.load_from_file()

    # 导出为DOT格式进行可视化
    db_graph.export_dot()

    # 查找从用户表到库存表的最短路径
    path = db_graph.find_shortest_path("用户表", "库存表")
    print(db_graph.print_path(path))
