"""This module initializes Workflows."""
import config
from src.workflow_v3 import SqlQuery, CheckDbStructure
# from utils import execute_sql_query, db_select_post_process, table_select_post_process, foreign_key_hub
import utils

sql_query = SqlQuery(
    execute_sql_query = utils.execute_sql_query,
    llm = config.llm_plus,
    max_iterate_num = config.MAX_ITERATE_NUM,
    cache_history_facts = True,
    specific_column_desc=config.enum_columns,
    default_sql_limit=config.MAX_SQL_RESULT_ROWS,
)
sql_query.agent_master.add_system_prompt_kv({
    "EXTEND INSTRUCTION": (
        # '''- 如果Company和InnerCode都搜不到，那么要考虑股票代码\n'''
        '''- CompanyCode跟InnerCode不对应，不能写`CompanyCode`=`InnerCode`，可以通过constantdb.secumain、constantdb.hk_secumain或constantdb.us_secumain换取对方\n'''
        '''- 涉及股票价格时：\n'''
        '''    - 筛选是否新高，要选择`最高价`字段(HighPrice)，而非收盘价(ClosePrice)，比如月度新高要看月最高价(HighPriceRM)，年度新高要看年最高价(HighPriceRY)，周新高要看周最高价(HighPriceRW)\n'''
        # '''- ConceptCode是数字，不是字符串\n'''
        '''- 在lc_actualcontroller中只有1条记录也代表实控人发生了变更\n'''
        # '''- 如果用户的前一条提问里提及某实体，那么后续追问虽未明说，但也应该是跟该实体相关\n'''
        # '''- 注意观察同一个表中的类型字段，结合用户的问题，判断是否要进行类型筛选\n'''
        # '''- 如果用户提问是希望知道名字，那么要把名字查出来\n'''
        '''- 中国的城市的AreaInnerCode是constantdb.lc_areacode里ParentName为'中国'的，你不应该也并不能获取到所有中国的城市代码，所以你需要用联表查询\n'''
        # '''- 我们的数据库查询是有一个默认的LIMIT的，这是个重要的信息，当你的SQL没有明确LIMIT的时候，你要知道获取到的数据可能不是全部。\n'''
        '''- 如果用户提问涉及某个年度的“年度报告”，默认该报告是在次年发布。例如，“2019年年度报告”是在2020年发布的。\n'''
        '''- 季度报告通常在下一个季度发布，例如，第一季度的报告会在第二季度发布。\n'''
        # '''- 如果用户想知道子类概念的名称，你应该去获取astockindustrydb.lc_conceptlist的ConceptName和ConceptCode\n'''
        '''- A股公司的基本信息在astockbasicinfodb.lc_stockarchives, 港股的在hkstockdb.hk_stockarchives, 美股的在usstockdb.us_companyinfo，这三张表不能互相关联\n'''
        '''- A股公司的上市基本信息在constantdb.secumain, 港股的在constantdb.hk_secumain, 美股的在constantdb.us_secumain，这三张表不能互相关联\n'''
        # '''- 作为筛选条件的名称，请务必分清楚它是公司名、人名还是其他什么名称，避免用错字段\n'''
        '''- 但凡筛选条件涉及到字符串匹配的，都采取模糊匹配，增加匹配成功概率\n'''
        '''- 比例之间的加减乘除，要务必保证算子是统一单位的，比如3%其实是0.03，0.02其实是2%\n'''
        '''- 时间日期字段都需要先做`DATE()`或`YEAR()`格式化再参与SQL的筛选条件，否则就扣你20美元罚款\n'''
        # '''- 关于概念，可以同时把ConceptName、SubclassName、ClassName查询出来，你就对概念有全面的了解，要记住概念有三个级别，据此理解用户提及的概念分别属于哪个级别\n'''
        '''- IndustryCode跟CompanyCode不对应，不能写`IndustryCode`=`CompanyCode`\n'''
        # '''- 指数内部编码（IndexInnerCode）：与“证券主表（constantdb.secumain）”中的“证券内部编码（InnerCode）”关联\n'''
        # '''- 证券内部编码（SecuInnerCode）：关联不同主表，查询证券代码、证券简称等基本信息。当0<SecuInnerCode<=1000000时，与“证券主表（constantdb.secuMain）”中的“证券内部编码（InnerCode）”关联；当1000000<SecuInnerCode<=2000000时，与“港股证券主表（constantdb.hk_secumain）”中的“证券内部编码（InnerCode）”关联；当7000000<SecuInnerCode<=10000000时，与“ 美股证券主表（constantdb.us_secumain）”中的“证券内部编码（InnerCode）”关联；\n'''
        # '''- 指数内部代码（IndexCode）：与“证券主表（constaintdb.secuMain）”中的“证券内部编码（InnerCode）”关联\n'''
        # '''- 假设A表有InnerCode, B表有ConceptCode和InnerCode，我们需要找出B表里的所有InnerCode，然后用这些InnerCode从A表获取统计数据，那么可以用联表查询 SELECT a FROM A WHERE InnerCode in (SELECT InnerCode FROM B WHERE ConceptCode=b)\n'''
        '''- 一个公司可以同时属于多个概念板块，所以如果问及一个公司所属的概念板块，指的是它所属的所有概念板块\n'''
        '''- ConceptCode跟InnerCode不对应，不能写`ConceptCode`=`InnerCode`\n'''
        '''- 如果用户要求用简称，那你要保证获取到简称(带Abbr标识)，比如constantdb.secumain里中文名称缩写是ChiNameAbbr\n'''
        '''- 关于分红的大小比较, 如果派现金额(Dividendsum)没记录，那么可以通过税后实派比例(ActualRatioAfterTax)来比价大小\n'''
        '''- 不能使用的关键词`Rank`作为别名，比如`SELECT a as Rank;`\n'''
        '''- AreaInnerCode跟CompanyCode不对应，不能写`AreaInnerCode`=`CompanyCode`\n'''
        '''- 本系统不具备执行python代码的能力，请使用SQL查询来完成数值计算\n'''
        '''- 对于枚举值类型的字段，要谨慎选择，切莫理解错误\n'''
        # '''- lc_suppcustdetail.SerialNumber为999的时候代表前五大客户/前五大供应商的合计值，如果用户问的是合计值，那么需要用枚举值来筛选\n'''
        '''- EndDate是个重要字段，往往代表交易数据的时间\n'''
        # '''- lc_suppcustdetail.Ratio就是客户占总营收的比例，单位是百分比，不需要在跟OperatingRevenue相除\n'''
        '''- 如果用户问股票列表且没有指明回答的具体形式，那么都要回答股票代码和简称\n'''
        '''- 数据示例的值不能作为已知条件，只能作为参考， 不能直接用数据示例的值，否则会损失10亿美元!不能直接用数据示例的值，否则会损失10亿美元!不能直接用数据示例的值，否则会损失10亿美元!\n'''
        '''- "这些CN公司"这里的CN其实是按ISO3166-1规定的国家代码\n'''
        '''- 公司上市时间比成立时间晚是正常的！\n'''
        # '''- 对于信息发布类的数据表，如果对一个时间范围（多日）的数据进行SUM，你可能会得到同一个实体的多日数据之和，你确定是用户想要的结果吗？\n'''
    ),
    "INDUSTRY TERMINOLOGY": (
        # '''- 概念分支指的是是subclass\n'''
        # '''- "化工"是2级概念(SubclassName)\n'''
        # '''- 子类概念的字段是ConceptName和ConceptCode，被纳入到2级概念(SubclassName)或者1级概念(ClassName)下\n'''
        '''- 基金管理人指的是负责管理基金的公司，而基金经理则是具体负责基金投资运作的个人\n'''
        '''- 高依赖公司是指单个客户收入占比超过30%的公司，低依赖公司是指收入来源较为分散、单个客户占比较小的公司。\n'''
        '''- 但凡涉及到概念，如果用户没有明确是一级概念、二级概念还是概念板块，那么你要对ClassName、SubclassName、ConceptName都进行查询，确认它属于哪个级别\n'''
    ),
    "思考流程": (
'''
1. **问题解析**：
   - 明确用户的核心需求
   - 识别问题中的关键实体、时间范围、比较关系等要素

2. **结构映射**：
   - 确定涉及的主表及关联表
   - 验证字段是否存在及命名一致性（特别注意日期格式和单位）
   - 识别必要的JOIN操作

3. **条件处理**：
   - 提取显式过滤条件（如"2023年后注册的用户"）
   - 推导隐式条件（如"最近一年"需转换为具体日期范围）
   - 处理特殊值（如status字段的枚举值映射）

4. **结果处理**：
   - 判断是否需要聚合函数（SUM/COUNT/AVG）
   - 确定分组维度（GROUP BY字段）
   - 处理排序和限制（ORDER BY/LIMIT）

5. **验证检查**：
   - 检查JOIN条件是否完备
   - 验证别名使用一致性
   - 确保聚合查询正确使用GROUP BY
   - 防范SQL注入风险（如正确使用参数化）
   - 前后查询结果是否存在矛盾
'''
    )
})

check_db_structure = CheckDbStructure(
    table_snippet = config.table_snippet,
    llm = config.llm_plus,
    name = "check_db_structure",
    get_relevant_table_columns = utils.get_relevant_table_columns,
    filter_table_columns = utils.filter_table_columns,
    get_db_info = utils.get_db_info,
    get_table_list = utils.get_table_list,
    get_column_list = utils.get_column_list,
    validate_column_filter = utils.validate_column_filter,
    use_concurrency = False,
    print_table_column = utils.print_table_column,
    enable_llm_search = config.ENABLE_LLM_SEARCH_DB,
    enable_vector_search = config.ENABLE_VECTOR_SEARCH_DB,
)
check_db_structure.agent_column_selector.add_system_prompt_kv({
    "EXTEND INSTRUCTION": (
        '''- 涉及股票价格时：\n'''
        '''    - 筛选是否新高，要选择`最高价`字段(HighPrice)，而非收盘价(ClosePrice)，比如月度新高要看月最高价(HighPriceRM)，年度新高要看年最高价(HighPriceRY)，周新高要看周最高价(HighPriceRW)\n'''
        # '''- 年度报告的时间条件应该通过astockbasicinfodb.lc_balancesheetall表的InfoPublDate字段来确认\n'''
        '''- 作为筛选条件的名称，请务必分清楚它是公司名、人名还是其他什么名称，避免用错字段\n'''
        # '''- 关于概念，可以同时把ConceptName、SubclassName、ClassName查询出来，你就对概念有全面的了解，要记住概念有三个级别，据此理解用户提及的概念分别属于哪个级别\n'''
        '''- 如果用户要求用简称，那你要保证获取到简称(带Abbr标识)，比如constantdb.secumain里中文名称缩写是ChiNameAbbr\n'''
        '''- 对于分红金额，如果有多个候选字段都可能代表分红金额，那么请把它们都选上\n'''
        '''- 对于枚举值类型的字段，要谨慎选择，切莫理解错误\n'''
        '''- EndDate是个重要字段，往往代表交易数据的时间\n'''
        '''- 如果用户问股票列表且没有指明回答的具体形式，那么都要回答股票代码和简称\n'''
        '''- 数据示例的值不能作为已知条件，只能作为参考，如果你直接用，会损失10亿美元\n'''
        '''- "这些CN公司"这里的CN其实是按ISO3166-1规定的国家代码\n'''
        '''- A股公司的基本信息在astockbasicinfodb.lc_stockarchives, 港股的在hkstockdb.hk_stockarchives, 美股的在usstockdb.us_companyinfo，这三张表不能互相关联\n'''
        '''- A股公司的上市基本信息在constantdb.secumain, 港股的在constantdb.hk_secumain, 美股的在constantdb.us_secumain，这三张表不能互相关联\n'''
    ),
    "INDUSTRY TERMINOLOGY": (
        # '''- 概念分支指的是是subclass\n'''
        # '''- "化工"是2级概念(SubclassName)\n'''
        # '''- SubclassName不是子类概念,子类概念是指ConceptCode和ConceptName\n'''
        '''- 基金管理人指的是负责管理基金的公司，而基金经理则是具体负责基金投资运作的个人\n'''
        '''- constantdb.us_secumain.DelistingDate、constantdb.hk_secumain.DelistingDate是退市日期，涉及退市的应该考虑它们\n'''
        '''- 高依赖公司是指单个客户收入占比超过30%的公司，低依赖公司是指收入来源较为分散、单个客户占比较小的公司。\n'''
        '''- 但凡涉及到概念，如果用户没有明确是一级概念、二级概念还是概念板块，那么你要对ClassName、SubclassName、ConceptName都进行查询，确认它属于哪个级别\n'''
    ),
})
