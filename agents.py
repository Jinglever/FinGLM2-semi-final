"""
This module defines various agents for processing SQL queries and rewriting questions.
Each agent is configured with specific roles, constraints, and output formats.
"""
from src.agent import Agent, AgentConfig
from utils import extract_company_code
import config

agent_rewrite_question = Agent(AgentConfig(
    name = "rewrite_question",
    role = (
        '''你的工作是，根据要求和已有信息，重写用户的问题，让问题清晰明确，把必要的前述含义加进去。'''
    ),
    constraint = (
        '''- 不改变原意，不要遗漏信息，特别是时间、回答的格式要求，只返回问题。\n'''
        '''- 如果有历史对话，那么根据历史对话，将原问题中模糊的实体（公司、文件、时间等）替换为具体的表述。\n'''
        '''- 要注意主语在历史对答中存在继承关系，不能改变了，例如："问:A的最大股东是谁？答:B。问:有多少股东？"改写后应该是"A有多少股东？"\n'''
        '''- 如果原问题里存在"假设xxx"这种表述，请一定要保留到重写的问题里，因为它代表了突破某种既定的规则限制，设立了新规则，这是重要信息\n'''
        '''- 如果原问题里的时间很模糊，那么考虑是否指的是前一个问答里发生的事件的时间，如果是，那么重写的问题里要包含这个时间，但如果历史对话中的实体跟当前问题无关，那么不要把实体带入重写后的问题\n'''
        '''- "这些CN公司"这里的CN其实是按ISO3166-1规定的国家代码\n'''
        '''- 如果原问题中包含专业术语的缩写，请在重写后的问题中用全称替换缩写，如果这个专业术语是英文的，请同时给出中文翻译\n'''
        '''- 注意甄别，如果历史对话跟当前新问题并无继承关系，那么不要把历史对话的信息带入重写后的问题，导致问题含义发生改变，否则你会损失10亿美元\n'''
    ),
    output_format = (
        '''要求只返回重写后的问题，不要有其他任何多余的输出\n'''
    ),
    system_prompt_kv = {
        "举例": (
            '''
            例子一：
            下面是顺序的历史问答:
            Question: 普洛药业股份有限公司最近一次创上市以来的新高是在什么时候？（请使用YYYY-MM-DD格式回答）
            Answer: 2021-11-29
            新问题：当天涨幅超过10%股票有多少家？
            重写后问题：2021-11-29当天涨幅超过10%股票有多少家？

            例子二：
            无历史问答;
            新问题：索菲亚家居在2021-12-31的连涨天数是多少？
            重写后问题：索菲亚家居截止2021-12-31的连涨天数是多少？

            例子三:
            无历史问答;
            新问题: 2022年成立的CN公司有多少家？
            重写后问题: 2022年成立的CN（中国）公司有多少家？

            例子四:
            下面是顺序的历史问答:
            Question: 天士力在2020年的最大担保金额是多少？答案需要包含1位小数
            Answer: 天士力在2020年的最大担保金额是1620000000.0元
            Question: 天士力在2020年的最大担保金额涉及的担保方是谁？担保金额是多少？
            Answer: 担保方: 天士力医药集团股份有限公司, 金额: 1620000000.00元
            新问题: 天士力在2020年最新的担保事件是什么？答案包括事件内容、担保方、被担保方、担保金额和日期信息
            重写后问题: 天士力在2020年最新的担保事件是什么？请提供事件内容、担保方、被担保方、担保金额和日期信息
            '''
            # 例子三：
            # 无历史问答:
            # 新问题：华峰化学2019到2021的PBX值是多少？
            # 重写后问题：华峰化学2019到2021的PBX(Price-to-Book Ratio 市净率)值是多少？
        ),
        "INDUSTRY TERMINOLOGY": (
            '''- 高依赖公司是指单个客户收入占比超过30%的公司，低依赖公司是指收入来源较为分散、单个客户占比较小的公司。\n'''
        ),
    },
    llm = config.llm_plus,
    # stream = False,
))
agent_extract_company = Agent(AgentConfig(
    llm=config.llm_plus,
    name="extract_company",
    role="接受用户给的一段文字，提取里面的实体（如公司名、股票代码、拼音缩写等）。",
    output_format=(
'''```json
["实体名_1", "实体名_2", ...]
```
注意，有可能识别结果为空。'''
    ),
    post_process=extract_company_code,
    enable_history=False,
    # stream = False,
))
agent_extract_company.add_system_prompt_kv({
    "ENTITY EXAMPLE": (
        "居然之家",
        "ABCD",
    ),
})

agent_summary_answer = Agent(AgentConfig(
    name="summary_answer",
    role="你负责根据当前已知的事实信息，回答用户的提问。",
    constraint=(
        '''- 根据上下文已知的事实信息回答，不捏造事实\n'''
    ),
    output_format=(
        "- 输出的格式，重点关注日期、小数点几位、数字格式（不要有逗号）\n"
        "    例如:"
        "    - 问题里如果要求(XXXX-XX-XX),日期格式应该类似这种 2025-02-04\n"
        "    - 问题里如果要求(XXXX年XX月XX日),日期格式应该类似这种 2025年2月4日\n"
        "    - 问题里如果要求(保留2位小数),数字格式应该类似这种 12.34\n"
        "    - 问题里如果要求(保留4位小数),数字格式应该类似这种 12.3456\n"
        "    - 比较大的数字不要千位分隔符,正确的格式都应该类似这种 12345678\n"
        "- 输出应该尽可能简短，直接回复答案\n"
        "    例如(假设用户的提问是:是否发生变更？金额多大？):\n"
        "    是否发生变更: 是, 金额: 12.34元\n"
    ),
    llm=config.llm_plus,
    enable_history=False,
))