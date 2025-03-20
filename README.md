# FinGLM2-semi-final
这个是由清华大学基础模型研究中心主办的《2024金融行业·大模型挑战赛》
https://competitions.zhipuai.cn/matchDetail?id=120241202000000003
本方案在复赛B榜最后一次提交拿到53.1分。
在DB召回上，有一些比较有趣的工程手段，后续会通过公众号文章整理本方案的思路和技巧。

### Python版本
```
Python 3.12.2
```

### 准备环境变量
```
cp .env-template .env
```
填入自己的key

### 准备题目
```
ls assets/金融复赛a榜.json
```

### 执行命令
```
PYTHONUNBUFFERED=1 python main.py | tee -a output/main.log
```
执行以上命令，就可以开始跑了。

---

## 其他说明

### 配置
在 config.py 文件里可以设定一些配置项。
```
MAX_ITERATE_NUM = 15  # 配置SQL Query工作流的最大迭代次数
MAX_SQL_RESULT_ROWS = 30 # 配置智谱SQL查询接口的LIMIT参数
MAX_CONCURRENT_THREADS = 10 # 配置并发线程数

START_INDEX = [0, 0]  # 起始下标 [team_index, question_idx]
END_INDEX = [0, 0] # 结束下标 [team_index, question_idx] (包含) 在main.py里会根据all_question的长度自动设置
SAVE_FILE_SUBFIX = "_f2ad09dc360c4249ac273521a378104f_J64xUpwA4TAc_v3.1.3" # 缓存解题过程的文件名标记

FLAG_IGNORE_CACHE = False # 是否忽略中间缓存的结果并重跑，如果为False，那么当解题过程被中断后，下次再运行时会从缓存中恢复继续跑剩下未解的题目
ENABLE_LLM_SEARCH_DB = True # 是否启用LLM搜索数据库
ENABLE_VECTOR_SEARCH_DB = True # 是否启用向量搜索数据库
ENABLE_BATCH_MODE = False # 是否启用按问题组批量模式，由于主办方要求逐个问题解，所以这里需要设置为False

llm_plus = llms.llm_glm_4_plus # 配置使用的LLM
embed = embeddings.glm_embedding_3 # 配置使用的Embedding
```

### 数据预处理
预处理好的结果都存储在 `cache/` 目录下。
如果想重新跑预处理，请执行 `assets_v2.ipynb` 和 `sql_template.ipynb` 里的代码，注意，其中有调用大模型能力的步骤，会消耗token。
并且由于大模型的随机性，得到的结果可能跟之前不一样。

`cache/`目录下的文件说明如下：

- schema.json: 所有数据库的schema信息
- unuse_columns.json: 字段值全是NULL的字段
- nullable_columns.json: 字段值可能为NULL的字段，包括NULL的占比
- table_relations.json: 表与表之间的外链关系
- column_questions.json: 针对每个字段，用大模型生成的用户提问
- column_vectors.npy: 上面每个字段的用户提问的向量
- column_bm25.pkl: 上面每个字段的用户提问的bm25词频
- db_table.json: 用大模型对每个数据表和每个数据库进行信息归纳得到的文本描述
- sql_template.json: 基于公开的数据，人工收集的"问题-SQL"样例
- sql_template_vectors.npy: 上面每个"问题-SQL"样例的向量

---

### DEBUG
在 manual.ipynb 里，可以针对某道题手工跑，精调prompt。

### 输出
在 output 目录下，看到很多 `ttt--`开头的就是每道题目的解题过程。
而 all_question.json 里会缓存一些中间数据。
Eva_Now_result.json 是最终可以提交的文件。

---

Have Fun ^_^
