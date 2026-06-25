# OpenJobs Candidate Screening Agent

当前已完成候选人知识库和第一版 RAG 搜索链路：清洗 1000 条 profile，将每条记录
转成 LangChain `Document`，建立 SQLite/FTS5 BM25 与 FAISS 向量索引，并通过
GLM-4.5-air 解析查询和生成推荐理由。

## 目录

```text
backend/app/ingestion/
├── cleaning.py     # 空白、异常值、重复项、日期/数值清洗
├── documents.py    # page_content 与 metadata 构建
├── storage.py      # SQLite 主存储和 FTS5/BM25
├── embeddings.py   # GLM embedding-3 与 FAISS
├── pipeline.py     # 端到端导入流程
└── config.py       # 环境变量配置
backend/app/rag/
├── query_parser.py # LLM 查询解析 + 规则兜底
├── retriever.py    # SQL 硬过滤、向量/BM25 混合召回、融合排序
├── glm.py          # GLM Chat Completions 客户端
├── models.py       # 查询条件与返回结构
└── service.py      # RAG 编排和推荐理由
backend/app/main.py # FastAPI 和 Markdown 聊天页
scripts/
├── ingest_profiles.py
└── test_rag.py
tests/
└── test_ingestion.py
```

## 存储设计

- `candidates`：`candidate_id`、结构化 metadata、原始 JSON、清洗后 JSON、
  `page_content`、内容哈希、清洗告警。
- `data/processed/candidates.cleaned.jsonl`：可独立审计和复用的清洗后 JSONL。
- `candidate_fts`：以 `page_content` 建立的 SQLite FTS5 索引，查询时使用
  `bm25()` 排序。
- `embedding_cache`：按 `candidate_id + model + content_hash` 缓存向量，失败后可续跑。
- `data/indexes/candidates.faiss`：归一化向量的 FAISS `IndexFlatIP`。
- `data/indexes/candidates.manifest.json`：FAISS 行号到 `candidate_id` 的稳定映射。

适合硬过滤的 metadata 包括：在职状态、当前职位/部门、管理层级、决策者标记、
总工作年限、最高学历层级。技能、角色、行业、公司、地点、专业等多值字段以 JSON
保存，后续可根据查询解析结果做精确或规范化过滤。

`page_content` 保留适合语义/关键词召回的 headline、summary、skills、完整工作经历、
教育、证书、课程、奖项、出版物和专利。内部 ID、公司规模编码等不进入检索文本。

## 运行

```bash
python -m venv .venv
./.venv/bin/python -m pip install -e '.[dev]'
cp .env.example .env
```

在 `.env` 中填写 `ZHIPUAI_API_KEY` 后运行完整导入：

```bash
./.venv/bin/python scripts/ingest_profiles.py
```

暂时不调用 embedding API，只构建清洗数据和 BM25：

```bash
./.venv/bin/python scripts/ingest_profiles.py --skip-embeddings
```

运行测试：

```bash
./.venv/bin/python -m pytest
```

## 查询链路

1. GLM-4.5-air 将自然语言解析成 `semantic_query`、`metadata_filter_must` 和
   `metadata_filter_should`；API 异常时使用本地规则兜底。
2. must 条件通过 SQLite 先过滤候选人。
3. 在过滤结果中分别取向量 Top20 和 BM25 Top20。
4. 两路结果去重，并各自归一化到 0–1。
5. should 条件按命中比例得到 0–1 分数。
6. 按 `0.35 * vector + 0.35 * bm25 + 0.30 * metadata_should` 排序。
7. GLM-4.5-air 根据 Top5 的原始履历、metadata 和分数组成生成 Markdown 推荐理由。

查询拆解约定：

- 核心编程语言/技术栈、明确年限等属于 must。
- “优先、最好、加分、倾向”等从句属于 should。
- 前端、后端、全栈等宽泛方向只保留在 `semantic_query`，不作为结构化过滤条件。
- 字段路由使用“主字段 + 合理辅助字段”：skills 可查技能、经历描述、summary、
  职位标题和 headline；roles 可查当前/历史职位、headline 和标准角色；industry、
  company、location、major、certification 也各自配置了相关辅助来源。
- `experience_descriptions` 始终只查工作经历描述，避免概念漂移。
- 技能关键词会扩展常见变体与生态词，例如 Python 同时搜索 Python3、Django、
  Flask、FastAPI。
- headline、summary、证书、专业、公司、地点等均有独立字段，不跨字段借用证据。
- 同一概念的多个变体是 OR 关系；不同 must 条件之间仍是 AND 关系。
- 英文关键词和 BM25 查询均忽略大小写。

性能相关实现：

- GLM 查询解析、query embedding 和理由生成共用一个 `httpx.AsyncClient` 连接池，
  复用 HTTPS keep-alive 连接。
- query embedding/FAISS 与 SQLite FTS5/BM25 通过 `asyncio.gather` 并行执行。
- 阻塞的 SQLite 和 FAISS 操作放入工作线程，不阻塞 FastAPI 事件循环。
- 连接池在 FastAPI 启动时创建，在应用关闭时统一释放。

命令行端到端测试：

```bash
./.venv/bin/python scripts/test_rag.py \
  "寻找至少5年经验的Python后端工程师，有云平台或DevOps经验优先"
```

启动聊天页面：

```bash
./.venv/bin/uvicorn backend.app.main:app --reload
```

浏览器打开 `http://127.0.0.1:8000`。API 为 `POST /api/chat`：

```json
{
  "conversation_id": "浏览器或客户端生成的会话 ID",
  "message": "寻找至少5年经验的Python后端工程师，有云平台经验优先"
}
```

## 多轮交互

每个 `conversation_id` 只维护两个状态：

- `current_query`：当前完整查询
- `current_results`：当前展示的候选人

支持四种意图：

- `new_search`：清空状态并完整检索。
- `refine`：结合当前查询重写完整查询，只在 `current_results` 范围内过滤和重排。
- `compare`：按序号或候选人 ID 定位候选人，并结合当前查询进行对比。
- `follow_up`：解释某位候选人、整个列表或当前排序。

程序重启后内存会话自动清空。命令行多轮测试：

```bash
./.venv/bin/python scripts/test_multiturn.py
```

## 离线排序评估

`scripts/evaluate_judge_ranking.py` 使用 LLM-as-a-judge 快速评估系统 Top10 排序：

1. 对 `data/eval_queries.json` 中的 10 条招聘查询执行解析、过滤、向量召回、BM25
   和融合排序，不调用推荐理由生成。
2. 将系统 Top10 确定性随机打乱，只把原始 query 和候选人资料交给 Judge；Judge
   看不到系统 rank、final score 或各分项得分。
3. 将 Judge Top5 作为弱监督强相关集合，计算 Top5 Overlap、二元 nDCG@10 和 MRR。
4. Judge JSON 解析失败时重试一次，仍失败则跳过该查询的平均指标。

运行：

```bash
./.venv/bin/python scripts/evaluate_judge_ranking.py
```

输出：

- `reports/judge_ranking_eval_results.json`
- `reports/judge_ranking_eval_report.md`

评估报告会说明：本评估使用 LLM-as-a-judge 对候选人 Top10 进行相对排序，Judge
Top5 被视为弱监督强相关集合，用于快速评估系统排序质量。

注意：运行该脚本会将每个查询对应的打乱后 Top10 候选人资料发送给配置的 GLM API。

## 清洗原则

- 使用流式逐行 JSON 解析，不用会误切 Unicode 行分隔符的 `splitlines()`。
- 统一 Unicode、零宽字符和连续空白，去掉页面抓取残留的 `Show less`。
- 空字符串转为缺失值；集合字段统一为列表；技能等标量列表去重。
- `company_size_range=-1` 视为未知；非法年月、负数时长记录告警并置空。
- 不删除 `MASKED` 等脱敏内容，不臆造缺失字段，不用假向量替代 API 结果。
