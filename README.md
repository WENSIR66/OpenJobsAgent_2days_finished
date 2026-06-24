# OpenJobs Candidate Screening Agent

当前阶段完成候选人知识库的第一步：清洗 1000 条 profile，将每条记录转成
LangChain `Document`，并建立 SQLite/FTS5 BM25 与 FAISS 向量索引。

## 目录

```text
backend/app/ingestion/
├── cleaning.py     # 空白、异常值、重复项、日期/数值清洗
├── documents.py    # page_content 与 metadata 构建
├── storage.py      # SQLite 主存储和 FTS5/BM25
├── embeddings.py   # GLM embedding-3 与 FAISS
├── pipeline.py     # 端到端导入流程
└── config.py       # 环境变量配置
scripts/
└── ingest_profiles.py
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

## 清洗原则

- 使用流式逐行 JSON 解析，不用会误切 Unicode 行分隔符的 `splitlines()`。
- 统一 Unicode、零宽字符和连续空白，去掉页面抓取残留的 `Show less`。
- 空字符串转为缺失值；集合字段统一为列表；技能等标量列表去重。
- `company_size_range=-1` 视为未知；非法年月、负数时长记录告警并置空。
- 不删除 `MASKED` 等脱敏内容，不臆造缺失字段，不用假向量替代 API 结果。
