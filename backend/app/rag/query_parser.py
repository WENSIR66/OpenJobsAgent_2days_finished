from __future__ import annotations

import json
import re

from .glm import AsyncGLMClient
from .models import MetadataCondition, ParsedQuery

SYSTEM_PROMPT = """
你是候选人检索查询解析器。只输出 JSON，不要解释。

输出结构：
{
  "semantic_query": "用于向量和 BM25 的完整英文检索语义",
  "metadata_filter_must": [],
  "metadata_filter_should": []
}

每个条件结构：
{"field": "...", "operator": "eq|contains|gte|lte|in", "value": ...}

可用字段：
is_working, current_title, current_department, management_level,
is_decision_maker, total_experience_months, total_experience_years,
highest_degree_level, skills, roles, levels, industries, companies,
locations, majors, education_countries。

规则：
1. must 是用户寻找目标的核心要求，不满足直接淘汰，包括明确年限、状态、学历和
   核心技能。
2. should 是“优先、最好、加分、倾向”等偏好，只加分，不淘汰。
3. skills 和 roles 的 must 不是只查 metadata 单字段，而会在候选人的 headline、
   summary、skills、当前/历史职位和工作经历描述中做全文证据匹配。
4. 对核心技能必须扩展常见写法和生态技术，使用一个 in 条件。
   例如 Python 可扩展为 Python、Python3、Django、Flask、FastAPI。
5. 同一概念的变体放在同一个 in 条件中，表示任一变体命中即可；不要拆成多个
   必须同时成立的条件。
6. “前端、后端、全栈”这类宽泛开发方向只保留在 semantic_query 中，不生成
   metadata_filter_must 或 metadata_filter_should。
7. semantic_query 保留完整岗位语义以及偏好，用于向量和 BM25 召回。
8. 所有英文字母匹配忽略大小写，Python、PYTHON、python 视为同一个词。
9. highest_degree_level：0 专科/副学士及以下，1 本科，2 硕士，3 博士。
10. 不要把“优先、最好、加分、倾向”从句误放到 must。

示例 1
用户：寻找至少5年经验的Python后端工程师，有云平台经验优先。
输出：
{
  "semantic_query": "Python backend engineer with cloud platform experience",
  "metadata_filter_must": [
    {"field": "total_experience_years", "operator": "gte", "value": 5},
    {
      "field": "skills",
      "operator": "in",
      "value": ["Python", "Python3", "Django", "Flask", "FastAPI"]
    }
  ],
  "metadata_filter_should": [
    {
      "field": "skills",
      "operator": "in",
      "value": ["AWS", "GCP", "Azure", "cloud platform", "cloud infrastructure"]
    }
  ]
}

示例 2
用户：找在职的数据分析师，至少3年经验，必须会SQL；Tableau优先，有金融行业经验加分。
输出：
{
  "semantic_query": "SQL data analyst with Tableau and financial industry experience",
  "metadata_filter_must": [
    {"field": "is_working", "operator": "eq", "value": true},
    {"field": "total_experience_years", "operator": "gte", "value": 3},
    {
      "field": "skills",
      "operator": "in",
      "value": ["SQL", "MySQL", "PostgreSQL", "T-SQL", "data querying"]
    },
    {
      "field": "roles",
      "operator": "in",
      "value": ["data analyst", "BI analyst", "business intelligence", "analytics analyst"]
    }
  ],
  "metadata_filter_should": [
    {"field": "skills", "operator": "in", "value": ["Tableau", "Power BI", "data visualization"]},
    {"field": "industries", "operator": "in", "value": ["finance", "financial services", "banking"]}
  ]
}

示例 3
用户：招聘8年以上Java后端工程师，Spring Boot和微服务是硬性要求，AWS优先，硕士最好。
输出：
{
  "semantic_query": "Java Spring Boot microservices backend engineer with AWS experience",
  "metadata_filter_must": [
    {"field": "total_experience_years", "operator": "gte", "value": 8},
    {
      "field": "skills",
      "operator": "in",
      "value": ["Java", "J2EE", "Spring", "Spring Boot"]
    },
    {"field": "skills", "operator": "in", "value": ["microservices", "microservice architecture"]}
  ],
  "metadata_filter_should": [
    {"field": "skills", "operator": "in", "value": ["AWS", "Amazon Web Services"]},
    {"field": "highest_degree_level", "operator": "gte", "value": 2}
  ]
}

示例 4
用户：找B端SaaS产品经理，至少4年经验，做过从0到1产品优先，会英语加分。
输出：
{
  "semantic_query": "B2B SaaS product manager with zero-to-one product and English experience",
  "metadata_filter_must": [
    {"field": "total_experience_years", "operator": "gte", "value": 4},
    {
      "field": "roles",
      "operator": "in",
      "value": ["product manager", "product owner", "product lead", "SaaS product"]
    }
  ],
  "metadata_filter_should": [
    {
      "field": "skills",
      "operator": "in",
      "value": ["0-to-1", "zero-to-one", "new product development", "product launch"]
    },
    {"field": "skills", "operator": "in", "value": ["English", "bilingual"]}
  ]
}
"""

SKILL_EXPANSIONS = {
    "python": ["Python", "Python3", "Django", "Flask", "FastAPI"],
    "java": ["Java", "J2EE", "Spring", "Spring Boot"],
    "sql": ["SQL", "MySQL", "PostgreSQL", "T-SQL", "data querying"],
    "cloud": ["AWS", "GCP", "Azure", "cloud platform", "cloud infrastructure"],
}
ROLE_EXPANSIONS = {
    "data analyst": [
        "data analyst",
        "BI analyst",
        "business intelligence",
        "analytics analyst",
    ],
    "product manager": ["product manager", "product owner", "product lead", "SaaS product"],
}

IGNORED_DIRECTION_TERMS = {
    "backend",
    "back-end",
    "backend engineer",
    "backend developer",
    "server-side",
    "frontend",
    "front-end",
    "frontend engineer",
    "frontend developer",
    "fullstack",
    "full-stack",
    "full stack",
}


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _expand_condition(condition: MetadataCondition) -> MetadataCondition:
    values = condition.value if isinstance(condition.value, list) else [condition.value]
    text = " ".join(str(value).casefold() for value in values)
    expansions = SKILL_EXPANSIONS if condition.field == "skills" else ROLE_EXPANSIONS
    for concept, aliases in expansions.items():
        if concept in text or any(alias.casefold() in text for alias in aliases):
            return MetadataCondition(field=condition.field, operator="in", value=aliases)
    return condition


def _is_ignored_direction(condition: MetadataCondition) -> bool:
    if condition.field not in {"roles", "current_title", "current_department"}:
        return False
    values = condition.value if isinstance(condition.value, list) else [condition.value]
    text = " ".join(str(value).casefold() for value in values)
    return any(term in text for term in IGNORED_DIRECTION_TERMS)


def _normalize_conditions(parsed: ParsedQuery) -> ParsedQuery:
    return ParsedQuery(
        semantic_query=parsed.semantic_query,
        metadata_filter_must=[
            _expand_condition(condition)
            for condition in parsed.metadata_filter_must
            if not _is_ignored_direction(condition)
        ],
        metadata_filter_should=[
            _expand_condition(condition)
            for condition in parsed.metadata_filter_should
            if not _is_ignored_direction(condition)
        ],
    )


def _rule_fallback(query: str) -> ParsedQuery:
    must: list[MetadataCondition] = []
    should: list[MetadataCondition] = []
    semantic_parts: list[str] = []

    years = re.search(r"(?:至少|不少于|有)?\s*(\d+(?:\.\d+)?)\s*年", query)
    if years:
        must.append(
            MetadataCondition(
                field="total_experience_years",
                operator="gte",
                value=float(years.group(1)),
            )
        )
    if re.search(r"在职|目前工作", query):
        must.append(MetadataCondition(field="is_working", operator="eq", value=True))

    for concept, aliases in SKILL_EXPANSIONS.items():
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(concept)}(?![A-Za-z0-9])", query, re.I):
            must.append(MetadataCondition(field="skills", operator="in", value=aliases))
            semantic_parts.append(concept)

    role_patterns = {
        r"数据分析": ("data analyst", ROLE_EXPANSIONS["data analyst"]),
        r"产品经理": ("product manager", ROLE_EXPANSIONS["product manager"]),
    }
    for pattern, (semantic, aliases) in role_patterns.items():
        if re.search(pattern, query):
            must.append(MetadataCondition(field="roles", operator="in", value=aliases))
            semantic_parts.append(semantic)
            break

    if re.search(r"后端|服务端", query):
        semantic_parts.append("backend engineer")
    elif re.search(r"前端", query):
        semantic_parts.append("frontend engineer")
    elif re.search(r"全栈", query):
        semantic_parts.append("full stack engineer")

    if re.search(r"云平台|云经验|AWS|GCP|Azure", query, re.I):
        cloud = MetadataCondition(field="skills", operator="in", value=SKILL_EXPANSIONS["cloud"])
        if re.search(r"优先|最好|加分|倾向", query):
            should.append(cloud)
        else:
            must.append(cloud)
        semantic_parts.append("cloud platform experience")

    return ParsedQuery(
        semantic_query=" ".join(semantic_parts) or query,
        metadata_filter_must=must,
        metadata_filter_should=should,
    )


class QueryParser:
    def __init__(self, client: AsyncGLMClient) -> None:
        self.client = client

    async def parse(self, query: str) -> ParsedQuery:
        try:
            content = await self.client.complete(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return _normalize_conditions(
                ParsedQuery.model_validate(_extract_json(content))
            )
        except Exception:
            return _rule_fallback(query)
