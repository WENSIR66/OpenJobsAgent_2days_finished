# Judge Ranking Evaluation

本评估使用 LLM-as-a-judge 对候选人 Top10 进行相对排序，Judge Top5 被视为弱监督强相关集合，用于快速评估系统排序质量。

## Summary

| Metric | Value |
|---|---:|
| mean_top5_overlap | 0.6889 |
| mean_ndcg_at_10 | 0.9858 |
| mean_mrr | 0.6944 |
| evaluated_queries | 9 / 10 |
| judge_failed_queries | 1 |

## Query Results

### 1. 寻找至少5年经验的Python工程师，有AWS或GCP云平台经验优先。

- Judge failed: `false`
- Top5_Overlap: `0.8000`
- nDCG@10: `0.9758`
- MRR: `0.2500`
- System Top10: `['636422631', '156898517', '774208709', '29411937', '160948639', '433578071', '436640423', '393092301', '474179942', '360439535']`
- LLM Judge Top10: `['29411937', '774208709', '636422631', '156898517', '393092301', '360439535', '160948639', '433578071', '474179942', '436640423']`
- Judge Top5: `['29411937', '774208709', '636422631', '156898517', '393092301']`
- Mismatch cases:
  - system_top5_but_not_judge_top5: `['160948639']`
  - judge_top5_but_not_system_top5: `['393092301']`

### 2. 找一名Java工程师，熟悉Spring Boot和微服务，8年以上经验，AWS经验优先。

- Judge failed: `false`
- Top5_Overlap: `0.6000`
- nDCG@10: `1.0000`
- MRR: `1.0000`
- System Top10: `['635878189', '156898517', '774208709']`
- LLM Judge Top10: `['635878189', '156898517', '774208709']`
- Judge Top5: `['635878189', '156898517', '774208709']`
- Mismatch cases:
  - system_top5_but_not_judge_top5: `[]`
  - judge_top5_but_not_system_top5: `[]`

### 3. 寻找数据分析师，要求会SQL，有Tableau或Power BI经验优先。

- Judge failed: `false`
- Top5_Overlap: `0.6000`
- nDCG@10: `1.0000`
- MRR: `1.0000`
- System Top10: `['638005831', '478171317', '427539918']`
- LLM Judge Top10: `['638005831', '478171317', '427539918']`
- Judge Top5: `['638005831', '478171317', '427539918']`
- Mismatch cases:
  - system_top5_but_not_judge_top5: `[]`
  - judge_top5_but_not_system_top5: `[]`

### 4. 找有政府合同或国防项目经验的项目经理，具备团队管理经验优先。

- Judge failed: `true`
- Judge error: `System returned no candidates`
- System Top10: `[]`

### 5. 寻找医疗账单和保险理赔方向的候选人，熟悉合规、Medicare或Medicaid优先。

- Judge failed: `false`
- Top5_Overlap: `0.6000`
- nDCG@10: `1.0000`
- MRR: `0.3333`
- System Top10: `['99486673', '231763512', '806050755']`
- LLM Judge Top10: `['806050755', '99486673', '231763512']`
- Judge Top5: `['806050755', '99486673', '231763512']`
- Mismatch cases:
  - system_top5_but_not_judge_top5: `[]`
  - judge_top5_but_not_system_top5: `[]`

### 6. 找有Kubernetes、Docker和CI/CD经验的DevOps或云工程师。

- Judge failed: `false`
- Top5_Overlap: `0.6000`
- nDCG@10: `1.0000`
- MRR: `1.0000`
- System Top10: `['636422631', '635878189', '226577945']`
- LLM Judge Top10: `['636422631', '226577945', '635878189']`
- Judge Top5: `['636422631', '226577945', '635878189']`
- Mismatch cases:
  - system_top5_but_not_judge_top5: `[]`
  - judge_top5_but_not_system_top5: `[]`

### 7. 寻找产品经理，做过B2B或SaaS产品，有从0到1产品经验优先。

- Judge failed: `false`
- Top5_Overlap: `0.8000`
- nDCG@10: `0.9068`
- MRR: `1.0000`
- System Top10: `['123173099', '396506447', '433657894', '393414383', '9312039', '75485227']`
- LLM Judge Top10: `['123173099', '75485227', '393414383', '9312039', '433657894', '396506447']`
- Judge Top5: `['123173099', '75485227', '393414383', '9312039', '433657894']`
- Mismatch cases:
  - system_top5_but_not_judge_top5: `['396506447']`
  - judge_top5_but_not_system_top5: `['75485227']`

### 8. 找具备机器学习、数据科学或算法经验的候选人，Python经验优先。

- Judge failed: `false`
- Top5_Overlap: `0.8000`
- nDCG@10: `1.0000`
- MRR: `0.5000`
- System Top10: `['522332113', '395210742', '632892002', '636137009']`
- LLM Judge Top10: `['395210742', '632892002', '636137009', '522332113']`
- Judge Top5: `['395210742', '632892002', '636137009', '522332113']`
- Mismatch cases:
  - system_top5_but_not_judge_top5: `[]`
  - judge_top5_but_not_system_top5: `[]`

### 9. 寻找至少10年经验的软件架构师，有微服务、分布式系统和云迁移经验优先。

- Judge failed: `false`
- Top5_Overlap: `0.6000`
- nDCG@10: `1.0000`
- MRR: `1.0000`
- System Top10: `['521932073', '29411937', '351339161']`
- LLM Judge Top10: `['521932073', '29411937', '351339161']`
- Judge Top5: `['521932073', '29411937', '351339161']`
- Mismatch cases:
  - system_top5_but_not_judge_top5: `[]`
  - judge_top5_but_not_system_top5: `[]`

### 10. 找财务或会计管理方向候选人，具备预算、审计或应收账款经验优先。

- Judge failed: `false`
- Top5_Overlap: `0.8000`
- nDCG@10: `0.9896`
- MRR: `0.1667`
- System Top10: `['217314122', '395242755', '480990750', '29322809', '156985353', '69758260', '218118758', '69815773', '521997531', '395241535']`
- LLM Judge Top10: `['69758260', '480990750', '395242755', '217314122', '29322809', '156985353', '218118758', '69815773', '521997531', '395241535']`
- Judge Top5: `['69758260', '480990750', '395242755', '217314122', '29322809']`
- Mismatch cases:
  - system_top5_but_not_judge_top5: `['156985353']`
  - judge_top5_but_not_system_top5: `['69758260']`
