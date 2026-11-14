# 项目数据

固定产品为 **太保盛世鸿运S（恒赢版）终身寿险（分红型）**，条款编码为 **太平洋人寿[2026]终身寿险085号**。官方来源见 [sources.json](raw/sources.json)，下载文件及哈希见 [download_manifest.json](raw/download_manifest.json)。

## 目录与构建

| 目录 | 内容 |
| --- | --- |
| `raw/` | 来源清单、下载清单、`pdf/` 官方原件、`listings/` 官网列表快照 |
| `pages/` | 每份PDF对应的逐页原文JSONL |
| `corpus/<语料版本>/` | 证据单元、原文位置、关系、覆盖清单及manifest |
| `index/<索引版本>/` | 关键词索引、向量及manifest，引用对应语料版本 |
| `evaluation/` | 题目要求的测试集和评分标签，不参与语料或索引构建 |

生成顺序为 `raw/pdf/ → pages/ → corpus/<语料版本>/ → index/<索引版本>/`。在项目根目录执行：

```bash
python scripts/crawl_cpic.py
python main.py build-corpus
python main.py build-index --build-dir data/corpus/<语料版本>
```

下载与抽取需要Poppler。将 `<语料版本>` 替换为语料构建返回的目录名；索引构建后，在 [pipeline.yaml](../configs/pipeline.yaml) 中设置返回的 `build_id`。各版本及文件哈希保存在对应目录的 `manifest.json` 中。

## 原始材料

| 文件 | 页数 | 用途 |
| --- | ---: | --- |
| `raw/pdf/sshy_s_hengying_terms.pdf` | 19 | 主要回答依据 |
| `raw/pdf/sshy_s_hengying_description.pdf` | 15 | 显式扩展范围下的产品说明 |
| `raw/pdf/sshy_s_hengying_sum_assured.pdf` | 720 | 基本保额和购买交清保额净保费表 |
| `raw/pdf/sshy_s_hengying_cash_value.pdf` | 1437 | 现金价值表 |

逐页JSONL由 `pdftotext -layout` 抽取，记录原始文本、物理页码、来源和文档类型。PDF是最终核验依据；段落清理只影响检索表示，不改变原文位置。官网上传目录日期不作为合同生效日期。

## 证据与索引

| 语料文件 | 内容 |
| --- | --- |
| `manifest.json` | 来源、版本和文件哈希 |
| `evidence_units.jsonl` | 证据正文、检索文本和来源跨度ID |
| `source_spans.jsonl` | PDF哈希、物理页码、Unicode字符区间和精确原文 |
| `relations.jsonl` | 必要前提、定义和明示引用 |
| `coverage.jsonl` | 每页收录、排除区间及原因 |

索引目录保存 `lexical.json`、`vectors.npy`、`manifest.json`，包含词频、ID映射、向量与关联语料版本。短条款整节保留，复杂责任按完整分支切分；跨页内容、全残定义注释和双人免责范围随证据保留。条款第1–2页目录与指引、说明书第9–15页投保示例不进入常规检索。

默认 `terms_only` 仅使用条款。说明书单独标记来源；仅在说明书明确的信息不能自动用于条款范围的答案。两份数表仅保留目录与适用条件，未开放个人精确金额查询；空单元格、年龄或性别选择规则、公式上标均不能凭排版推断。

## 保留测试集

- `evaluation/fixed_questions.jsonl`：原Excel的10道标准问和10道口语问；原附件未提供参考答案。
- `evaluation/final_questions.jsonl`：12道自建问题。
- `evaluation/fixed_review_criteria.json`：官方问题对应的原文检查点与评分标签。
- `evaluation/final_review_criteria.json`：自建问题对应的原文检查点与评分标签。

评分标签由助手按原文整理，供离线审查使用，未经人工金标准审定；不输入在线问答链路。

## 自建评测集的设计

自建集共12道单轮问题，六类各2道。问题由助手依据本产品官方条款和项目中已知的理解风险编写，不是实际客户对话，也不复制固定题原句。每题的预期行为、必答要点、禁止出现的错误、相关章节和PDF页码保存在 [final_review_criteria.json](evaluation/final_review_criteria.json)。

| 覆盖范围 | 题号 | 检查重点 |
| --- | --- | --- |
| 主体与角色 | 01–02 | 区分投保人、被保险人及不同主体的适用条件 |
| 时间与起算点 | 03–04 | 区分自然年、保单年度、生效与复效起点 |
| 双人责任分支 | 05–06 | 区分一人身故、同时身故及领取主体 |
| 多目标与概念比较 | 07–08 | 不漏答、不混用规则及比例基数 |
| 来源与能力边界 | 09–10 | 不编造确定金额，不引用允许范围外的信息 |
| 必要追问 | 11–12 | 条件不足时指出缺失信息，不猜测事故或分支 |

构造时先确定条款中的易混概念和条件，再写成口语、条件组合、错误前提或信息不足的问题，最后回查原文确定评分依据。采用六类均衡覆盖，是为了补充固定20问的诊断范围；每类两题可减少只凭单个例子判断能力的偶然性，12题也便于逐条核对答案和出处。这个规模用于小型项目的定向诊断，不能估计真实客户问题的总体准确率，也不覆盖连续多轮对话。

评分分别记录完整答对、必要追问、部分回答、实质错误和未回答；完整答对要求目标覆盖、条件正确和出处支持，必要追问单独统计。这套题已用于开发排查，属于开发评测集，不能称为独立盲测集。
