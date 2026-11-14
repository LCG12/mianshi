"""把通过检查的草稿编译为正文和原文出处；结构正确不等于语义正确。"""

import re
from urllib.parse import urlsplit

from .citations import validate_citation_references
from .errors import AppError
from .schemas import AnswerBlock, AnswerResult, Citation, Gap


def _require(ok, code, message):
    if not ok:
        raise AppError(code, message, 422)


def _unique(values):
    _require(all(isinstance(value, str) and value.strip() for value in values)
             and len(values) == len(set(values)), "invalid_answer_structure", "答案标识为空或重复。")
    return set(values)


def _product_guards(claim, claims, evidence):
    """本产品已核对的边界；仅检查本句及其声明前提实际引用的条款。"""
    pending, visited, sections = [claim.claim_id], set(), set()
    while pending:
        cid = pending.pop()
        if cid not in visited:
            visited.add(cid)
            sections.update(evidence[eid].section for eid in claims[cid].evidence_ids)
            pending.extend(claims[cid].depends_on_claim_ids)
    text = "\n".join(re.sub(r"\s+", "", claims[cid].text) for cid in sorted(visited))
    sentences = re.split(r"[。；;\n]", text)

    def positive(pattern):
        # 复杂否定保守拒绝，词面通过仍不构成语义证明。
        return any(re.search(pattern, sentence) and not re.search(
            r"不是|并非|非单人|非因|不因|无需|无须|不必|不需要|不用|不要求|不限|无论|不论|尚未|未达到", sentence)
            for sentence in sentences)

    if any(section.startswith("2.5.") for section in sections):
        _require(not re.search(r"(?:起算点|起算日|起点)(?:为|是)[^，。；;\n]{0,80}"
                               r"(?:90|九十)(?:日|天)(?:之|以)?后(?:[，。；;\n]|$)", text),
                 "benefit_start_point_mismatch", "90日从生效与最后一次复效的较迟日开始计算；90日后是常规给付门槛，不是起算点。")

    if "1.3" in sections:
        _require(not re.search(r"投保人(?:的)?(?:投保)?年龄(?:范围|上限|限制)", claim.text),
                 "eligibility_subject_mismatch", "投保年龄范围限定被保险人的投保周岁，不能写成投保人的年龄限制。")
    if "6.2" in sections and re.search(r"(?:可(?:以)?|能)(?:申请|办理)(?:保单)?贷款", claim.text):
        conditions = (r"合同有效", r"犹豫期后", r"审核同意", r"被保险人[^。；;\n]{0,16}书面同意")
        _require(all(positive(p) for p in conditions), "policy_value_conditions_missing",
                 "可办理保单贷款须保留合同有效、犹豫期后、审核同意及被保险人书面同意的前提。")
    if "2.1" in sections and re.search(r"退还[^。；;\n]{0,30}现金价值", claim.text):
        conditions = (r"审核同意", r"(?:保险费|保费)[^。；;\n]{0,12}不低于[^。；;\n]{0,12}最低标准")
        _require(all(positive(p) for p in conditions), "policy_value_conditions_missing",
                 "减保退款须保留公司审核同意、减少后保险费不低于规定最低标准的前提。")

    if "2.6.2" in sections:
        # 原文(8)–(12)仅为意外身故/全残豁免保险费的免责，不能扩成保险金免责。
        waiver_only = (r"猝死|医疗事故|药物过敏|未遵医嘱|私自使用药物|怀孕|流产|堕胎|分娩|剖宫产|异位妊娠|"
                       r"避孕|节育|绝育|不孕不育|产前产后检查|高风险活动|潜水|跳伞|攀岩|滑翔机|滑翔伞|"
                       r"探险|摔跤|武术比赛|特技表演|赛马|赛车")
        waiver = r"(?:保险费|保费)豁免|豁免(?:保险费|保费|责任)"
        for sentence in re.split(r"[。\n]", claim.text):
            # 同一句内仅列保费豁免项目时，分号不切断清单的共同范围。
            waiver_list = re.search(waiver, sentence) and not re.search(r"保险金|保险责任|不赔|拒赔", sentence)
            _require(waiver_list or all(re.search(waiver, part) or not re.search(waiver_only, part)
                                       for part in re.split(r"[；;]", sentence)),
                     "expanded_exclusion_scope", "猝死等条目属于豁免保险费的免责，不能扩大为身故或全残保险金免责。")
    if any(section.startswith("2.5.2.") for section in sections):
        # 包括反向否定句也保守拒绝；此处不尝试用词面规则解释复杂否定。
        _require(not re.search(r"每(?:一)?(?:名|位|个)受益人[^。；;\n]{0,60}(?:50\s*[%％]|一半|百分之五十)", claim.text),
                 "beneficiary_scope_ambiguous", "应说明每名被保险人对应的受益人合计领取比例，不能写成每位受益人各领一半。")
    # 仅借用公式的 reference 不代表适用该条款的责任条件。
    rule_sections = {evidence[eid].section for cid in visited for eid in claims[cid].evidence_ids
                     if "reference" not in evidence[eid].roles}
    early = {"2.5.1.early90", "2.5.2.ordered_early90", "2.5.2.unordered_early90"}
    payout = re.search(r"(?:按|以|将|向|我们|本公司|保险公司)[^。；;\n]{0,100}"
                       r"(?<!不予)(?<!不承担)(?:给付|赔付)(?!的?(?:比例|份额|条件|规则|分支))", claim.text)
    if rule_sections & early and payout:
        conditions = (r"合同(?:仍|持续)?(?:处于|在)?有效",
                      r"(?:不属(?:于)?|不涉及|不存在|未触发|不在)[^。；;\n]{0,12}(?:免责|责任免除)")
        _require(all(positive(p) for p in conditions), "early_benefit_limits_missing",
                 "90日内的给付结论也须保留合同有效、不属于对应免责的前提；引用原文不能代替本句或声明前提。")
    if rule_sections & {"2.5.1.under18", "2.5.1.before_payment_end", "2.5.1.after_payment_end"}:
        # 仅筛查本句及已声明前提中的明示条件；引用原文、定义或免责不能代替正文条件。
        single_insured = positive(r"被保险人(?:人数)?(?:为|是|仅为|只有)[一1](?:人|名)|"
                                  r"(?:仅有|只有)[一1](?:名|位)被保险人|(?:减少|减)为[一1](?:名|位)被保险人|单人")
        nonaccidental = r"非意外|意外(?:伤害)?以外|疾病"
        accidental = positive(r"(?:因(?:遭受)?|遭受)意外伤害[^，。；;\n]{0,20}(?:身故|全残)|意外(?:身故|全残)")
        after_90_days = positive(r"生效[^。；;\n]{0,32}最后(?:一次)?复效[^。；;\n]{0,32}"
                                 r"(?:90|九十)(?:日|天)(?:之|以)?后")
        nonaccidental_after_90_days = (
            positive(nonaccidental + r"[^，。；;\n]{0,20}(?:身故|全残)") and after_90_days
            and positive(r"较迟|较晚")
            and not re.search(r"(?:90|九十)(?:日|天)(?:内|以内|前)", text)
        )
        _require(single_insured, "single_insured_scope_missing", "单人给付分支须明示仅一名被保险人或两名生存者减少为一名。")
        if re.search(nonaccidental, text):
            trigger_present = nonaccidental_after_90_days
        else:
            trigger_present = accidental
        _require(trigger_present,
                 "benefit_trigger_missing", "须明示意外身故或全残，或生效/最后一次复效90日后的非意外身故或全残条件。")

    denial = r"不承担|拒赔|不赔|不予赔付"
    if ("10.1" in sections and re.search(r"如实告知|告知义务", text)
            and re.search(denial + r"|(?:有权|可(?:以)?)解除", claim.text)):
        # 只约束告知引起的不赔/解除结论；不把10.2外推到年龄性别错误等其他解除权。
        limits = (
            r"足以影响[^。；;\n]{0,24}(?:承保|保险费率|费率)",
            r"(?:知道|知悉)有?解除事由[^。；;\n]{0,12}(?:30|三十)(?:日|天)[^。；;\n]{0,12}不行使[^。；;\n]{0,8}(?:消灭|失效)",
            r"成立[^。；;\n]{0,12}超过(?:2|两|二)年[^。；;\n]{0,12}(?:不得|不能|无权)解除",
            r"订立(?:合同时|时)[^。；;\n]{0,8}(?:已经|已)知道[^。；;\n]{0,24}(?:不得|不能|无权)解除",
        )
        _require("10.2" in sections and all(positive(p) for p in limits), "disclosure_limits_missing",
                 "告知导致不赔或解除的说明须保留承保影响门槛及30日、2年和订约已知的解除权限制。")
        _require(not ("重大过失" in text and re.search(denial, claim.text))
                 or positive(r"对(?:保险)?事故(?:的)?发生[^。；;\n]{0,8}严重影响"),
                 "disclosure_causation_missing", "重大过失未告知导致不赔，还须严重影响保险事故发生。")

    if "4.2" in sections and re.search(r"未及时(?:通知|报案|告知)|迟报", text) and re.search(denial, claim.text):
        limits = (
            r"故意|重大过失",
            r"(?:无法|难以)确定的部分[^。；;\n]{0,12}(?:不承担|拒赔|不赔|不予赔付)",
            r"(?:已经|已)及时知道", r"应当及时知道",
            r"不影响[^。；;\n]{0,12}确定[^。；;\n]{0,32}"
            r"(?:除外|(?<![不未])(?:仍(?:需)?|应当|应)承担(?:相应)?(?:保险)?责任)",
        )
        _require(all(positive(p) for p in limits), "notice_exceptions_missing",
                 "迟报不赔只限无法确定部分，须保留已及时或应当及时知道、未影响事故确定的例外。")

    if rule_sections & {"1.4", "9.1"}:
        refund = r"退还[^，。；;\n]{0,12}现金价值|现金价值[^，。；;\n]{0,12}退还"
        after = r"犹豫期(?:届满|结束)?(?:后|外)|超过\s*15\s*(?:日|天)|" + refund
        within = r"犹豫期内|15\s*(?:日|天)内"
        for sentence in re.split(r"[。；;\n]", claim.text):
            if not re.search(after, sentence):
                continue
            for fee in re.finditer(r"工本费|手续费|相关费用", sentence):
                prefix = sentence[:fee.start()]
                periods = list(re.finditer(after + "|" + within, prefix))
                inside = bool(periods and re.fullmatch(within, periods[-1].group()))
                negated = re.search(r"(?:不|未|无需|无须)(?:会|再|额外|另行)*"
                                    r"(?:扣除|扣取|收取|加收|支付|承担)(?:任何|额外的)?$", prefix)
                _require(inside or negated, "unsupported_surrender_fee",
                         "犹豫期后的现金价值退保不能追加无原文依据的工本费、手续费或相关费用。")
            _require(not re.search(refund, sentence) or "9.1" in rule_sections, "surrender_source_mismatch",
                     "犹豫期后退还现金价值须引用9.1，不能从1.4的犹豫期规则反推。")


def _citations(draft, pack):
    """原文文件已由证据构建器校验；这里核对本轮映射并生成链接，不采用模型链接。"""
    spans = {s["span_id"]: s for s in pack.source_spans}
    _require(len(spans) == len(pack.source_spans), "invalid_citation_source", "原文位置标识重复。")
    evidence = {item.evidence_id: item for item in pack.evidence_items}
    allowed = {"terms"} if pack.source_policy == "terms_only" else {
        "terms", "product_description", "cash_value_table", "sum_assured_table"}
    titles = {"terms": "保险条款", "product_description": "产品说明书",
              "cash_value_table": "现金价值表", "sum_assured_table": "基本保险金额表"}
    citations, by_unit, by_evidence = [], {}, {}
    for eid in dict.fromkeys(eid for claim in draft.claims for eid in claim.evidence_ids):
        item = evidence[eid]
        _require(len(item.unit_ids) == 1 and bool(item.source_span_ids),
                 "invalid_citation_source", "证据缺少可核对的原文单元。")
        _unique(item.source_span_ids)
        _require(set(item.source_span_ids) <= spans.keys(), "invalid_citation_source", "引用缺少原文位置。")
        selected = [spans[sid] for sid in item.source_span_ids]
        first = selected[0]
        identity = (first["document_id"], first["document_sha256"], first["source_type"], first["source_url"])
        _require(first["source_type"] in allowed and re.fullmatch(r"[A-Za-z0-9_-]+", first["document_id"])
                 and re.fullmatch(r"[a-f0-9]{64}", first["document_sha256"])
                 and urlsplit(first["source_url"]).scheme in {"http", "https"}
                 and bool(urlsplit(first["source_url"]).netloc), "invalid_citation_source", "引用来源或版本无效。")
        for span in selected:
            same_document = (span["document_id"], span["document_sha256"],
                             span["source_type"], span["source_url"]) == identity
            valid_span = (same_document and type(span["pdf_page"]) is int and span["pdf_page"] > 0
                          and type(span["start_char"]) is int and type(span["end_char"]) is int
                          and 0 <= span["start_char"] < span["end_char"]
                          and span["end_char"] - span["start_char"] == len(span["exact"]))
            _require(valid_span, "invalid_citation_source", "引用的页码、字符位置或文档版本不一致。")
        _require(item.text == "\n".join(s["exact"] for s in selected),
                 "invalid_citation_source", "证据正文与原文片段不一致。")
        positions = [(s["pdf_page"], s["start_char"], s["end_char"]) for s in selected]
        _require(positions == sorted(positions), "invalid_citation_source", "引用原文顺序与页内位置不一致。")
        uid = item.unit_ids[0]
        signature = (tuple(item.source_span_ids), item.section, item.text)
        if uid in by_unit:
            cid, previous = by_unit[uid]
            _require(signature == previous, "invalid_citation_source", "同一原文单元出现不一致的引用。")
        else:
            cid = f"C{len(citations) + 1}"
            pages = sorted({s["pdf_page"] for s in selected})
            citations.append(Citation(citation_id=cid, document_id=first["document_id"],
                document_title=titles[first["source_type"]], section=item.section, pdf_pages=pages,
                source_url=first["source_url"],
                local_url=f"/sources/{first['document_id']}/{first['document_sha256']}.pdf#page={pages[0]}",
                quote=item.text))
            by_unit[uid] = cid, signature
        by_evidence[eid] = cid
    return citations, by_evidence


def compile_answer(draft, plan, pack, *, semantic_checked=False):
    """无模型调用。仅独立语义复核通过后交付正文；其结果仍不是人工正确率证明。"""
    _require(type(semantic_checked) is bool, "invalid_answer_structure", "语义检查状态无效。")
    _require(pack.source_policy == plan.source_policy, "answer_scope_mismatch", "答案与问题来源范围不一致。")
    _require(pack.execution_status in {"ok", "degraded"}, "invalid_evidence_pack", "证据包尚不可用于回答。")
    validate_citation_references(draft, pack)
    aspect_ids = _unique([a.aspect_id for a in plan.aspects])
    objectives = {a.aspect_id: a.objective for a in plan.aspects}
    amount_ids = {a.aspect_id for a in plan.aspects if a.task == "amount_query"}
    claim_ids = _unique([c.claim_id for c in draft.claims])
    claims = {c.claim_id: c for c in draft.claims}
    evidence = {e.evidence_id: e for e in pack.evidence_items}
    results = {a.aspect_id: a for a in draft.aspect_results}
    _require(_unique([a.aspect_id for a in draft.aspect_results]) == aspect_ids,
             "answer_aspect_mismatch", "答案必须逐一回应本轮的所有问题方面。")
    mentions = {m.mention_id: m for m in plan.mentions}
    _unique([m.mention_id for m in plan.mentions])
    _require(not draft.clarifications and all(a.disposition != "needs_user_info" for a in draft.aspect_results),
             "clarification_disabled", "必要追问尚未接入，不能生成不可执行的追问。")
    gaps, aspect_results = [], []
    for aspect in draft.aspect_results:
        referenced = _unique(aspect.claim_ids)
        _require(referenced <= claim_ids, "unknown_claim", "方面引用了不存在的陈述。")
        _unique(aspect.gap_ids)
        if aspect.disposition == "addressed":
            _require(bool(referenced) and not aspect.gap_ids,
                     "invalid_aspect_disposition", "已回应的方面必须有正文且没有未解决缺口。")
        if aspect.disposition != "addressed" or aspect.aspect_id in amount_ids:
            code = "gap_unverified_" + aspect.aspect_id
            if aspect.disposition != "addressed":
                _require(aspect.gap_ids == [code], "unknown_answer_gap", "未完成方面须使用本轮允许的缺口标识。")
            message = ("当前未提供个人精确金额计算；可说明条款规则，但不能据此给出个人金额。"
                       if aspect.aspect_id in amount_ids else f"本次未能为“{objectives[aspect.aspect_id]}”形成充分依据。")
            gaps.append(Gap(code=code, message=message, aspect_ids=[aspect.aspect_id]))
        row = aspect.model_dump()
        if aspect.aspect_id in amount_ids:
            row.update(disposition="unavailable", gap_ids=["gap_unverified_" + aspect.aspect_id])
        aspect_results.append(row)
    for claim in draft.claims:
        aids = _unique(claim.aspect_ids)
        _require(aids <= aspect_ids, "answer_aspect_mismatch", "陈述引用了未知问题方面。")
        _require(aids == {aid for aid, result in results.items() if claim.claim_id in result.claim_ids},
                 "answer_aspect_mismatch", "陈述与问题方面的关联不一致。")
        _unique(claim.evidence_ids)
        _unique(claim.depends_on_claim_ids)
        premise_ids = _unique(claim.premise_mention_ids)
        _require(premise_ids <= mentions.keys(), "unknown_premise", "答案使用了本轮未提供的用户条件。")
        _require((claim.kind == "rule" and not premise_ids)
                 or (claim.kind == "conditional_application" and bool(premise_ids)),
                 "invalid_premise", "规则说明不能应用个人前提，个案解释必须关联用户原话。")
        _require(all(mentions[mid].modality in {"asserted", "hypothetical"} for mid in premise_ids),
                 "unconfirmed_premise", "转述或含糊信息不能作为个案判断前提。")
        _require(all(not mentions[mid].aspect_ids or aids.intersection(mentions[mid].aspect_ids) for mid in premise_ids),
                 "invalid_premise", "用户条件与陈述的问题方面不一致。")
        _require(bool(claim.text.strip()) and not re.search(r"https?://|www\.|\[[^\]]+\]|<[^>]+>|第\s*\d+\s*页", claim.text),
                 "model_generated_citation", "答案正文不能自造链接、页码或引用标记。")
        _require(not re.search(r"(?:条款|合同|资料|证据)(?:正文|本身|当中|中|里|内|上)?\s*[，,:：]?\s*"
                               r"(?:并|也|均|都|尚|仍|暂)?(?:未曾|未|没有|没|无)"
                               r"(?:明确|具体|直接|进一步|相关|任何|此类|相应)*"
                               r"(?:规定|约定|提供|说明|提及|记载|载明|写明|给出|列示|列明|列出)", claim.text),
                 "unsupported_absence_claim", "本轮证据不足不能证明条款全文未规定相关内容。")
        if plan.product_id == "sshy_s_hengying":
            _product_guards(claim, claims, evidence)
    try:
        citations, mapping = _citations(draft, pack)
    except (KeyError, TypeError, ValueError) as exc:
        raise AppError("invalid_citation_source", "引用原文元数据不完整或格式无效。", 422) from exc
    ordered, visited = [], set()

    def visit(cid):
        if cid not in visited:
            visited.add(cid)
            for dependency in claims[cid].depends_on_claim_ids:
                visit(dependency)
            ordered.append(claims[cid])

    for cid in claims:
        visit(cid)
    complete = semantic_checked and all(a["disposition"] == "addressed" for a in aspect_results)
    if not semantic_checked and draft.claims:
        gaps.append(Gap(code="semantic_review_required", message="答案尚未完成原文支持核对。",
                        aspect_ids=[a.aspect_id for a in plan.aspects]))

    blocks = []
    coverage = "none"
    semantic_status = "not_run"
    if semantic_checked:
        for claim in ordered:
            citation_ids = list(dict.fromkeys(mapping[eid] for eid in claim.evidence_ids))
            blocks.append(AnswerBlock(claim_id=claim.claim_id, text=claim.text, citation_ids=citation_ids))
        semantic_status = "model_checked"
        if complete:
            coverage = "complete"
        elif ordered:
            coverage = "partial"

    return AnswerResult(answer_status="answered" if complete else "insufficient_evidence",
        execution_status="degraded" if not semantic_checked or pack.execution_status == "degraded" else "ok",
        coverage_status=coverage, answer_blocks=blocks,
        citations=citations if semantic_checked else [], gaps=gaps,
        aspect_results=[{**a, "semantic_status": semantic_status} for a in aspect_results],
        validation={"structural": "passed", "source": "pack_checked", "dependency": "passed",
                    "domain": "capability_and_premise_checked", "semantic": semantic_status},
        trace={"pack_id": pack.pack_id, "validation_scope": "trusted_pack_spans_and_references",
               "semantic_review_is_proof": False}, versions={**pack.versions, "compiler": "answer-compiler/1.0"})
