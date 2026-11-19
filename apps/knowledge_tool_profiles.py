"""Human/agent usage descriptions read from the existing Knowledge database."""

import argparse
import json
from pathlib import Path

from apps import knowledge_literature_tools as literature
from apps import knowledge_source_tools as pages
from db.knowledge.repository import KnowledgeRepositoryError
from packages.knowledge_connectors.usage import seal_profile

PYTHON = "/opt/data/test/envs/MMAgent1/bin/python -m "
LIT = PYTHON + "apps.knowledge_literature_tools "
PAGE = PYTHON + "apps.knowledge_source_tools "

# Project-specific usage, not an assertion of the remote institution's full API.
DETAILS = {
    "nice": {
        "name": "NICE NG81青光眼指南",
        "purpose": "先确定检查项目、测量方法、转诊条件和随访框架。",
        "best_for": ["检查树的主方向及眼压/视野/视神经头检查", "核对NG81条款原文及适用条件"],
        "not_for": ["任意疾病全站搜索", "把转诊阈值当患病概率"],
        "inputs": ["固定NG81推荐页；glaucoma精选目录关键词"],
        "outputs": ["NG81目录链接", "完整网页body.bin及响应摘要，不是自动医学结论"],
        "examples": [PYTHON + "apps.glaucoma_knowledge_intake", PAGE + "fetch nice"],
        "limitations": [
            "当前为NG81固定入口，不是NICE全站搜索；24mmHg是特定测量/转诊条件，不是排除青光眼的界限"
        ],
    },
    "aao": {
        "name": "AAO青光眼PPP官方指南",
        "purpose": "补齐临床体征、鉴别解释、结构/功能对应及进展检查。",
        "best_for": [
            "视盘、盘沿、RNFL、盘周萎缩和血管位置等临床检查定义",
            "POAG、疑似病例、房角关闭三个主题的完整指南阅读",
        ],
        "not_for": ["AAO全站自由搜索", "将指南文字直接变成本例已观察的结果"],
        "inputs": [
            "AAO_POAG_PPP_2026、AAO_POAG_SUSPECT_PPP_2026或AAO_ANGLE_CLOSURE_PPP_2026",
            "PDF物理起始页1—200，每次1—10页，1—16000字符",
        ],
        "outputs": ["官方PDF摘要、总页数、实际读取页码和正文excerpt", "目录命中，非全站远程排序"],
        "examples": [
            LIT + "search --providers aao --topic glaucoma",
            LIT + "fetch aao --record-id AAO_POAG_PPP_2026 "
            "--page-start 24 --page-count 2 --max-chars 12000",
        ],
        "limitations": [
            "2025批准版对应2026期刊版；从AAO官网公开PDF读取，期刊网页可能403",
            "文本预览不自动解析表格、OCR或全文建索引；本地已有说明不保证远程永远可达",
        ],
    },
    "egs": {
        "name": "EGS青光眼教育内容",
        "purpose": "解释视盘/盘沿/RNFL/OCT及出血现象，并检索已归档原文。",
        "best_for": ["盘大小对杯盘比的影响", "盘周出血的其他解释", "现有四篇原文的本地片段搜索"],
        "not_for": ["把Tip教育例子当普适诊断公式", "EGS所有指南/全站内容的穷尽搜索"],
        "inputs": ["固定视盘大小网页", "已入库四篇内容的query，例如hemorrhage、glaucoma"],
        "outputs": ["网页正文", "精确SourceVersion/Chunk引用、原文位置和检索分量"],
        "examples": [
            PAGE + "fetch egs",
            PYTHON + "apps.glaucoma_egs_content search --query hemorrhage",
        ],
        "limitations": [
            "已有四篇索引共98个片段，不代表整个EGS网站；本地检索与新联网取文分开",
            "盘沿比例需核对径向定义、区域、盘大小，不能代入包围盒间距",
        ],
    },
    "dalhousie": {
        "name": "Dalhousie近视与青光眼研究栏目",
        "purpose": "了解近视如何影响视神经结构、影像解释及研究方向。",
        "best_for": ["近视相关混淆因素与研究线索", "眼科研究栏目内的有限关键词追踪"],
        "not_for": ["大学全站搜索引擎", "以项目介绍代替同行评议实验结果"],
        "inputs": ["固定myopia项目页", "glaucoma、myopia、oct、rnfl等公开关键词与max-pages上限"],
        "outputs": ["项目网页正文", "限定栏目内匹配页面及摘录"],
        "examples": [
            PAGE + "fetch dalhousie",
            PAGE + "search dalhousie --query myopia --max-pages 6",
        ],
        "limitations": ["仅沿允许的眼科栏目链接抓取，不是任意网页爬虫；返回项目方向需继续追溯论文"],
    },
    "pubmed": {
        "name": "PubMed医学文献检索",
        "purpose": "广泛发现某项体征、数值、诊断效能或方法的原始医学研究。",
        "best_for": [
            "杯盘比、RNFL、盘出血、近视、视野等主题研究",
            "按PMID读取摘要并确认研究人群、模态和终点",
        ],
        "not_for": ["保证每篇论文都能取到期刊全文", "把搜索排名当证据强度或直接生成病例结论"],
        "inputs": ["topics列出的22个公开主题；每次limit 1—20", "fetch使用纯数字PMID"],
        "outputs": ["标题、PMID/DOI、作者、日期等检索命中", "PubMed XML摘要记录及有界摘录"],
        "examples": [
            LIT + "search --providers pubmed --topic cup_disc_ratio --limit 10",
            LIT + "fetch pubmed --record-id 42744592 --max-chars 8000",
        ],
        "limitations": [
            "邮箱由MMAGENT_NCBI_EMAIL或本机受限配置读取，详情中不包含邮箱值",
            "partial/page_limit是本次预算截断；摘要不等于全文",
        ],
    },
    "europepmc": {
        "name": "Europe PMC文献与开放全文",
        "purpose": "发现研究并优先取得可机器读取的开放论文全文。",
        "best_for": ["补充PubMed结果", "按PMCID查完整方法、适用域、公式及结果"],
        "not_for": ["访问无开放许可的任意论文全文", "只凭摘要数值外推本例阈值"],
        "inputs": ["22个公开主题及limit 1—20", "开放全文使用PMC开头的PMCID，不是PMID"],
        "outputs": ["文献命中及PMCID/PMID/DOI", "开放全文XML及文字摘录"],
        "examples": [
            LIT + "search --providers europepmc --topic vessels --limit 10",
            LIT + "fetch europepmc --record-id PMC11754513 --max-chars 8000",
        ],
        "limitations": [
            "只有支持开放全文的记录可取全文；部分返回书目/摘要",
            "全文下载和结构化归档、建索引是不同操作",
        ],
    },
    "crossref": {
        "name": "Crossref论文元数据与DOI",
        "purpose": "扩展发现论文并核对出版身份、年份、作者及原文链接。",
        "best_for": ["查DOI和出版信息", "与PubMed/Europe PMC交叉匹配文献身份"],
        "not_for": ["保证全文下载", "把图表、章节或相似题名都当独立医学研究"],
        "inputs": ["22个公开主题及limit 1—20", "fetch使用完整DOI字符串"],
        "outputs": ["按相关性排序的书目记录", "DOI元数据JSON，可能含摘要或原文链接"],
        "examples": [
            LIT + "search --providers crossref --topic rim --limit 10",
            LIT + "fetch crossref --record-id 10.1002/14651858.CD008803.pub2",
        ],
        "limitations": ["不是全文库；相关性排序不能代表质量", "同DOI可归并，不凭相似标题强行合并"],
    },
    "cochrane": {
        "name": "Cochrane公开证据摘要",
        "purpose": "核对影像检查诊断效能的系统综述结论及限制。",
        "best_for": ["视神经影像诊断青光眼的总体证据", "帮助避免把单篇实验结果当普遍效果"],
        "not_for": ["任意Cochrane全站关键词搜索", "绕过订阅限制或把综述摘要当完整付费正文"],
        "inputs": ["当前精选记录COCHRANE_CD008803"],
        "outputs": ["官方公开综述摘要、页面来源和摘要字节绑定"],
        "examples": [LIT + "fetch cochrane --record-id COCHRANE_CD008803 --max-chars 8000"],
        "limitations": ["当前只安装一个精选综述，不声称覆盖整个数据库；综述时间范围需独立核对"],
    },
    "nei": {
        "name": "NEI青光眼公众教育网页",
        "purpose": "解释疾病概念和一般风险背景。",
        "best_for": ["患者易读介绍", "年龄、家族史等一般背景线索"],
        "not_for": ["精确图像阈值或诊断算法依据", "医学论文检索"],
        "inputs": ["固定glaucoma教育页面"],
        "outputs": ["公众教育网页正文；网络失败时返回失败类别"],
        "examples": [PAGE + "fetch nei"],
        "limitations": [
            "2026-09-18最近取页失败；登记存在不代表当前网络可用",
            "公众教育年龄界限是风险背景，不是疾病判定阈值",
        ],
    },
}


def provider(key):
    return (
        literature.provider_record(literature.provider_config(key))
        if key in literature.PROVIDERS
        else pages.record(key)
    )


def profile(key):
    record = provider(key)
    reference = record.reference().model_dump(mode="json")
    reference.pop("kind")
    details = DETAILS[key]
    return seal_profile(
        key=key,
        provider_ref=reference,
        **details,
        workflow=[
            "先读取本详情并选择适合的来源",
            "使用示例命令发现/获取原文",
            "核对人群、模态、测量方法和终点，再关联证据需求；文献相关不等于患者结果",
        ],
        prerequisites=[
            "在/opt/data/test/MMAgent目录使用ENV-001解释器",
            "本地Knowledge数据库服务及来源登记可用",
            "联网来源需要既有网络路径；PubMed、AAO、NICE和文献来源按原入口使用HTTPS_PROXY",
        ],
        failure_actions=[
            "零命中、未取全文和网络失败分开记录",
            "403不伪装成无医学证据；尝试已允许的其他来源",
            "按精确引用读回，撤回/禁用不因详情存在而恢复调用",
        ],
    )


def read(repo, key):
    expected = profile(key)
    value = repo.read_tool_usage(expected.provider_ref)
    if value != expected:
        raise ValueError("installed_tool_usage_differs_from_config")
    return {
        "profile": value.model_dump(mode="json"),
        "database_readback": True,
        "provider_state": "not_checked_by_description_read",
        "network_status": "not_probed_by_description_read",
    }


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "list", "show", "configured"))
    parser.add_argument("key", nargs="?", choices=tuple(DETAILS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.action == "show" and not args.key:
        parser.error("show requires a tool key")
    if args.output and (args.output.exists() or not args.output.parent.is_dir()):
        parser.error("output must be new in an existing directory")
    if args.action == "configured":
        result = [profile(k).model_dump(mode="json") for k in DETAILS]
    else:
        repo = pages.repository("editor" if args.action == "register" else "reader")
        if args.action == "register":
            for key in DETAILS:
                repo.put_tool_usage(profile(key))
        result = [read(repo, k) for k in ([args.key] if args.key else DETAILS)]
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with args.output.open("x", encoding="utf-8") as target:
            target.write(text + "\n")
        print(json.dumps({"output": str(args.output), "count": len(result)}))
    else:
        print(text)


def main(argv=None):
    try:
        _main(argv)
    except (KnowledgeRepositoryError, ValueError) as error:
        print(json.dumps({"status": "error", "category": type(error).__name__}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
