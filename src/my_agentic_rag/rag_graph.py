"""
============================================================
代码故事（从程序启动到回答用户问题的完整过程）
============================================================
用户提问后，【rag_graph.py】启动，首先进入【retrieve_initial()】节点...
这个节点执行三路检索：
  1. 原始查询直接检索
  2. HyDE：生成假设性文档 → 用陈述句检索
  3. Step-back：生成抽象退步问题 → 用更通用的问题检索
三路结果合并去重后传给【grade_documents()】节点...
这个节点用 LLM 判断每个文档是否与问题相关...
根据评分结果决定路由：
  - 如果有相关文档 → 进入【generate()】节点 → 结束
  - 如果没有相关文档 → 进入【rewrite_question()】节点 → 【retrieve_expanded()】→ 结束
============================================================
"""

"""
============================================================
面试版：Agentic RAG 的核心流程（v2.0 多路检索增强）
============================================================
Agentic RAG 使用 LangGraph 实现条件路由...
核心是【grade_documents()】节点，它用 LLM 评分...
评分结果决定下一步：
  - 评分通过 → 直接生成答案
  - 评分不通过 → 重写问题 + 扩展检索

v2.0 新增：
  - HyDE（假设性文档）：将疑问句转为陈述句检索，解决向量空间距离问题
  - Step-back（退步问题）：用抽象问题兜底检索，提升覆盖面
  - 三路检索独立执行，结果合并去重，各不污染
v2.1 新增：
  - 语义校验：改写后向量相似度回检，改歪丢弃（阈值 0.8）
v2.2 新增：
  - 结构化评分：Pydantic GradeResult 替代字符串匹配，杜绝 "不相关"误判为"相关"
v2.3 新增：
  - 子查询拆解：复杂问题（Vite+Vue+Pinia）自动拆成 2-3 个子问题各自检索
v3.0 新增（GPT 评审驱动）：
  - 文档相关性由 LLM 语义判断（不是靠向量相似度阈值）
  - 失败分类路由：区分模糊/缺知识/多跳/检索失败四种原因，按需改写给拒答
  - 证据锚定生成：每项声明须标注文档编号，禁止从元数据推断事实关系
============================================================
"""

from typing import TypedDict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic


def get_response_text(response: Union[str, Any]) -> str:
    """从 LLM 响应中提取文本内容"""
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list) and len(content) > 0:
            # 处理 content 是列表的情况（如 DeepSeek 返回的 thinking + text）
            texts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        texts.append(item.get("text", ""))
                    elif "text" in item:
                        texts.append(item["text"])
                elif hasattr(item, "text"):
                    texts.append(item.text)
            if texts:
                return "\n".join(texts)
    if isinstance(response, list) and len(response) > 0:
        # 处理响应本身就是列表的情况
        texts = []
        for item in response:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif "text" in item:
                    texts.append(item["text"])
            elif hasattr(item, "text"):
                texts.append(item.text)
        if texts:
            return "\n".join(texts)
    return str(response)

from . import config
from .vectorstore import get_vectorstore
from .embedding import get_embedding_service
from .prompts import (
    GENERATE_PROMPT,
    REWRITE_QUERY_PROMPT,
    GRADE_DOCUMENTS_PROMPT,
    HYDE_PROMPT,
    STEP_BACK_QUESTION_PROMPT,
    DECOMPOSE_QUERY_PROMPT,
    CLASSIFY_FAILURE_PROMPT,
    GROUNDED_GENERATE_PROMPT,
)


# ================================================================
# RAG 状态定义
# ================================================================

class RAGState(TypedDict):
    """
    RAG 流程状态类

    用于在整个 RAG 流程中传递和共享数据。
    每个节点都可以读取和写入这些字段。
    """
    question: str              # 用户原始问题
    query: str                 # 当前使用的查询（可能是重写后的）
    context: str               # 拼接后的上下文（用于 LLM 生成答案）
    docs: List[dict]           # 检索到的所有文档列表
    relevant_docs: List[dict] # 经过 LLM 评分判定为相关的文档
    route: str                 # 路由决定，决定下一步走哪条分支
    answer: str                # 最终生成的答案
    baseline: bool             # True=仅路径1原始检索，False=四路全开


# ================================================================
# LLM 模型
# ================================================================

def get_llm():
    """
    获取 LLM 模型实例（单例模式）

    Returns:
        ChatAnthropic: 配置好的 DeepSeek API 客户端

    说明：
        - 使用单例模式避免重复创建模型实例
        - 配置 temperature=0.3，平衡创造性和确定性
        - base_url 指向 DeepSeek 的 Anthropic 兼容接口
    """
    return ChatAnthropic(
        model=config.MODEL_ID,
        api_key=config.ANTHROPIC_API_KEY,
        base_url=config.ANTHROPIC_BASE_URL,
        temperature=0.3,
        # DeepSeek 的 thinking 内容也占输出 token；默认 1024 会被 thinking 吃光，
        # 导致正文还没开始就被截断。给足空间。
        max_tokens=8000,
    )


# ================================================================
# 多路检索辅助函数
# ================================================================

def _generate_hypothetical_document(llm, question: str) -> str:
    """
    生成假设性文档（HyDE）

    将疑问句转为陈述句风格的"伪文档"，缩小问题和文档在向量空间的距离。

    Args:
        llm: LLM 实例
        question: 用户原始问题

    Returns:
        str: 假设性文档文本，失败返回空字符串
    """
    prompt = HYDE_PROMPT.format(question=question)
    try:
        response = llm.invoke(prompt)
        text = get_response_text(response).strip()
        return text if text else ""
    except Exception:
        return ""


def _generate_step_back_question(llm, question: str) -> str:
    """
    生成退步问题（Step-back）

    将具体问题抽象为更通用的概念性问题，用于兜底检索。

    Args:
        llm: LLM 实例
        question: 用户原始问题

    Returns:
        str: 抽象退步问题，失败返回空字符串
    """
    prompt = STEP_BACK_QUESTION_PROMPT.format(question=question)
    try:
        response = llm.invoke(prompt)
        text = get_response_text(response).strip()
        return text if text else ""
    except Exception:
        return ""


def _decompose_question(llm, question: str) -> list:
    """
    复杂问题拆解：判断是否需要拆，需要则拆成 2-3 个子问题

    例如：
      "Vite + Vue 3 + Pinia 搭建项目"
      → ["Vite 项目配置步骤", "Vue 3 组件写法规范", "Pinia store 定义方式"]

    如果 LLM 判断问题足够简单（单一主题，一次检索能搞定），返回空列表。
    如果 LLM 调用失败，也返回空列表（不阻塞主流程）。

    Args:
        llm: LLM 实例
        question: 用户原始问题

    Returns:
        list: 子问题列表，简单问题或无结果时为空
    """
    prompt = DECOMPOSE_QUERY_PROMPT.format(question=question)
    try:
        response = llm.invoke(prompt)
        text = get_response_text(response).strip()

        # LLM 返回 "SIMPLE" 表示不需要拆解
        if "SIMPLE" in text.upper():
            return []

        # 提取子问题（格式：子问题1：...）
        sub_questions = []
        for line in text.split("\n"):
            line = line.strip()
            if line and ("子问题" in line or "：" in line or ":" in line):
                # 去掉编号前缀
                for prefix in ["子问题1：", "子问题2：", "子问题3：",
                               "子问题1:", "子问题2:", "子问题3:",
                               "1.", "2.", "3.", "1、", "2、", "3、"]:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break
                if line and len(line) > 2:
                    sub_questions.append(line)

        return sub_questions[:3]  # 最多 3 个子问题
    except Exception:
        return []


def _merge_dedupe_docs(docs_list: list) -> list:
    """
    合并且去重多路检索结果

    去重逻辑：取每个文档内容的前 200 字符作为指纹，
    相同指纹的文档视为重复，保留分数更高的版本。
    合并后按分数降序排列。

    Args:
        docs_list: 多路检索结果的列表，每个元素是一组文档列表

    Returns:
        list: 去重合并后的文档列表
    """
    seen: dict = {}
    for docs in docs_list:
        for doc in docs:
            # 用内容前 200 字符作为去重指纹
            sig = doc.get("content", "")[:200]
            if sig in seen:
                # 保留分数更高的版本
                if doc.get("score", 0) > seen[sig].get("score", 0):
                    seen[sig] = doc
            else:
                seen[sig] = doc

    merged = sorted(seen.values(), key=lambda d: d.get("score", 0), reverse=True)
    return merged


def _validate_semantic_similarity(original: str, rewritten: str) -> bool:
    """
    语义校验：改写后的文本和原问题相似度是否达标

    原理：
      - 用同一个 BGE 模型把原始问题和改写文本都向量化
      - BGE 的 normalize_embeddings=True，向量已是 L2 归一化的
      - 归一化向量的余弦相似度 = 点积（dot product）
      - 相似度 < 阈值 → 改写"歪了"，丢弃

    为什么重要：
      视频 1 专门讲了"改写模型一旦幻觉，会把'怎么退款'改成'怎么赚钱'，
      南辕北辙"。这个校验就是那个"黄色盾牌"。

    Args:
        original: 原始问题
        rewritten: 改写后的文本（HyDE 文档或 Step-back 问题）

    Returns:
        bool: True=通过校验，False=改歪了丢弃

    线上落地注意：
      - 阈值 0.8 是经验值，需要根据实际数据用 RAGAS 调优
      - 阈值太高 → 很多有效改写被误杀（召回率下降）
      - 阈值太低 → 噪声混进来（准确率下降）
    """
    threshold = config.SEMANTIC_VALIDATION_THRESHOLD
    try:
        embedding_service = get_embedding_service()
        # 获取原始问题和改写文本的稠密向量
        orig_vec = embedding_service.get_dense_embedding(original)
        rewr_vec = embedding_service.get_dense_embedding(rewritten)
        # 余弦相似度 = 归一化向量的点积
        similarity = sum(a * b for a, b in zip(orig_vec, rewr_vec))
        return similarity >= threshold
    except Exception:
        # 校验失败时不阻塞主流程，保守策略：通过
        return True


# ================================================================
# 结构化评分模型（v2.2：替代字符串匹配）
# ================================================================

class GradeResult(BaseModel):
    """
    文档相关性评分结果

    为什么用结构化输出而不是字符串匹配？
      - 字符串匹配 `"相关" in grade` 极其脆弱：
        模型返回"该文档与问题相关" → 命中
        模型返回"文档内容不相关" → 也命中！因为"不相关"里包含"相关"二字
      - Pydantic 强制 LLM 输出 JSON → 解析不会出错
      - reason 字段留作调试日志，后续可以分析评分质量
    """
    relevant: bool = Field(description="文档是否包含有助于回答用户问题的信息")
    reason: str = Field(default="", description="简短理由（用于日志和调试）")


def _grade_single_document(llm, question: str, doc_content: str) -> bool:
    """
    用结构化输出评判单个文档是否与问题相关

    流程：
      1. 构建评分提示词
      2. llm.with_structured_output(GradeResult) 强制返回 JSON
      3. 解析 GradeResult.relevant 字段

    为什么逐个文档评分而不是批量？
      - 批量评分时 LLM 容易混淆文档之间的边界
      - 逐个评分的 token 消耗略高，但准确率更稳定
      - 如果文档数量很大（>10），可改为分批

    Args:
        llm: LLM 实例
        question: 用户问题
        doc_content: 单个文档的内容

    Returns:
        bool: True=相关，False=不相关
    """
    prompt = GRADE_DOCUMENTS_PROMPT.format(question=question, document=doc_content)

    # DeepSeek thinking 模式不支持 with_structured_output 的 tool_choice，
    # 直接让它回答"相关/不相关"，避免每个文档先发一次注定失败的请求。
    if "deepseek" in config.MODEL_ID.lower():
        try:
            response = llm.invoke(prompt)
            grade = get_response_text(response).strip()
            return "相关" in grade and "不相关" not in grade
        except Exception:
            return False

    try:
        structured_llm = llm.with_structured_output(GradeResult)
        result: GradeResult = structured_llm.invoke(prompt)
        return result.relevant
    except Exception:
        # 结构化输出解析失败时，降级到字符串匹配（向后兼容）
        response = llm.invoke(prompt)
        grade = get_response_text(response).strip().lower()
        return "相关" in grade and "不相关" not in grade


# ================================================================
# RAG 图节点
# ================================================================

def retrieve_initial(state: RAGState) -> RAGState:
    """
    节点1：多路检索（v2.0 增强）

    执行四路独立检索，然后合并去重：
      路径1：原始查询直接检索
      路径2：HyDE — 生成假设性文档 → 用陈述句检索
      路径3：Step-back — 生成抽象退步问题 → 用更通用词检索
      路径4：子查询拆解 — 复杂问题拆成 2-3 个子问题各自检索

    四路检索各自独立，互不污染。任何一路失败自动跳过，
    不会影响其他路继续执行。

    Args:
        state: 包含用户问题的 RAG 状态

    Returns:
        更新后的 state，包含合并去重后的文档列表
    """
    question = state["question"]
    print(f"\n[Step 1] 多路检索: {question[:50]}...")

    llm = get_llm()
    vectorstore = get_vectorstore()
    all_docs: list = []

    # ----------------------------------------------------------
    # 路径1：原始查询检索
    # ----------------------------------------------------------
    print("  [1/4] 原始查询检索...")
    docs_original = vectorstore.retrieve(question, top_k=config.RETRIEVAL_TOP_K)
    all_docs.append(docs_original)
    print(f"         检索到 {len(docs_original)} 个文档")

    # Baseline 模式：只走路径1，跳过增强策略
    if state.get("baseline"):
        merged = _merge_dedupe_docs(all_docs)
        state["query"] = question
        state["docs"] = merged
        print(f"[Step 1] Baseline 模式，跳过增强检索，共 {len(merged)} 个文档")
        return state

    # ----------------------------------------------------------
    # 路径2：HyDE 假设性文档检索（v2.1：加语义校验）
    # ----------------------------------------------------------
    print("  [2/4] HyDE 假设性文档检索...")
    hyde_doc = _generate_hypothetical_document(llm, question)
    if hyde_doc and _validate_semantic_similarity(question, hyde_doc):
        print(f"         假设性文档: {hyde_doc[:60]}...")
        print(f"         语义校验通过 ✓")
        docs_hyde = vectorstore.retrieve(hyde_doc, top_k=config.RETRIEVAL_TOP_K)
        all_docs.append(docs_hyde)
        print(f"         检索到 {len(docs_hyde)} 个文档")
    elif hyde_doc:
        print("         语义校验未通过 ✗（改歪了，丢弃）")
    else:
        print("         生成失败，跳过此路")

    # ----------------------------------------------------------
    # 路径3：Step-back 退步问题检索（v2.1：加语义校验）
    # ----------------------------------------------------------
    print("  [3/4] Step-back 退步问题检索...")
    step_back_q = _generate_step_back_question(llm, question)
    if step_back_q and step_back_q != question and _validate_semantic_similarity(question, step_back_q):
        print(f"         退步问题: {step_back_q[:60]}...")
        print(f"         语义校验通过 ✓")
        docs_stepback = vectorstore.retrieve(step_back_q, top_k=config.RETRIEVAL_TOP_K)
        all_docs.append(docs_stepback)
        print(f"         检索到 {len(docs_stepback)} 个文档")
    elif step_back_q and step_back_q != question:
        print("         语义校验未通过 ✗（改歪了，丢弃）")
    else:
        print("         生成失败或与原问题相同，跳过此路")

    # ----------------------------------------------------------
    # 路径4：子查询拆解检索（v2.3：复杂问题自动拆解）
    # ----------------------------------------------------------
    print("  [4/4] 子查询拆解检索...")
    sub_questions = _decompose_question(llm, question)
    if sub_questions:
        print(f"         拆解为 {len(sub_questions)} 个子问题:")
        for i, sq in enumerate(sub_questions, 1):
            print(f"           {i}. {sq[:50]}...")
            docs_sub = vectorstore.retrieve(sq, top_k=max(2, config.RETRIEVAL_TOP_K // 2))
            all_docs.append(docs_sub)
            print(f"             → 检索到 {len(docs_sub)} 个文档")
    else:
        print("         问题较简单，无需拆解")

    # ----------------------------------------------------------
    # 合并去重
    # ----------------------------------------------------------
    merged = _merge_dedupe_docs(all_docs)

    state["query"] = question
    state["docs"] = merged

    print(f"[Step 1] 四路检索合并后共 {len(merged)} 个文档（去重，原始共 {sum(len(d) for d in all_docs)} 个）")
    return state


def grade_documents(state: RAGState) -> RAGState:
    """
    节点2：相关性评分（Agentic RAG 核心）

    Args:
        state: 包含检索文档的 RAG 状态

    Returns:
        更新后的 state，包含相关文档和路由决定

    功能说明：
        1. 遍历每个检索到的文档
        2. 使用 LLM 判断文档是否与用户问题相关
        3. 根据判断结果分为"相关"和"不相关"两组
        4. 根据相关文档数量决定路由：
           - 有相关文档 → 路由到 "generate"（生成答案）
           - 无相关文档 → 路由到 "rewrite"（重写问题）

    为什么是核心：
        - 普通 RAG 使用向量相似度阈值判断（不准确）
        - Agentic RAG 使用 LLM 语义理解判断（更智能）
        - 这是实现"条件路由"的关键节点
    """
    question = state["question"]
    docs = state["docs"]
    llm = get_llm()

    print(f"\n[Step 2] 正在评估 {len(docs)} 个文档的相关性...")

    relevant_docs = []
    llm_calls = 0

    for i, doc in enumerate(docs):
        llm_calls += 1
        is_relevant = _grade_single_document(llm, question, doc["content"])
        if is_relevant:
            relevant_docs.append(doc)
            print(f"  文档 {i+1}: [LLM 判相关]")
        else:
            print(f"  文档 {i+1}: [LLM 判不相关]")

    print(f"[Step 2] LLM 调用: {llm_calls} 次, 相关文档: {len(relevant_docs)}/{len(docs)}")

    state["relevant_docs"] = relevant_docs
    if len(relevant_docs) >= 1:
        state["route"] = "generate"
    else:
        state["route"] = "rewrite"
        print("[Step 2] 全部不相关，触发失败分类路由")

    return state


def rewrite_question(state: RAGState) -> RAGState:
    """
    节点3：失败分类 + 按原因路由（v3.0）

    先判断"为什么第一次检索全部不相关"，再决定怎么做：
      - ambiguous_query → 重写问题
      - missing_knowledge → 诚实拒答（跳过 rewrite）
      - multi_hop_query → 不重写，去 generate 用子查询结果
      - retrieval_failure → 重写 + 扩展检索
    """
    question = state["question"]
    llm = get_llm()

    docs = state.get("docs", [])
    doc_summaries = "\n".join(f"- {d.get('content', '')[:80]}" for d in docs[:5])
    classify_prompt = CLASSIFY_FAILURE_PROMPT.format(question=question, doc_summaries=doc_summaries)

    # 分类失败原因
    try:
        response = llm.invoke(classify_prompt)
        failure_type = get_response_text(response).strip().lower()
    except Exception:
        failure_type = "ambiguous_query"  # 默认当模糊问题处理

    print(f"\n[Step 3] 检索失败分类: {failure_type}")

    if "missing_knowledge" in failure_type:
        # 知识库里没有 → 直接拒答，不浪费 rewrite
        state["query"] = question
        state["route"] = "generate"  # generate 会输出"没有找到相关信息"
        print("[Step 3] 知识库缺失 → 跳过 rewrite，直接拒答")

    elif "multi_hop" in failure_type:
        # 跨文档综合题 → 子查询拆解已经在 retrieve_initial 做了
        # 如果没有拆解（baseline 模式），保持原样去 generate
        state["query"] = question
        state["route"] = "generate"
        print("[Step 3] 跨文档综合题 → 跳过 rewrite")

    else:
        # ambiguous_query 或 retrieval_failure → 重写
        rewrite_prompt = REWRITE_QUERY_PROMPT.format(question=question)
        response = llm.invoke(rewrite_prompt)
        rewritten_query = get_response_text(response).strip()
        state["query"] = rewritten_query
        state["route"] = "rewrite"
        print(f"[Step 3] 重写后: {rewritten_query[:50]}...")

    return state


def retrieve_expanded(state: RAGState) -> RAGState:
    """
    节点4：扩展检索

    Args:
        state: 包含重写后查询的 RAG 状态

    Returns:
        更新后的 state，包含扩展检索到的相关文档

    功能说明：
        1. 使用重写后的问题进行第二次向量检索
        2. 检索结果重新经过 LLM 评分
        3. 保存评分后的相关文档

    与 Step 1 的区别：
        - Step 1 使用用户原始问题检索
        - Step 4 使用 LLM 重写后的问题检索
        - 重写后的问题更清晰，检索结果可能更好
    """
    query = state["query"]
    print(f"\n[Step 4] 使用重写后的问题进行扩展检索...")

    # 使用重写后的问题进行第二次检索
    vectorstore = get_vectorstore()
    docs = vectorstore.retrieve(query, top_k=config.RETRIEVAL_TOP_K)

    state["docs"] = docs

    # 对扩展检索结果重新进行 LLM 结构化评分
    question = state["question"]
    llm = get_llm()

    relevant_docs = []
    for i, doc in enumerate(docs):
        is_relevant = _grade_single_document(llm, question, doc["content"])
        if is_relevant:
            relevant_docs.append(doc)

    state["relevant_docs"] = relevant_docs
    print(f"[Step 4] 扩展检索到 {len(relevant_docs)} 个相关文档")

    return state


def generate(state: RAGState) -> RAGState:
    """
    节点5：证据锚定生成（v3.0）

    关键改进：
      - 每个事实性声明必须标注源于哪个文档编号
      - 禁止从元数据（作者、文件名、来源）推断事实关系
      - 禁止引用文档外知识
      - 没找到充分证据就明确说"未找到足够证据"

    v3.0 背景（GPT 评审反馈）：
      AntFu 那道题：Agentic RAG 搜到了更多文档，但把"Skill 文档作者"
      推理成了"项目贡献者"——这是典型的 Retrieval Amplified Hallucination。
      证据锚定生成通过强制要求每项声明有文档依据来杜绝这种推理。
    """
    question = state["question"]
    relevant_docs = state["relevant_docs"]
    llm = get_llm()

    print(f"\n[Step 5] 生成答案（证据锚定模式）...")

    # 构建上下文：给每个文档编号，LLM 必须在回答里引用编号
    context_parts = []
    for i, doc in enumerate(relevant_docs):
        source = doc["metadata"].get("source", "unknown")
        context_parts.append(f"[文档 {i+1}] (来源: {source})\n{doc['content']}")

    context = "\n\n".join(context_parts)

    # 使用证据锚定提示词（禁止元数据推理 + 要求标注文档编号）
    prompt = GROUNDED_GENERATE_PROMPT.format(context=context, question=question)

    response = llm.invoke(prompt)
    answer = get_response_text(response).strip()

    # 如果上下文中没有文档（拒答场景），使用旧版提示词
    if not relevant_docs:
        prompt = GENERATE_PROMPT.format(context="（无相关文档）", question=question)
        response = llm.invoke(prompt)
        answer = get_response_text(response).strip()

    state["context"] = context
    state["answer"] = answer

    print(f"[Step 5] 答案生成完成 ({len(answer)} 字符)")
    return state


# ================================================================
# 构建 RAG 图
# ================================================================

def build_rag_graph():
    """
    构建 RAG 流程图

    Returns:
        CompiledStateGraph: 编译后的 LangGraph 图

    流程图结构：
        retrieve_initial → grade_documents
                                      ↓
                              [条件路由判断]
                                 ↓       ↓
                            generate   rewrite_question
                                          ↓
                                    retrieve_expanded
                                          ↓
                                          ↓
                                        generate

    说明：
        - 使用 StateGraph 定义状态和节点
        - 使用 add_conditional_edges 实现条件路由
        - 路由规则由 grade_documents 节点的 route 字段决定
    """
    graph = StateGraph(RAGState)

    # 添加 5 个节点到图中
    graph.add_node("retrieve_initial", retrieve_initial)  # 初次检索
    graph.add_node("grade_documents", grade_documents)     # LLM 评分
    graph.add_node("rewrite_question", rewrite_question)   # 查询重写
    graph.add_node("retrieve_expanded", retrieve_expanded) # 扩展检索
    graph.add_node("generate", generate)                   # 生成答案

    # 设置入口节点（第一个执行的节点）
    graph.set_entry_point("retrieve_initial")

    # 添加边：retrieve_initial 之后执行 grade_documents
    graph.add_edge("retrieve_initial", "grade_documents")

    # 添加条件边：根据评分结果决定下一步
    def route_decision(state: RAGState) -> str:
        """
        路由决策函数

        Args:
            state: 当前 RAG 状态

        Returns:
            str: 路由目标节点名称
        """
        return state.get("route", "rewrite")

    # 当 grade_documents 完成后，根据 route 字段决定下一步
    graph.add_conditional_edges(
        "grade_documents",      # 从评分节点出发
        route_decision,         # 根据 route 字段路由
        {
            "generate": "generate",           # route="generate" → 生成答案
            "rewrite": "rewrite_question"     # route="rewrite" → 重写问题
        }
    )

    # rewrite_question 之后按失败类型路由
    graph.add_conditional_edges(
        "rewrite_question",
        route_decision,  # 复用同一个路由函数
        {
            "generate": "generate",           # missing_knowledge/multi_hop → 直接生成
            "rewrite": "retrieve_expanded",   # ambiguous/retrieval_failure → 扩展检索
        }
    )

    # retrieve_expanded 之后执行 generate
    graph.add_edge("retrieve_expanded", "generate")

    # generate 是终节点，连接到 END
    graph.add_edge("generate", END)

    # 编译图，返回可执行的图实例
    return graph.compile()


# ================================================================
# 执行 RAG
# ================================================================

def run_rag(question: str, baseline: bool = False) -> dict:
    """
    运行完整的 RAG 流程

    Args:
        question: 用户问题
        baseline: True=仅路径1原始检索（传统 RAG），False=四路全开（Agentic RAG）

    Returns:
        dict: 包含 question, answer, relevant_docs, query 字段
    """
    rag_graph = build_rag_graph()

    initial_state = RAGState(
        question=question,
        query="",
        context="",
        docs=[],
        relevant_docs=[],
        route="",
        answer="",
        baseline=baseline,
    )

    # 执行图，获取最终状态
    result = rag_graph.invoke(initial_state)

    # 提取关键结果返回
    return {
        "question": result["question"],
        "answer": result["answer"],
        "relevant_docs": result["relevant_docs"],
        "query": result["query"]
    }


# ================================================================
# 调试：打印图结构
# ================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("RAG 图结构")
    print("=" * 50)

    rag_graph = build_rag_graph()

    # 打印图的可视化描述
    print("""
    流程说明 (v2.0)：
    1. retrieve_initial: 三路检索
       - 原始查询检索
       - HyDE 假设性文档检索
       - Step-back 退步问题检索
       → 合并去重
    2. grade_documents: LLM 评分文档相关性
    3. [条件路由]
       - 有相关文档 → generate
       - 无相关文档 → rewrite_question
    4. rewrite_question: 重写问题
    5. retrieve_expanded: 扩展检索
    6. generate: 生成答案
    """)
