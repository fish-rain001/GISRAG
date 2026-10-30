# -*- coding: utf-8 -*-
"""
Agentic RAG 外层图：路由 + 工具调用循环。

知识问题走原有五节点 RAG；
操作/混合问题走 agent ⇄ tools 循环，最后用工具结果和规范依据生成答案。
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, List, Optional, TypedDict

import jsonschema
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from .. import config
from ..rag_graph import get_llm, get_response_text, run_rag
from .arcmap_tools import build_arcmap_tools
from .mcp_stdio import McpStdioClient
from .rag_tool import RagSearchTool
from .tools.base import Tool
from .tools.registry import ToolRegistry


# ================================================================
# 状态与路由
# ================================================================

class AgentState(TypedDict, total=False):
    question: str
    intent: str
    route_reason: str
    messages: List[Any]
    tool_rounds: int
    tool_log: List[dict]
    answer: str
    relevant_docs: List[dict]


ROUTER_PROMPT = """你是测绘成果质检助手的路由器。判断用户问题需要哪种能力：

- knowledge：问规范要求、术语定义、限差、依据、作业流程，不需要读取用户电脑上的具体文件。
- operation：要求检查、扫描、读取、统计某个文件/文件夹/图层/mxd。
- mixed：既需要规范依据，又需要检查具体成果数据。

用户问题：{question}

只回复一个英文单词：knowledge、operation 或 mixed。不要解释，不要加标点。"""


def _keyword_route(question: str) -> str:
    """LLM 路由失败时的保守兜底。"""
    q = question.lower()
    has_path = bool(re.search(r"[a-z]:[\\/]", q)) or any(
        ext in q for ext in (".mxd", ".shp", ".gdb", ".tif", ".img", ".dwg")
    )
    has_action = any(
        w in question for w in ("检查", "扫描", "读取", "列出", "统计", "打开", "看一下", "查一下")
    )
    asks_standard = any(
        w in question for w in ("规范", "标准", "条款", "依据", "限差", "要求", "定义")
    )
    # "不需要查规范/只查数据" 这类否定表达，不能当成要查规范
    if any(w in question for w in ("不需要查规范", "不用查规范", "只查数据", "只检查数据")):
        asks_standard = False

    if (has_path or has_action) and asks_standard:
        return "mixed"
    if has_path or has_action:
        return "operation"
    return "knowledge"


def router_node(state: AgentState) -> dict:
    question = state["question"]
    try:
        # 不用 with_structured_output：DeepSeek thinking 模式不支持 tool_choice
        response = get_llm().invoke(ROUTER_PROMPT.format(question=question))
        text = get_response_text(response).strip().lower()
        if "mixed" in text:
            intent = "mixed"
        elif "operation" in text:
            intent = "operation"
        elif "knowledge" in text:
            intent = "knowledge"
        else:
            intent = _keyword_route(question)
            text = f"无法解析路由结果：{text[:80]}"
        reason = text[:120]
    except Exception as e:
        intent = _keyword_route(question)
        reason = f"路由 LLM 失败，使用关键词兜底：{e}"

    print(f"\n[router] intent={intent} ({reason})")
    return {"intent": intent, "route_reason": reason}


def route_after_router(state: AgentState) -> str:
    return "knowledge" if state.get("intent") == "knowledge" else "agent"


# ================================================================
# 知识分支：直接复用原有五节点 RAG
# ================================================================

def knowledge_node(state: AgentState) -> dict:
    question = state["question"]
    print("\n[knowledge] 走原有 RAG 流程")
    try:
        result = run_rag(question)
        return {
            "answer": result.get("answer", ""),
            "relevant_docs": result.get("relevant_docs", []),
            "tool_log": [{
                "tool": "search_standards",
                "args": {"query": question},
                "status": "ok",
                "preview": result.get("answer", "")[:300],
                "via": "knowledge_node",
            }],
        }
    except Exception as e:
        return {
            "answer": f"知识库检索失败：{type(e).__name__}: {e}",
            "relevant_docs": [],
            "tool_log": [{
                "tool": "search_standards",
                "args": {"query": question},
                "status": "error",
                "preview": str(e)[:300],
                "via": "knowledge_node",
            }],
        }


# ================================================================
# Agent 分支：LLM + 工具循环
# ================================================================

SYSTEM_PROMPT = """你是测绘成果质检助手。你可以使用两类工具：

1. search_standards：检索测绘规范、法规、技术大纲。用于回答"要求是什么、限差多少、依据哪一条"。
2. arcmap_*：读取 ArcMap 工程和数据的事实，例如 mxd 版本、断源图层、图层清单、字段、要素数量。

工作规则：
- 先判断用户要的是规范知识、检查具体数据，还是两者都要。
- 两者都要时，先查规范拿到判定标准，再查数据拿到事实。
- 只能调用提供的工具，不要编造工具或参数；参数名可能是西班牙语，严格按 schema 填写。
- 工具报错时，把缺什么告诉用户（例如 ArcMap 未打开、路径不存在），不要假装成功。
- 最终回答必须区分"事实"（工具返回）和"依据"（规范条款）；没有检索到依据就明确说"未找到条款依据"。
- 不要编造工具没有返回的数据，也不要把推测写成事实。
"""


def _validate_tool_input(tool: Tool, args: dict) -> Optional[str]:
    """必填参数 + jsonschema 校验，返回错误消息或 None。"""
    try:
        schema = tool.input_schema() or {}
    except Exception:
        schema = {}

    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        available = list(schema.get("properties", {}).keys())
        return (
            f"缺少必填参数：{', '.join(missing)}。"
            f"可用参数：{', '.join(available) if available else '(无)'}"
        )

    if schema:
        try:
            jsonschema.validate(args, schema)
        except jsonschema.ValidationError as e:
            field = ".".join(str(p) for p in e.absolute_path) or "(root)"
            return f"参数 {field} 不合法：{e.message}"
        except jsonschema.SchemaError:
            pass

    custom = tool.validate_input(args, None)
    if custom:
        return custom
    return None


def make_agent_node(llm_with_tools):
    async def agent_node(state: AgentState) -> dict:
        messages = list(state.get("messages") or [])
        if not messages:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=state["question"]),
            ]

        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        return {"messages": messages}

    return agent_node


def make_tools_node(registry: ToolRegistry):
    async def tools_node(state: AgentState) -> dict:
        messages = list(state.get("messages") or [])
        last = messages[-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        logs = list(state.get("tool_log") or [])
        new_messages: List[ToolMessage] = []

        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args") or {}
            call_id = call.get("id", "")

            tool = registry.get(name)
            if tool is None:
                content, is_error = f"未知工具：{name}", True
            else:
                validation_error = _validate_tool_input(tool, args)
                if validation_error:
                    content, is_error = f"InputValidationError: {validation_error}", True
                else:
                    print(f"\n[tool] {name}({args})")
                    try:
                        result = await asyncio.wait_for(
                            tool.call(args, None),
                            timeout=max(60, getattr(tool, "_timeout", 120) + 30),
                        )
                        content, is_error = result.content, result.is_error
                    except asyncio.TimeoutError:
                        content, is_error = f"工具 {name} 执行超时", True
                    except Exception as e:
                        content, is_error = (
                            f"工具执行失败：{type(e).__name__}: {e}",
                            True,
                        )

            new_messages.append(
                ToolMessage(content=content, tool_call_id=call_id, name=name)
            )
            logs.append({
                "tool": name,
                "args": args,
                "status": "error" if is_error else "ok",
                "preview": content[:300],
            })
            print(f"       -> {'ERROR' if is_error else 'OK'} {content[:120]}...")

        return {
            "messages": messages + new_messages,
            "tool_rounds": (state.get("tool_rounds") or 0) + 1,
            "tool_log": logs,
        }

    return tools_node


def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"


def route_after_tools(state: AgentState) -> str:
    if (state.get("tool_rounds") or 0) >= config.AGENT_MAX_TOOL_ROUNDS:
        return "finalize"
    return "agent"


FINALIZE_PROMPT = """请根据以上工具结果回答用户问题。要求：
1. 先列事实：来自哪个工具、返回了什么。
2. 再列依据：规范名称、标准号、条款号；没有检索到就明确说"未找到条款依据"。
3. 最后给结论或建议。
4. 不要编造工具没有返回的数据。

用户问题：{question}"""


async def finalize_node(state: AgentState) -> dict:
    messages = list(state.get("messages") or [])
    messages.append(HumanMessage(content=FINALIZE_PROMPT.format(question=state["question"])))
    response = await get_llm().ainvoke(messages)
    messages.append(response)
    return {"messages": messages, "answer": get_response_text(response)}


# ================================================================
# 构图与运行
# ================================================================

def build_agent_graph(tools: List[Tool]):
    registry = ToolRegistry()
    registry.register_all(tools)
    registry.register(RagSearchTool())

    llm_with_tools = get_llm().bind_tools(registry.get_tool_schemas())

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("agent", make_agent_node(llm_with_tools))
    graph.add_node("tools", make_tools_node(registry))
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router", route_after_router,
        {"knowledge": "knowledge", "agent": "agent"},
    )
    graph.add_conditional_edges(
        "agent", route_after_agent,
        {"tools": "tools", "end": END},
    )
    graph.add_conditional_edges(
        "tools", route_after_tools,
        {"agent": "agent", "finalize": "finalize"},
    )
    graph.add_edge("knowledge", END)
    graph.add_edge("finalize", END)

    return graph.compile(), registry


def _extract_answer(state: dict) -> str:
    answer = (state.get("answer") or "").strip()
    if answer:
        return answer
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            text = get_response_text(msg).strip()
            if text:
                return text
    return ""


async def run_agent(question: str, use_mcp: bool = True) -> dict:
    """
    运行 Agentic RAG。

    MCP 连接失败时自动降级为仅 RAG，不阻塞问答。
    """
    tools: List[Tool] = []
    mcp: Optional[McpStdioClient] = None
    mcp_error = ""

    if use_mcp:
        try:
            mcp = McpStdioClient()
            await asyncio.wait_for(mcp.connect(), timeout=30)
            tools = build_arcmap_tools(mcp)
            print(f"[agent] ArcMap MCP 已连接，注册 {len(tools)} 个工具")
        except Exception as e:
            mcp_error = f"{type(e).__name__}: {e}"
            print(f"[agent] ArcMap MCP 不可用，降级为仅 RAG：{mcp_error}")
            if mcp is not None:
                try:
                    await mcp.close()
                except Exception:
                    pass
                mcp = None

    try:
        graph, registry = build_agent_graph(tools)
        initial: AgentState = {
            "question": question,
            "intent": "",
            "route_reason": "",
            "messages": [],
            "tool_rounds": 0,
            "tool_log": [],
            "answer": "",
            "relevant_docs": [],
        }
        result = await graph.ainvoke(initial)
        return {
            "question": question,
            "answer": _extract_answer(result),
            "intent": result.get("intent", ""),
            "route_reason": result.get("route_reason", ""),
            "relevant_docs": result.get("relevant_docs", []),
            "tool_log": result.get("tool_log", []),
            "available_tools": registry.get_names(),
            "mcp_error": mcp_error,
        }
    finally:
        if mcp is not None:
            try:
                await mcp.close()
            except Exception:
                pass
