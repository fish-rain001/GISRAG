# -*- coding: utf-8 -*-
"""把现有的 Agentic RAG 流程包装成一个工具，供 agent 按需调用。"""
from __future__ import annotations

import asyncio
from typing import Any

from .tools.base import Tool, ToolResult


class RagSearchTool(Tool):
    """
    检索测绘规范/法规/技术大纲知识库。

    内部直接复用 rag_graph.run_rag（五节点 Agentic RAG），
    不改变原有检索链路，只是从"每次必走"变成"agent 需要时才走"。
    """

    name = "search_standards"
    description = (
        "检索测绘规范、法律法规、技术大纲知识库，返回带来源的答案。"
        "适合回答：规范要求是什么、限差多少、依据哪一条、术语定义、作业流程。"
        "不用于读取用户电脑上的具体文件或 ArcMap 图层。"
    )
    search_hint = "查测绘规范、标准、限差、依据"
    # 五节点 RAG 会多次调用 LLM，冷启动还要加载 BGE + Reranker，给足超时
    _timeout = 900

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要检索的问题，尽量包含标准名称、指标或专业术语",
                }
            },
            "required": ["query"],
        }

    def is_read_only(self, input_data: dict) -> bool:
        return True

    def is_concurrency_safe(self, input_data: dict) -> bool:
        # 复用单例向量库和 LLM，串行更稳
        return False

    async def call(self, input_data: dict, context: Any) -> ToolResult:
        query = (input_data or {}).get("query", "").strip()
        if not query:
            return ToolResult(content="缺少参数 query", is_error=True)

        try:
            from ..rag_graph import run_rag

            result = await asyncio.to_thread(run_rag, query)
        except Exception as e:
            return ToolResult(
                content=f"规范检索失败：{type(e).__name__}: {e}",
                is_error=True,
            )

        answer = result.get("answer", "")
        sources = []
        for doc in result.get("relevant_docs", []):
            src = doc.get("metadata", {}).get("source", "")
            if src and src not in sources:
                sources.append(src)

        content = f"【规范检索结果】\n{answer}"
        if sources:
            content += "\n\n【来源文件】\n" + "\n".join(f"- {s}" for s in sources)
        else:
            content += "\n\n【来源文件】\n- 未检索到可用来源"

        return ToolResult(content=content)
