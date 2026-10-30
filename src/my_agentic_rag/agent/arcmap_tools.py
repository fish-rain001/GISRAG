# -*- coding: utf-8 -*-
"""
把 arcmap-mcp 的 MCP 工具包装成项目里的 Tool。

设计原则：
  - 只暴露"查"的工具，不暴露任何会改数据的工具
  - 工具名统一加 arcmap_ 前缀，避免和 search_standards 混淆
  - 描述用中文，参数名保持 MCP 原始 schema（服务器强制 additionalProperties=false）
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import config
from .mcp_stdio import McpStdioClient
from .tools.base import Tool, ToolResult

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = PROJECT_ROOT / "out" / "tool_results"

# 首期白名单：全部是只读/检查类工具
ARCMAP_ALLOWLIST = [
    "describe_mxd",
    "audit_folder",
    "ping",
    "get_arcmap_info",
    "list_layers",
    "get_layer_info",
    "describe_data",
    "list_fields",
    "count_features",
    "get_unique_values",
    "list_broken_data_sources",
]

# 每个工具的中文说明和超时（秒）
TOOL_META: Dict[str, Dict[str, Any]] = {
    "describe_mxd": {
        "label": "检查 mxd 版本",
        "desc": (
            "读取 .mxd 文件声明的 ArcMap 版本并判断兼容性，不需要打开 ArcMap。"
            "参数 ruta = .mxd 文件路径。"
        ),
        "timeout": 120,
    },
    "audit_folder": {
        "label": "批量审计 mxd",
        "desc": (
            "扫描文件夹内所有 .mxd：版本、图层、断源数据源、定义查询，"
            "并区分图层是否在出图范围内。不需要打开 ArcMap。"
            "参数 ruta = 文件夹路径；可选 con_capas（是否读取图层，默认 true）、"
            "timeout_por_documento（每个文档超时秒数）、max_documentos（最多扫描数）、"
            "recursivo（是否递归子目录）。"
        ),
        "timeout": 900,
    },
    "ping": {
        "label": "检查 ArcMap 桥接",
        "desc": "检查 ArcMap 是否已打开且 Add-in 已启动，返回版本和当前文档。",
        "timeout": 30,
    },
    "get_arcmap_info": {
        "label": "读取当前文档信息",
        "desc": "读取当前 mxd、数据框、比例尺、视图。需要 ArcMap 已打开并启动桥接。",
        "timeout": 60,
    },
    "list_layers": {
        "label": "列出图层",
        "desc": (
            "列出当前 ArcMap 文档的图层（名称、可见性、数据源、定义查询）。"
            "需要 ArcMap 已打开并启动桥接。"
        ),
        "timeout": 60,
    },
    "get_layer_info": {
        "label": "图层详情",
        "desc": "读取单个图层的详细信息。参数 capa = 图层名或 TOC 路径。",
        "timeout": 60,
    },
    "describe_data": {
        "label": "读取数据信息",
        "desc": (
            "读取 shp/gdb/栅格的数据信息：字段、范围、坐标系、要素数等。"
            "参数 ruta = 数据路径。需要 ArcMap 已打开并启动桥接。"
        ),
        "timeout": 60,
    },
    "list_fields": {
        "label": "列出字段",
        "desc": "列出图层字段。参数 capa = 图层名。",
        "timeout": 60,
    },
    "count_features": {
        "label": "统计要素数",
        "desc": "统计图层要素数量。参数 capa = 图层名；可选 where = 筛选表达式。",
        "timeout": 120,
    },
    "get_unique_values": {
        "label": "字段唯一值",
        "desc": "统计字段的唯一值及数量。参数 capa = 图层名，campo = 字段名。",
        "timeout": 120,
    },
    "list_broken_data_sources": {
        "label": "列出断源",
        "desc": "列出当前文档中断开的数据源。需要 ArcMap 已打开并启动桥接。",
        "timeout": 60,
    },
}


def _content_to_text(result: Any) -> str:
    """把 CallToolResult.content 里的文本块拼起来。"""
    parts: List[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
            continue
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "image":
                parts.append("[图片内容]")
    return "\n".join(p for p in parts if p).strip()


def _persist(text: str, name: str) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path.write_text(text, encoding="utf-8")
    return path


class ArcMapTool(Tool):
    """一个 arcmap-mcp 工具的项目内包装。"""

    def __init__(
        self,
        client: McpStdioClient,
        mcp_name: str,
        schema: dict,
        description: str,
        timeout: int = 120,
    ):
        self._client = client
        self._mcp_name = mcp_name
        self._schema = schema or {"type": "object", "properties": {}}
        self._timeout = timeout
        self.name = f"arcmap_{mcp_name}"
        self.description = description
        self.search_hint = TOOL_META.get(mcp_name, {}).get("label", mcp_name)

    def input_schema(self) -> dict:
        return self._schema

    def is_read_only(self, input_data: dict) -> bool:
        return True

    def is_concurrency_safe(self, input_data: dict) -> bool:
        # ArcMap 桥接一次只处理一个命令，串行执行
        return False

    async def call(self, input_data: dict, context: Any) -> ToolResult:
        try:
            result = await self._client.call_tool(
                self._mcp_name, input_data, timeout_seconds=self._timeout
            )
        except Exception as e:
            return ToolResult(
                content=f"调用 {self._mcp_name} 失败：{type(e).__name__}: {e}",
                is_error=True,
            )

        text = _content_to_text(result)
        is_error = bool(getattr(result, "isError", False))
        if not text:
            text = "(工具未返回文本内容)"

        if len(text) > config.TOOL_RESULT_MAX_CHARS:
            path = _persist(text, self._mcp_name)
            text = (
                text[: config.TOOL_RESULT_MAX_CHARS]
                + f"\n\n[结果过长，完整内容已保存到 {path}]"
            )

        if is_error:
            return ToolResult(content=f"工具 {self._mcp_name} 报错：{text}", is_error=True)
        return ToolResult(content=text)


def build_arcmap_tools(
    client: McpStdioClient,
    allowlist: Optional[List[str]] = None,
) -> List[Tool]:
    """从 MCP 的 tools/list 结果里挑出白名单工具，包装成 Tool 列表。"""
    allowlist = allowlist or ARCMAP_ALLOWLIST
    available = {getattr(t, "name", ""): t for t in client.tools}

    tools: List[Tool] = []
    for name in allowlist:
        mcp_tool = available.get(name)
        if mcp_tool is None:
            print(f"[agent] MCP 服务没有提供工具 {name}，跳过")
            continue
        meta = TOOL_META.get(name, {})
        schema = (
            getattr(mcp_tool, "inputSchema", None)
            or getattr(mcp_tool, "input_schema", None)
            or {"type": "object", "properties": {}}
        )
        desc = meta.get("desc") or getattr(mcp_tool, "description", "") or name
        if name in ("ping", "get_arcmap_info", "list_layers", "get_layer_info",
                    "describe_data", "list_fields", "count_features",
                    "get_unique_values", "list_broken_data_sources"):
            desc += " 需要 ArcMap 已打开且 Add-in 已启动；否则会返回 PUENTE CAIDO。"
        tools.append(
            ArcMapTool(
                client=client,
                mcp_name=name,
                schema=schema,
                description=desc,
                timeout=int(meta.get("timeout", 120)),
            )
        )
    return tools
