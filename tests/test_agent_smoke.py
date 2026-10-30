# -*- coding: utf-8 -*-
"""Agent 层冒烟测试：不需要 LLM、Milvus 或 ArcMap。"""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from my_agentic_rag.agent.arcmap_tools import ARCMAP_ALLOWLIST, TOOL_META
from my_agentic_rag.agent.graph import (
    _keyword_route,
    _validate_tool_input,
    build_agent_graph,
)
from my_agentic_rag.agent.rag_tool import RagSearchTool
from my_agentic_rag.agent.tools.base import Tool, ToolResult
from my_agentic_rag.agent.tools.registry import ToolRegistry


class DummyTool(Tool):
    name = "dummy"
    description = "测试工具"

    def input_schema(self):
        return {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": False,
        }

    async def call(self, input_data, context):
        return ToolResult(content="ok")


class AgentSmokeTest(unittest.TestCase):
    def test_keyword_route(self):
        self.assertEqual(_keyword_route("GB/T 24356 的限差是多少？"), "knowledge")
        self.assertEqual(_keyword_route("扫描 D:/成果 里的 mxd"), "operation")
        self.assertEqual(_keyword_route("检查这个 shp 并按规范判断"), "mixed")
        self.assertEqual(
            _keyword_route("不需要查规范，只扫描 D:/成果 的 mxd"), "operation"
        )

    def test_validation(self):
        tool = DummyTool()
        self.assertIn("缺少必填参数", _validate_tool_input(tool, {}))
        self.assertIn("Additional properties", _validate_tool_input(tool, {"a": "x", "b": 1}))
        self.assertIsNone(_validate_tool_input(tool, {"a": "x"}))

    def test_registry_schema(self):
        registry = ToolRegistry()
        registry.register(DummyTool())
        registry.register(RagSearchTool())
        schemas = registry.get_tool_schemas()
        by_name = {s["name"]: s for s in schemas}
        self.assertEqual(by_name["dummy"]["input_schema"]["required"], ["a"])
        self.assertEqual(by_name["search_standards"]["input_schema"]["required"], ["query"])

    def test_arcmap_allowlist_metadata(self):
        self.assertEqual(len(ARCMAP_ALLOWLIST), 11)
        for name in ARCMAP_ALLOWLIST:
            self.assertIn(name, TOOL_META)

    def test_build_graph_without_mcp(self):
        graph, registry = build_agent_graph([])
        self.assertEqual(registry.get_names(), ["search_standards"])
        self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()
