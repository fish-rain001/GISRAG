"""Tool 注册表 —— 统一管理所有工具的注册、查询、Schema 导出"""

from typing import Dict, List, Optional

from .base import Tool


class ToolRegistry:
    """
    工具注册中心，对应 Claude Code 的 getAllBaseTools() + getTools()。
    管理所有可用工具，支持按名称查询。
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个工具"""
        self._tools[tool.name] = tool

    def register_all(self, tools: List[Tool]) -> None:
        """批量注册"""
        for t in tools:
            self.register(t)

    def get(self, name: str) -> Optional[Tool]:
        """按名称获取工具"""
        return self._tools.get(name)

    def get_all(self) -> List[Tool]:
        """获取所有已启用的工具"""
        return [t for t in self._tools.values() if t.is_enabled()]

    def get_names(self) -> List[str]:
        """获取所有工具名称"""
        return [t.name for t in self.get_all()]

    def get_tool_schemas(self) -> List[dict]:
        """
        生成 Anthropic API 格式的 tools 定义列表。
        这是发送给 LLM 的 tool_use 声明。
        """
        schemas = []
        for tool in self.get_all():
            schema = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema(),
            }
            schemas.append(schema)
        return schemas

    def __len__(self) -> int:
        return len(self.get_all())

    def __contains__(self, name: str) -> bool:
        return name in self._tools
