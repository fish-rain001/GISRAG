"""
Tool 基类 —— 对齐 Claude Code Tool.ts。

源文件: src/Tool.ts

对应关系:
  - Tool interface (line 362-695)  → Tool 抽象基类
  - TOOL_DEFAULTS (line 757-768)   → _TOOL_DEFAULTS
  - buildTool() (line 783-792)     → build_tool()
  - ToolResult (line 321-336)      → ToolResult

Tool 接口约 40 个方法，其中 ~25 个是 React/Ink UI 渲染方法。
mini-agent 保留结构性的核心方法，跳过 UI 渲染方法。
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from anthropic.types import ToolResultBlockParam


@dataclass
class ToolResult:
    """
    工具执行结果 —— 对应 Tool.ts:321-336 ToolResult<T>。

    字段:
      data (源码): 工具特定数据
      newMessages (源码): 工具可以返回附加消息
      contextModifier (源码): 修改上下文的回调
      mcpMeta (源码): MCP 协议元数据
    """
    content: str
    is_error: bool = False
    context_modifier: Optional[Callable] = None

    def to_block_param(self, tool_use_id: str) -> ToolResultBlockParam:
        """对应 mapToolResultToToolResultBlockParam()"""
        return ToolResultBlockParam(
            type="tool_result",
            tool_use_id=tool_use_id,
            content=self.content,
            is_error=self.is_error,
        )


# ================================================================
# ToolUseContext —— 对应 Tool.ts:158-300
# ================================================================
@dataclass
class ToolUseContext:
    """
    工具执行上下文 —— 对应 Tool.ts ToolUseContext 类型。

    源码中有 50+ 字段（options, abortController, readFileState,
    getAppState, setAppState, messages, ...）。
    mini-agent 保留核心字段。
    """
    messages: List[Dict[str, Any]] = field(default_factory=list)
    working_dir: str = "."
    abort_signal: Optional[Any] = None  # asyncio.Event
    extra: Dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """
    Tool 抽象基类 —— 对应 Tool.ts:362-695 Tool 类型。

    每个工具必须提供:
      name, description       — 元数据
      input_schema()          — JSON Schema 定义
      call()                  — 执行逻辑
      prompt()                — 生成 prompt 描述

    可选覆盖:
      is_enabled(), is_read_only(), is_concurrency_safe(), is_destructive()
      validate_input(), check_permissions()
      user_facing_name(), get_activity_description()
    """

    # ---- 元数据（对应 Tool 接口字段） ----
    name: str = ""
    description: str = ""
    search_hint: str = ""       # 对应 searchHint
    aliases: List[str] = []     # 对应 aliases

    # ---- Schema ----
    @abstractmethod
    def input_schema(self) -> dict:
        """对应 inputSchema —— JSON Schema"""
        ...

    # ---- 生命周期（对应默认方法 + TOOL_DEFAULTS） ----
    def is_enabled(self) -> bool:
        """对应 isEnabled() —— 默认 true"""
        return True

    def is_read_only(self, input_data: dict) -> bool:
        """对应 isReadOnly() —— 默认 false（安全优先）"""
        return False

    def is_concurrency_safe(self, input_data: dict) -> bool:
        """对应 isConcurrencySafe() —— 默认 false"""
        return False

    def is_destructive(self, input_data: dict) -> bool:
        """对应 isDestructive() —— 默认 false"""
        return False

    # ---- 校验（对应 validateInput + checkPermissions） ----
    def validate_input(self, input_data: dict, context: Any) -> Optional[str]:
        """对应 validateInput() —— 返回错误消息或 None"""
        return None

    async def check_permissions(self, input_data: dict, context: Any) -> dict:
        """
        对应 checkPermissions() —— 返回 PermissionResult。

        默认: { 'behavior': 'allow', 'updatedInput': input_data }
        """
        return {"behavior": "allow", "updatedInput": input_data}

    # ---- 核心方法 ----
    @abstractmethod
    async def call(self, input_data: dict, context: Any) -> ToolResult:
        """对应 call() —— 执行工具"""
        ...

    # ---- Prompt 生成 ----
    def prompt(self) -> str:
        """对应 prompt() —— 生成工具的 prompt 描述"""
        return f"- {self.name}: {self.description}"

    # ---- 用户可见 ----
    def user_facing_name(self, input_data: Optional[dict] = None) -> str:
        """对应 userFacingName() —— 默认返回 name"""
        return self.name

    def get_activity_description(self, input_data: dict) -> Optional[str]:
        """对应 getActivityDescription()"""
        return None

    # ---- 分类器输入（对应 toAutoClassifierInput） ----
    def to_auto_classifier_input(self, input_data: dict) -> Any:
        """对应 toAutoClassifierInput() —— 默认 '' (skip)"""
        return ""

    # ---- 结果序列化（对应 mapToolResultToToolResultBlockParam） ----
    def map_result_to_block_param(
        self, result: ToolResult, tool_use_id: str
    ) -> ToolResultBlockParam:
        """
        对应 mapToolResultToToolResultBlockParam()。

        默认实现：用 ToolResult 的通用格式。
        工具可以覆盖此方法实现自定义序列化（如 SkillTool 的 fork vs inline）。
        """
        return result.to_block_param(tool_use_id)


# ================================================================
# TOOL_DEFAULTS —— 对应 Tool.ts:757-768
# ================================================================
_TOOL_DEFAULTS: Dict[str, Any] = {
    "is_enabled": lambda self: True,
    "is_read_only": lambda self, input_data: False,
    "is_concurrency_safe": lambda self, input_data: False,
    "is_destructive": lambda self, input_data: False,
    "check_permissions": lambda self, input_data, context: {
        "behavior": "allow",
        "updatedInput": input_data,
    },
    "to_auto_classifier_input": lambda self, input_data: "",
    "user_facing_name": lambda self, input_data: "",
    "validate_input": lambda self, input_data, context: None,
    "get_activity_description": lambda self, input_data: None,
    "aliases": [],
    "search_hint": "",
}


# ================================================================
# build_tool —— 对应 Tool.ts:783-792 buildTool()
# ================================================================
def build_tool(cls: type) -> type:
    """
    对应 buildTool() —— 将部分定义的 Tool 类填充默认值。

    运行时等价于 { ...TOOL_DEFAULTS, ...def }：
      如果子类定义了方法 → 用子类的
      如果子类没定义 → 用 TOOL_DEFAULTS 的

    fail-closed 安全策略:
      - is_read_only          → False（假设写入）
      - is_concurrency_safe   → False（假设不安全）
      - is_destructive        → False
      - check_permissions     → allow（委托给通用权限系统）
      - to_auto_classifier_input → ''（跳过分类器）
      - user_facing_name      → name

    用法:
        @build_tool
        class MyTool(Tool):
            name = "MyTool"
            ...
    """
    for attr_name, default_fn in _TOOL_DEFAULTS.items():
        if not hasattr(cls, attr_name):
            # 如果子类没定义，注入默认实现
            if callable(default_fn) and not isinstance(default_fn, type):
                # 对于 lambda 定义的 callable 默认值，转为 staticmethod
                if attr_name in ("aliases", "search_hint"):
                    setattr(cls, attr_name, default_fn)
                # 其他跳过，因为 ABC 已有默认方法
    return cls
