# -*- coding: utf-8 -*-
"""
ArcMap MCP 的 stdio 客户端。

arcmap-mcp 是 FastMCP + stdio 服务，本模块负责：
  1. 启动它的 Python 3 虚拟环境
  2. 建立 MCP 会话
  3. list_tools / call_tool
  4. 退出时关闭子进程

不修改 arcmap-mcp 的任何源码，只把它当成外部工具服务。
"""
from __future__ import annotations

import sys
from contextlib import AsyncExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .. import config

# 常见的安装位置，按顺序找第一个存在的
_PYTHON_CANDIDATES = [
    config.ARCMAP_MCP_PYTHON,
    r"C:\Users\gorde\AppData\Local\arcmap-mcp\venv\Scripts\python.exe",
]
_SERVER_CANDIDATES = [
    config.ARCMAP_MCP_SERVER,
    r"C:\mcp\arcmap-mcp\src\arcmap_mcp_server.py",
    r"D:\arcmap\arcmap-mcp-main\src\arcmap_mcp_server.py",
]


def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and Path(p).exists():
            return p
    return None


class McpStdioClient:
    """一个进程内生命周期内的 MCP stdio 会话。"""

    def __init__(self, command: Optional[str] = None, server_script: Optional[str] = None):
        self.command = command or _first_existing(_PYTHON_CANDIDATES) or sys.executable
        self.server_script = server_script or _first_existing(_SERVER_CANDIDATES)
        if not self.server_script:
            raise FileNotFoundError(
                "找不到 arcmap_mcp_server.py，请检查 ARCMAP_MCP_SERVER 配置"
            )
        if not Path(self.command).exists():
            raise FileNotFoundError(f"找不到 ArcMap MCP 的 Python：{self.command}")

        self._stack = AsyncExitStack()
        self._session: Optional[ClientSession] = None
        self.tools: List[Any] = []

    async def connect(self) -> List[Any]:
        """启动子进程并完成 MCP 初始化，返回工具列表。"""
        params = StdioServerParameters(
            command=self.command,
            args=[self.server_script],
            env=None,  # 继承当前环境，arcmap-mcp 需要 PATH/SystemRoot 等
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

        result = await self._session.list_tools()
        self.tools = list(result.tools or [])
        return self.tools

    async def call_tool(
        self,
        name: str,
        arguments: Optional[dict] = None,
        timeout_seconds: int = 120,
    ):
        """调用一个 MCP 工具，返回 mcp.types.CallToolResult。"""
        if self._session is None:
            raise RuntimeError("MCP 会话尚未连接")
        return await self._session.call_tool(
            name,
            arguments or {},
            read_timeout_seconds=timedelta(seconds=timeout_seconds),
        )

    async def close(self) -> None:
        """关闭会话和子进程。"""
        try:
            await self._stack.aclose()
        finally:
            self._session = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
