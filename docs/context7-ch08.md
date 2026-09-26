# Ch08 Context7 API 核对（2026-09-27）

在第 8 章实现前查询 Context7 的官方仓库文档；安装依赖后仍须按实际锁定版本复核。

实装复核：先装到既有虚拟环境的 `langchain-mcp-adapters==0.3.1` 依赖元数据只要求 `mcp>=1.24.0`，未排除 2.x；无约束解析装入 `mcp==2.2.0` 后，适配器导入 `mcp.shared.context.RequestContext` 失败。故将官方 SDK 约束为 `mcp>=1.24,<2`，锁定并安装 `mcp==1.30.0`、`langchain-mcp-adapters==0.3.2` 后成功导入。此 SDK 版本以 `from mcp.server.fastmcp import FastMCP`、构造器的 host/port、`run(transport="streamable-http")` 启动；适配器构造器支持 `handle_tool_errors`。Context7 的 `/modelcontextprotocol/python-sdk/v1.12.4` 版本文档也给出 FastMCP/Streamable HTTP 用法。

- MCP 官方 Python SDK：`/modelcontextprotocol/python-sdk`。当前主线文档示例使用 `from mcp.server import MCPServer`（或 `mcp.server.mcpserver`）、`@mcp.tool()`，以 `mcp.run(transport="streamable-http", json_response=True)` 启动；transport 的 host/port/path 参数传给 `run()`。文档亦指出 v1 的 `FastMCP` 路径不同，不能只凭主线示例断定安装版本。
- LangChain MCP Adapters：`/langchain-ai/langchain-mcp-adapters`。`MultiServerMCPClient` 使用每 Server 一个配置字典，Streamable HTTP transport 值是 `streamable_http`；`get_tools(server_name=...)` 返回 `list[BaseTool]`。当前文档列出 `handle_tool_errors`；安装后用签名检查确认。
- Python jsonschema：`/python-jsonschema/jsonschema`。`validate(instance=..., schema=...)` 对无效参数抛 `ValidationError`，包含 `message`、`path` 等诊断字段。

官方来源：

- https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/run/index.md
- https://github.com/modelcontextprotocol/python-sdk/blob/main/examples/snippets/servers/mcpserver_quickstart.py
- https://github.com/langchain-ai/langchain-mcp-adapters/blob/main/_autodocs/api-reference-client.md
- https://github.com/langchain-ai/langchain-mcp-adapters/blob/main/_autodocs/configuration.md
- https://github.com/python-jsonschema/jsonschema/blob/main/docs/validate.rst
