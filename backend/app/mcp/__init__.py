"""The MCP tool layer — task-shaped coaching tools over the per-user API (issue #42).

Everywhere else an LLM is handed a fixed context blob assembled ahead of time by
a prompt builder guessing what the coach would need. This package inverts that:
it publishes a small set of **task-shaped, read-only tools** and lets the model
ask.

Two consumers, one tool layer
-----------------------------
- the **internal agent** (issue #43), running on-server against a user's own
  session context, and
- **external MCP clients**, authenticating with a personal access token (#46).

They arrive by different doors — :mod:`backend.app.mcp.server` for the second, a
direct :func:`backend.app.mcp.dispatch.call_tool` for the first — and reach the
same tools through the same checks. There is deliberately no "internal" bypass on
anything except the rate limiter.

What lives where
----------------
=========================================  =============================================
Concern                                     Home
=========================================  =============================================
Tool declarations (name, scopes, schemas)   :mod:`backend.app.mcp.registry`
Running one tool, with every check applied  :mod:`backend.app.mcp.dispatch`
Model-facing failures                       :mod:`backend.app.mcp.errors`
Response shaping helpers (units, paging)    :mod:`backend.app.mcp.shaping`
The tools themselves                        :mod:`backend.app.mcp.tools`
MCP protocol + HTTP transport               :mod:`backend.app.mcp.server`
=========================================  =============================================

Isolation is physical, not logical
----------------------------------
Every user's training data is a separate SQLite file under a separate encryption
context, with no crypto-level defence in depth behind it: **access control is the
boundary**. So no tool resolves an identity, opens a session, or looks up an
athlete for itself — :func:`backend.app.mcp.dispatch.call_tool` does all three,
through the same ``open_user_session`` / ``load_athlete`` the HTTP routes use,
and hands the handler a ready-made
:class:`~backend.app.mcp.dispatch.ToolRun`. A tool that could open its own
session would be a tool that could open the wrong one.
"""

from backend.app.mcp.errors import ToolError
from backend.app.mcp.registry import Tool, all_tools, get_tool, tool

__all__ = ["Tool", "ToolError", "all_tools", "get_tool", "tool"]
