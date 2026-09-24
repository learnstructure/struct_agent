"""
Base types shared by every struct_agent tool.

Every tool returns a :class:`ToolResult` so that the orchestrator (or an
LLM wrapper) can decide what to do next in a uniform way. The result
itself carries structured engineering data; the AI layer never has to
parse a free-form string to recover the engineering state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentError:
    """Machine-readable engineering error returned by a tool."""

    error_type: str
    message: str
    possible_causes: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    """
    Structured return value for every struct_agent tool.

    Attributes
    ----------
    status
        One of ``"success"``, ``"failed"``, ``"skipped"``.
    tool
        The tool name (e.g. ``"run_analysis"``).
    data
        Structured engineering payload produced by the tool (e.g. analysis
        case IDs, governing DCR, drawing paths). Keep this small and explicit.
    error
        Optional :class:`AgentError` describing the failure.
    next_suggested_stage
        Optional workflow stage that the orchestrator should advance to.
    """

    status: str
    tool: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[AgentError] = None
    next_suggested_stage: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "tool": self.tool,
            "data": self.data,
            "error": self.error.to_dict() if self.error else None,
            "next_suggested_stage": self.next_suggested_stage,
        }

    @classmethod
    def success(
        cls,
        tool: str,
        data: Dict[str, Any],
        next_suggested_stage: Optional[str] = None,
    ) -> "ToolResult":
        return cls(
            status="success",
            tool=tool,
            data=data,
            next_suggested_stage=next_suggested_stage,
        )

    @classmethod
    def failure(
        cls,
        tool: str,
        error: AgentError,
        data: Optional[Dict[str, Any]] = None,
        next_suggested_stage: Optional[str] = None,
    ) -> "ToolResult":
        return cls(
            status="failed",
            tool=tool,
            data=data or {},
            error=error,
            next_suggested_stage=next_suggested_stage,
        )