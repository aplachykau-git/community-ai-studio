"""
Terminal Demo Logger for Community AI Studio.
Provides real-time, beautifully formatted, colorized terminal logs for:
- User prompts and attachments
- Multi-agent orchestration and routing handoffs
- Tool invocations and results
- Model execution and LLM responses
- A2A (Agent-to-Agent) RPC requests, responses, and latencies
"""

import datetime
import json
import time
from typing import Any, Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ANSI Color Codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Agent Colors matching UI tokens
COLOR_ROOT = "\033[94m"  # Blue (#4285F4)
COLOR_RECEIPT = "\033[92m"  # Green (#34A853)
COLOR_VIDEO = "\033[93m"  # Yellow/Gold (#FBBC04)
COLOR_LINKEDIN = "\033[36m"  # Teal/Blue (#0A66C2)
COLOR_REGISTRATION = "\033[95m"  # Purple (#A142F4)
COLOR_PLANNER = "\033[91m"  # Red (#EA4335)
COLOR_AGENDA = "\033[96m"  # Cyan (#24C1E0)
COLOR_OFFICE = "\033[33m"  # Orange (#FA7B17)
COLOR_TOOL = "\033[35m"  # Magenta
COLOR_SUCCESS = "\033[92m"  # Bright Green
COLOR_ERROR = "\033[91m"  # Bright Red
COLOR_MUTED = "\033[90m"  # Bright Black (Gray)

AGENT_COLORS = {
    "root_agent": COLOR_ROOT,
    "receipt_scanner": COLOR_RECEIPT,
    "video_editor": COLOR_VIDEO,
    "linkedin_post_generator": COLOR_LINKEDIN,
    "registration_manager": COLOR_REGISTRATION,
    "event_planner": COLOR_PLANNER,
    "agenda_generator": COLOR_AGENDA,
    "office_secretary": COLOR_OFFICE,
}

AGENT_LABELS = {
    "root_agent": "Main Orchestrator",
    "receipt_scanner": "Receipt Scanner (A2A)",
    "video_editor": "Live Video Editor (A2A)",
    "linkedin_post_generator": "LinkedIn Planner",
    "registration_manager": "Registrations Manager",
    "event_planner": "Event Scheduler",
    "agenda_generator": "Agenda Formatter",
    "office_secretary": "Office Secretary",
}


def _now() -> str:
    """Current timestamp formatted for terminal logs."""
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _format_dict_preview(data: dict[str, Any], max_len: int = 140) -> str:
    """Format dictionary into a concise preview, truncating binary/large payloads."""
    clean = {}
    for k, v in data.items():
        if isinstance(v, (bytes, bytearray)):
            clean[k] = f"<{len(v)} bytes>"
        elif isinstance(v, str) and len(v) > 80:
            clean[k] = v[:77] + "..."
        elif isinstance(v, list) and len(v) > 3:
            clean[k] = f"[{len(v)} items: {v[0]}, ...]"
        else:
            clean[k] = v
    try:
        s = json.dumps(clean, ensure_ascii=False)
    except Exception:
        s = str(clean)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


class TerminalDemoLoggerPlugin(BasePlugin):
    """ADK Plugin that produces live demo-ready terminal logs."""

    def __init__(self, name: str = "terminal_demo_logger") -> None:
        super().__init__(name=name)
        self._agent_start_times: dict[str, float] = {}
        self._tool_start_times: dict[str, float] = {}

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> Optional[types.Content]:
        text_parts = []
        file_count = 0
        if hasattr(user_message, "parts") and user_message.parts:
            for p in user_message.parts:
                if hasattr(p, "text") and p.text:
                    text_parts.append(p.text)
                elif hasattr(p, "inline_data") or hasattr(p, "file_data"):
                    file_count += 1

        prompt = " ".join(text_parts).strip()
        if len(prompt) > 200:
            prompt = prompt[:197] + "..."

        session_id = getattr(invocation_context, "session_id", "session")
        print(
            f"\n{BOLD}{COLOR_ROOT}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓{RESET}"
        )
        print(
            f"{BOLD}{COLOR_ROOT}┃ 📩 USER REQUEST [{_now()}] (Session: {session_id[:8] if session_id else 'default'}){RESET}"
        )
        print(f"{BOLD}{COLOR_ROOT}┃{RESET}  {BOLD}Query:{RESET} {prompt or '(empty message)'}")
        if file_count > 0:
            print(f"{BOLD}{COLOR_ROOT}┃{RESET}  {COLOR_MUTED}Attachments: {file_count} media/file payload(s){RESET}")
        print(
            f"{BOLD}{COLOR_ROOT}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛{RESET}\n",
            flush=True,
        )
        return None

    async def before_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> Optional[types.Content]:
        name = getattr(agent, "name", "agent")
        color = AGENT_COLORS.get(name, COLOR_ROOT)
        label = AGENT_LABELS.get(name, name)
        model = getattr(agent, "model", None)
        model_info = f" ({model})" if model else ""

        self._agent_start_times[name] = time.time()
        print(
            f"{COLOR_MUTED}[{_now()}]{RESET} {BOLD}{color}🤖 [{label}]{RESET} Activated{COLOR_MUTED}{model_info}{RESET}...",
            flush=True,
        )
        return None

    async def after_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> Optional[types.Content]:
        name = getattr(agent, "name", "agent")
        color = AGENT_COLORS.get(name, COLOR_ROOT)
        label = AGENT_LABELS.get(name, name)

        start = self._agent_start_times.pop(name, None)
        elapsed_str = f" in {time.time() - start:.2f}s" if start else ""

        print(
            f"{COLOR_MUTED}[{_now()}]{RESET} {BOLD}{color}✅ [{label}]{RESET} Completed turn{COLOR_MUTED}{elapsed_str}{RESET}",
            flush=True,
        )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict[str, Any]]:
        tool_name = getattr(tool, "name", str(tool))
        self._tool_start_times[tool_name] = time.time()
        args_preview = _format_dict_preview(tool_args)

        print(
            f"{COLOR_MUTED}[{_now()}]{RESET}   {BOLD}{COLOR_TOOL}🔧 [TOOL CALL]{RESET} {BOLD}{tool_name}{RESET} -> {COLOR_MUTED}{args_preview}{RESET}",
            flush=True,
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        tool_name = getattr(tool, "name", str(tool))
        start = self._tool_start_times.pop(tool_name, None)
        elapsed_str = f" ({time.time() - start:.2f}s)" if start else ""

        if isinstance(result, dict):
            res_preview = _format_dict_preview(result, max_len=160)
        else:
            res_preview = str(result)
            if len(res_preview) > 160:
                res_preview = res_preview[:157] + "..."

        print(
            f"{COLOR_MUTED}[{_now()}]{RESET}   {BOLD}{COLOR_SUCCESS}✨ [TOOL RESULT]{RESET} {tool_name}{COLOR_MUTED}{elapsed_str}{RESET} => {COLOR_MUTED}{res_preview}{RESET}",
            flush=True,
        )
        return None

    async def on_agent_error_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
        error: Exception,
    ) -> None:
        name = getattr(agent, "name", "agent")
        print(
            f"{COLOR_MUTED}[{_now()}]{RESET} {BOLD}{COLOR_ERROR}❌ [AGENT ERROR - {name}]{RESET} {error}",
            flush=True,
        )

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> Optional[dict[str, Any]]:
        tool_name = getattr(tool, "name", str(tool))
        print(
            f"{COLOR_MUTED}[{_now()}]{RESET}   {BOLD}{COLOR_ERROR}❌ [TOOL ERROR - {tool_name}]{RESET} {error}",
            flush=True,
        )
        return None


class TerminalA2ALoggingMiddleware(BaseHTTPMiddleware):
    """Starlette middleware logging incoming A2A (Agent-to-Agent) RPC traffic."""

    def __init__(self, app: Any, service_name: str, color: str = COLOR_VIDEO) -> None:
        super().__init__(app)
        self.service_name = service_name
        self.color = color

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        # Ignore noisy internal browser probes / favicon
        path = request.url.path
        if path in ("/favicon.ico", "/robots.txt"):
            return await call_next(request)

        method = request.method
        start_time = time.time()

        # Log incoming A2A RPC request
        print(
            f"{COLOR_MUTED}[{_now()}]{RESET} {BOLD}{self.color}⚡ [A2A {self.service_name}]{RESET} 📥 {BOLD}{method}{RESET} {path}",
            flush=True,
        )

        try:
            response = await call_next(request)
            elapsed = time.time() - start_time
            status_color = COLOR_SUCCESS if response.status_code < 400 else COLOR_ERROR
            print(
                f"{COLOR_MUTED}[{_now()}]{RESET} {BOLD}{self.color}⚡ [A2A {self.service_name}]{RESET} 📤 {status_color}{response.status_code} {response.status_code}{RESET} ({elapsed:.3f}s)",
                flush=True,
            )
            return response
        except Exception as exc:
            elapsed = time.time() - start_time
            print(
                f"{COLOR_MUTED}[{_now()}]{RESET} {BOLD}{COLOR_ERROR}❌ [A2A {self.service_name} FAILED]{RESET} {method} {path} ({elapsed:.3f}s): {exc}",
                flush=True,
            )
            raise


terminal_demo_logger = TerminalDemoLoggerPlugin()
