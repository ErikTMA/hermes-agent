"""Claude CLI client — a drop-in stand-in for ``anthropic.Anthropic()``.

Routes model calls through the Claude Code CLI (``claude --print``) instead of
calling ``api.anthropic.com`` directly.

Why this exists
---------------
Subscription (OAuth) auth is accepted from Claude Code itself. A direct SDK
call carrying the same token is refused with::

    400 invalid_request_error: You're out of extra usage.

which is the org-level overage rejection surfaced in the CLI's own
``rate_limit_event`` line (``overageDisabledReason: org_level_disabled``), not
an exhausted balance. Running the real, first-party ``claude`` binary is the
supported way to spend a subscription from an automated context; this module
does that and adapts its output back into the shapes the Anthropic SDK exposes.

What it implements
------------------
``client.messages.create(...)``  → a ``Message``
``client.messages.stream(...)``  → a context manager that yields SSE-style
                                   events and offers ``get_final_message()``

The CLI's ``--output-format stream-json --include-partial-messages`` mode emits
the upstream Anthropic events verbatim, wrapped as
``{"type": "stream_event", "event": {...}}``, so streaming is an unwrap rather
than a reimplementation.

Conversation continuity
-----------------------
Each turn would otherwise be a brand-new CLI session, re-sending the whole
history and throwing away prompt caching. Instead the conversation *prefix*
(everything before the newest user turn) is hashed and mapped to the CLI
session id it produced; the next turn resumes that session with ``--resume``
and sends only the new user message. The map is persisted on the data volume
(not in ``~/.hermes``, which is image-local), so continuity survives a restart.

A cache miss is harmless — it just starts a fresh session with the full
history rendered into the prompt.

Tools
-----
Hermes' agent loop cannot drive tools through this path — the CLI returns a
finished turn rather than raw ``tool_use`` blocks. The CLI is therefore given a
narrow allowlist of its own tools (see ``_DEFAULT_ALLOWED_TOOLS``) so the agent
can at least read and write in its working directory and record memories.
Override with ``HERMES_CLAUDE_CLI_ALLOWED_TOOLS``.

MCP servers are read from Hermes' own ``config.yaml`` (``mcp_servers:``), so
adding one is a config change rather than a code change, and each configured
server is allowlisted as ``mcp__<name>``. Offering a capability as an MCP tool
is preferred over widening ``Bash``: it stays a named capability instead of
arbitrary shell.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# A CLI turn can legitimately take a long time on Opus.
_CLI_TIMEOUT_SECONDS = int(os.getenv("HERMES_CLAUDE_CLI_TIMEOUT", "900"))

# Sessions unused for this long are dropped from the map.
_SESSION_TTL_SECONDS = int(os.getenv("HERMES_CLAUDE_CLI_SESSION_TTL", str(30 * 24 * 3600)))

# Tools the CLI may run on its own.
#
# Hermes' agent loop cannot drive tools through this path: the CLI returns a
# finished turn, never raw tool_use blocks for Hermes to execute. With no tools
# at all the agent can only talk — it cannot take a note, maintain its own
# CLAUDE.md, or record a memory, and it tends to misreport that as a sandbox
# permission error.
#
# The default is deliberately narrow: read/write within its working directory,
# plus the one shell command it needs to persist memory. It is NOT general
# shell access — this pod holds cluster RBAC, an SSH key to the kali workspace,
# and Infisical credentials. Widen it consciously, not by accident.
# Tools the CLI may run on its own, before MCP servers are added.
#
# Hermes' agent loop cannot drive tools through this path: the CLI returns a
# finished turn, never raw tool_use blocks for Hermes to execute. Anything the
# agent cannot do here it reports as being "blocked" or "denied permission",
# which reads like an environment fault and is not — it is simply an absent
# tool.
#
# Bash is unrestricted by owner decision. Be clear about what that means: the
# pod runs a privileged dind sidecar with DOCKER_HOST on loopback, so shell
# here reaches the node, not just the container. It also holds cluster RBAC, an
# SSH key to the kali workspace, and Infisical credentials for every project.
# Narrow it with HERMES_CLAUDE_CLI_ALLOWED_TOOLS if that trade stops being
# worth it.
_BASE_ALLOWED_TOOLS = os.getenv(
    "HERMES_CLAUDE_CLI_BASE_TOOLS", "Read,Write,Edit,Glob,Grep,Bash"
)

# The CLI inherits the gateway's cwd otherwise, which is not the agent's
# working directory and is not on persistent storage.
_WORKDIR = os.getenv("HERMES_CLAUDE_CLI_CWD", "/home/hermes/workspace")


def _mcp_config_for(servers: Dict[str, dict]) -> Optional[str]:
    """Materialise an MCP config for one turn, or None if there is nothing to wire."""
    override = os.getenv("HERMES_CLAUDE_CLI_MCP_CONFIG")
    if override:
        return override if os.path.exists(override) else None
    if not servers:
        return None
    try:
        fd, path = tempfile.mkstemp(prefix="hermes-mcp-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"mcpServers": servers}, fh)
        os.chmod(path, 0o600)
        return path
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not write MCP config (%s); continuing without it", exc)
        return None


def _plan_for(request: str, cached: Optional[List[str]]) -> Tuple[Dict[str, dict], List[str], Optional[dict], Optional[dict]]:
    """Capabilities plus the execution plan (model, effort, ultracode)."""
    from agent.capability_router import load_catalogue, select_plan, load_execution_policy

    catalogue = load_catalogue()
    if not catalogue:
        return {}, [], None, None
    caps, plan, record = select_plan(catalogue, request)
    # Capability grants are cached per conversation; the plan is not, because
    # difficulty varies turn to turn within one thread.
    if cached is not None:
        kept = {n: catalogue[n] for n in cached if n in catalogue}
        if kept:
            caps = kept
    return ({n: c.server for n, c in caps.items()}, sorted(caps), plan, record)


def _capabilities_for(request: str, cached: Optional[List[str]]) -> Tuple[Dict[str, dict], List[str], Optional[dict]]:
    """Pick this turn's capabilities.

    Returns (servers, granted_names, decision_record). ``cached`` is the grant
    from earlier in the same conversation: capability needs rarely change
    mid-thread, and re-running the router on every "ok, thanks" would add its
    latency to turns that need nothing.
    """
    from agent.capability_router import load_catalogue, select_capabilities

    catalogue = load_catalogue()
    if not catalogue:
        return {}, [], None

    if cached is not None:
        granted = {name: catalogue[name] for name in cached if name in catalogue}
        if granted:
            return ({n: c.server for n, c in granted.items()}, sorted(granted), None)

    caps, record = select_capabilities(catalogue, request)
    return ({n: c.server for n, c in caps.items()}, sorted(caps), record)


def _allowed_tools_for(granted: Sequence[str]) -> str:
    """Base tools plus one allowlist entry per granted capability."""
    override = os.getenv("HERMES_CLAUDE_CLI_ALLOWED_TOOLS")
    if override:
        return override
    tools = [_BASE_ALLOWED_TOOLS] + [f"mcp__{name}" for name in sorted(granted)]
    return ",".join(t for t in tools if t)

# The CLI inherits the gateway's cwd otherwise, which is not the agent's
# working directory and is not on persistent storage.
_WORKDIR = os.getenv("HERMES_CLAUDE_CLI_CWD", "/home/hermes/workspace")


# ---------------------------------------------------------------------------
# SDK-shaped response objects
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class TextBlock:
    text: str = ""
    type: str = "text"


@dataclass
class ThinkingBlock:
    thinking: str = ""
    type: str = "thinking"


@dataclass
class ToolUseBlock:
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)
    type: str = "tool_use"


@dataclass
class Message:
    id: str = ""
    type: str = "message"
    role: str = "assistant"
    content: list = field(default_factory=list)
    model: str = ""
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Usage = field(default_factory=Usage)


def _parse_content_block(block: dict):
    kind = block.get("type", "text")
    if kind == "text":
        return TextBlock(text=block.get("text", ""))
    if kind == "thinking":
        return ThinkingBlock(thinking=block.get("thinking", ""))
    if kind == "tool_use":
        return ToolUseBlock(
            id=block.get("id", ""),
            name=block.get("name", ""),
            input=block.get("input", {}) or {},
        )
    return block


class _Event:
    """Recursive attribute view over a parsed SSE event.

    The consumer reaches for ``event.type``, ``event.delta.type``,
    ``event.delta.text`` and ``event.content_block.type``; missing attributes
    must read as ``None`` rather than raising, because the consumer probes with
    ``getattr(..., None)``.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict):
        object.__setattr__(self, "_data", data or {})

    def __getattr__(self, name: str):
        value = self._data.get(name)
        if isinstance(value, dict):
            return _Event(value)
        if isinstance(value, list):
            return [_Event(v) if isinstance(v, dict) else v for v in value]
        return value

    def __repr__(self) -> str:
        return f"_Event({self._data!r})"


# ---------------------------------------------------------------------------
# Conversation -> CLI session mapping
# ---------------------------------------------------------------------------

def _state_dir() -> str:
    """Where this module keeps state that must outlive the container.

    ``~/.hermes`` is baked into the image: only specific subdirectories are
    symlinked onto the data volume, so a file written directly there is lost on
    every restart — taking conversation continuity and capability grants with
    it. Prefer the data volume when it is mounted.
    """
    override = os.getenv("HERMES_CLI_STATE_DIR")
    if override:
        return override
    for candidate in ("/data/hermes",):
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            return candidate
    return _hermes_home()


def _hermes_home() -> str:
    try:
        from utils import get_hermes_home  # type: ignore
        return str(get_hermes_home())
    except Exception:
        return os.path.expanduser(os.getenv("HERMES_HOME", "~/.hermes"))


class SessionMap:
    """Persistent conversation-prefix -> CLI session id map."""

    def __init__(self, path: Optional[str] = None):
        self._path = path or os.path.join(_state_dir(), "claude_cli_sessions.json")
        self._lock = threading.Lock()

    def _load(self) -> Dict[str, dict]:
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not read CLI session map (%s); starting empty", exc)
            return {}

    def _save(self, data: Dict[str, dict]) -> None:
        cutoff = time.time() - _SESSION_TTL_SECONDS
        pruned = {k: v for k, v in data.items() if v.get("updated_at", 0) >= cutoff}
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            tmp = f"{self._path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(pruned, fh)
            os.replace(tmp, self._path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not persist CLI session map (%s)", exc)

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            return self._load().get(key)

    def put(self, key: str, session_id: str, grant: Optional[Sequence[str]] = None) -> None:
        if not key or not session_id:
            return
        with self._lock:
            data = self._load()
            data[key] = {
                "session_id": session_id,
                "grant": list(grant or []),
                "updated_at": time.time(),
            }
            self._save(data)


def _text_of(content: Any) -> str:
    """Flatten a message's content into plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind == "thinking":
            continue
        elif kind == "tool_result":
            parts.append(f"[tool result: {json.dumps(block.get('content', ''))}]")
        elif kind == "tool_use":
            parts.append(f"[tool call: {block.get('name', '')}({json.dumps(block.get('input', {}))})]")
    return "\n".join(p for p in parts if p)


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n\n".join(parts)
    return ""


def _conversation_key(system: str, messages: list) -> str:
    """Stable fingerprint of a conversation prefix.

    Two different chats only collide if their entire history *and* system
    prompt are byte-identical, in which case resuming the same CLI session
    leaks nothing that was not already identical.
    """
    payload = json.dumps(
        {"system": system, "messages": messages},
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _render_conversation(messages: list) -> str:
    """Render a full history for a cold start (no session to resume)."""
    lines: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        text = _text_of(msg.get("content", ""))
        if not text:
            continue
        if role == "user":
            lines.append(f"User: {text}")
        elif role == "assistant":
            lines.append(f"Assistant: {text}")
    return "\n\n".join(lines)


def _last_user_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return _text_of(msg.get("content", ""))
    return ""


# ---------------------------------------------------------------------------
# CLI invocation
# ---------------------------------------------------------------------------

def _claude_binary() -> str:
    path = shutil.which("claude")
    if not path:
        raise FileNotFoundError(
            "Claude Code CLI not found in PATH. "
            "Install with: npm install -g @anthropic-ai/claude-code"
        )
    return path


class _Run:
    """One CLI invocation, exposing its stream-json lines as parsed dicts."""

    def __init__(self, *, model: str, system: str, prompt: str, resume: Optional[str],
                 servers: Optional[Dict[str, dict]] = None,
                 granted: Optional[Sequence[str]] = None,
                 plan: Optional[dict] = None):
        self.plan = plan or {}
        self.servers = servers or {}
        self.granted = list(granted or [])
        self.model = model
        self.session_id: Optional[str] = resume
        self.final_message: Optional[Message] = None
        self._system = system
        self._prompt = prompt
        self._resume = resume
        self._proc: Optional[subprocess.Popen] = None
        self._sys_file: Optional[str] = None
        self._stderr_tail: List[str] = []

    def _args(self) -> List[str]:
        args = [
            _claude_binary(),
            "--print",
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
        ]
        allowed = _allowed_tools_for(self.granted)
        if self.plan.get("ultracode"):
            # Workflow is what makes ultracode more than a keyword.
            allowed = f"{allowed},Workflow,Task"
        args += ["--allowedTools", allowed]
        effort = self.plan.get("effort")
        if effort:
            args += ["--effort", effort]
        mcp_config = _mcp_config_for(self.servers)
        if mcp_config:
            # --strict-mcp-config so only these servers are loaded, never a
            # stray user-level MCP config from the image or the volume.
            args += ["--mcp-config", mcp_config, "--strict-mcp-config"]
        model = self.plan.get("model") or self.model
        if model:
            args += ["--model", model]
        if self._resume:
            args += ["--resume", self._resume]
        if self._system:
            # NOT --system-prompt. Linux caps a single argv entry at
            # MAX_ARG_STRLEN (128 KiB), and the system prompt carries MEMORY.md,
            # USER.md and the skill descriptions — it crosses that on an ordinary
            # long-running conversation. exec() then fails with
            # `[Errno 7] Argument list too long`, which the retry loop treats as
            # a transient provider fault and retries three times with the same
            # oversized argv. The user sees "the model provider failed after
            # retries" and every retry is guaranteed to fail identically.
            fd, path = tempfile.mkstemp(prefix="hermes-sysprompt-", suffix=".md")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(self._system)
            self._sys_file = path
            args += ["--system-prompt-file", path]
        return args

    def start(self) -> None:
        args = self._args()
        logger.debug(
            "claude CLI: model=%s resume=%s prompt_chars=%d",
            self.model, self._resume or "-", len(self._prompt),
        )
        cwd = _WORKDIR if os.path.isdir(_WORKDIR) else None
        # The prompt goes on stdin for the same reason, and it is the half that
        # grows without bound: it carries the conversation. Nothing may be
        # passed positionally.
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=cwd,
        )
        # Written from a thread: a pipe holds 64 KiB and the prompt is routinely
        # larger, so a synchronous write would block until the child drains it.
        # The child does drain it before emitting output today, but that is its
        # behaviour rather than a guarantee, and the deadlock it would cause is
        # a silent hang rather than an error.
        def _feed() -> None:
            try:
                assert self._proc is not None and self._proc.stdin is not None
                self._proc.stdin.write(self._prompt)
                self._proc.stdin.close()
            except (BrokenPipeError, ValueError, AssertionError):
                # Child exited early; its stderr is the useful diagnostic.
                pass

        threading.Thread(target=_feed, name="claude-cli-stdin", daemon=True).start()

    def _cleanup_sys_file(self) -> None:
        if self._sys_file:
            try:
                os.unlink(self._sys_file)
            except OSError:
                pass
            self._sys_file = None

    def lines(self) -> Iterator[dict]:
        """Yield parsed JSON lines, tracking session id and final message."""
        assert self._proc is not None and self._proc.stdout is not None
        deadline = time.time() + _CLI_TIMEOUT_SECONDS
        try:
            for raw in self._proc.stdout:
                if time.time() > deadline:
                    self._proc.kill()
                    raise TimeoutError(f"Claude CLI exceeded {_CLI_TIMEOUT_SECONDS}s")
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                sid = data.get("session_id")
                if sid:
                    self.session_id = sid
                kind = data.get("type")
                if kind == "assistant":
                    self._absorb_assistant(data.get("message") or {})
                elif kind == "result" or "total_cost_usd" in data:
                    self._absorb_result(data)
                yield data
        finally:
            # Covers normal completion, the timeout kill, and an abandoned
            # generator. The system prompt is not secret, but it is the user's
            # memory — leaving copies in /tmp for every turn is both a slow leak
            # and a disclosure nobody asked for.
            self._cleanup_sys_file()

    def _absorb_assistant(self, message: dict) -> None:
        usage = message.get("usage") or {}
        self.final_message = Message(
            id=message.get("id", ""),
            model=message.get("model", self.model),
            content=[_parse_content_block(b) for b in message.get("content", []) if isinstance(b, dict)],
            stop_reason=message.get("stop_reason"),
            stop_sequence=message.get("stop_sequence"),
            usage=Usage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
                cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
            ),
        )

    def _absorb_result(self, data: dict) -> None:
        if self.final_message is None:
            text = data.get("result", "")
            self.final_message = Message(
                id=data.get("session_id", ""),
                model=self.model,
                content=[TextBlock(text=text)] if text else [],
            )
        # The result line carries the authoritative stop_reason and usage.
        if data.get("stop_reason"):
            self.final_message.stop_reason = data.get("stop_reason")
        elif self.final_message.stop_reason is None:
            self.final_message.stop_reason = "end_turn"
        usage = data.get("usage") or {}
        if usage:
            self.final_message.usage = Usage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
                cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
            )

    def finish(self) -> None:
        if self._proc is None:
            return
        stderr = ""
        try:
            if self._proc.stderr is not None:
                stderr = self._proc.stderr.read() or ""
        except Exception:
            pass
        code = self._proc.wait()
        if code != 0:
            raise RuntimeError(f"Claude CLI failed (exit {code}): {stderr.strip()[:500]}")
        if self.final_message is None:
            raise RuntimeError("Claude CLI produced no assistant message")


# ---------------------------------------------------------------------------
# SDK-shaped surface
# ---------------------------------------------------------------------------

class _StreamManager:
    """Context manager mirroring ``client.messages.stream(...)``."""

    def __init__(self, run: _Run, on_complete):
        self._run = run
        self._on_complete = on_complete
        # The consumer snapshots ``stream.response`` for diagnostics; there is
        # no httpx response behind a subprocess, and it probes defensively.
        self.response = None

    def __enter__(self) -> "_StreamManager":
        self._run.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self._run.finish()
            self._on_complete(self._run)
        else:
            try:
                if self._run._proc:
                    self._run._proc.kill()
            except Exception:
                pass
        return False

    def __iter__(self) -> Iterator[_Event]:
        for data in self._run.lines():
            if data.get("type") == "stream_event":
                event = data.get("event")
                if isinstance(event, dict):
                    yield _Event(event)

    def get_final_message(self) -> Optional[Message]:
        return self._run.final_message


class _Messages:
    def __init__(self, sessions: SessionMap):
        self._sessions = sessions

    def _prepare(self, *, model: str, messages: list, system: Any) -> _Run:
        system_text = _system_text(system)
        prefix = messages[:-1] if messages else []
        key = _conversation_key(system_text, prefix)
        entry = self._sessions.get(key)
        resume = entry.get("session_id") if entry else None
        cached_grant = entry.get("grant") if entry else None

        if resume:
            prompt = _last_user_text(messages)
            if not prompt:
                prompt = _render_conversation(messages)
        else:
            prompt = _render_conversation(messages) or _last_user_text(messages)

        # Route on the newest user turn — that is what states the intent.
        request = _last_user_text(messages) or prompt
        servers, granted, plan, record = _plan_for(request, cached_grant)
        if record:
            logger.info("Turn plan: %s", record)

        if plan and plan.get("ultracode"):
            # ultracode is keyword-triggered in the CLI, not a flag.
            prompt = f"ultracode\n\n{prompt}"

        run = _Run(
            model=model, system=system_text, prompt=prompt, resume=resume,
            servers=servers, granted=granted, plan=plan,
        )
        run._hermes_key_basis = (system_text, messages)  # type: ignore[attr-defined]
        return run

    def _remember(self, run: _Run) -> None:
        """Map the conversation *including* this reply to the CLI session."""
        basis = getattr(run, "_hermes_key_basis", None)
        if not basis or not run.session_id or run.final_message is None:
            return
        system_text, messages = basis
        reply = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": b.text}
                for b in run.final_message.content
                if isinstance(b, TextBlock)
            ],
        }
        self._sessions.put(
            _conversation_key(system_text, list(messages) + [reply]),
            run.session_id,
            run.granted,
        )

        # Hermes' memory tool never fires on this path, so persist durable facts
        # here instead. Backgrounded — the reply has already gone out.
        try:
            from agent.memory_writer import record_async
            reply_text = "\n".join(
                b.text for b in run.final_message.content if isinstance(b, TextBlock)
            )
            record_async(_last_user_text(messages), reply_text)
        except Exception as exc:  # pragma: no cover - never break a turn for this
            logger.debug("memory writer unavailable (%s)", exc)

    def stream(self, *, model: str, messages: list, system: Any = None, **kwargs) -> _StreamManager:
        run = self._prepare(model=model, messages=messages, system=system)
        return _StreamManager(run, self._remember)

    def create(self, *, model: str, messages: list, system: Any = None, **kwargs) -> Message:
        run = self._prepare(model=model, messages=messages, system=system)
        run.start()
        for _ in run.lines():
            pass
        run.finish()
        self._remember(run)
        return run.final_message  # type: ignore[return-value]


class ClaudeCliClient:
    """Stand-in for ``anthropic.Anthropic()`` backed by the Claude Code CLI."""

    def __init__(self):
        self._sessions = SessionMap()
        self.messages = _Messages(self._sessions)

    def close(self) -> None:
        return None


def build_claude_cli_client() -> ClaudeCliClient:
    _claude_binary()
    return ClaudeCliClient()
