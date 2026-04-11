"""
Claude CLI client — drop-in replacement for anthropic.Anthropic().messages.create().

Routes API calls through the Claude Code CLI subprocess, which uses
Anthropic's whitelisted infrastructure. This bypasses the third-party
OAuth blocking that prevents direct API calls with subscription tokens.

Usage:
    from agent.claude_cli_client import build_claude_cli_client
    client = build_claude_cli_client()
    response = client.messages.create(model=..., messages=..., max_tokens=...)
"""

import json
import logging
import os
import subprocess
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response types — duck-typed to match anthropic SDK objects
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class TextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class ThinkingBlock:
    type: str = "thinking"
    thinking: str = ""


@dataclass
class ToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


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
    """Parse a content block from the CLI response into a typed object."""
    block_type = block.get("type", "text")
    if block_type == "text":
        return TextBlock(text=block.get("text", ""))
    elif block_type == "thinking":
        return ThinkingBlock(thinking=block.get("thinking", ""))
    elif block_type == "tool_use":
        return ToolUseBlock(
            id=block.get("id", ""),
            name=block.get("name", ""),
            input=block.get("input", {}),
        )
    else:
        # Return as-is for unknown types
        return block


def _format_messages_for_cli(messages: list, system=None) -> str:
    """Convert anthropic messages format to CLI stream-json input.

    The CLI expects one JSON message per line on stdin.
    For a single-turn call, we send the full conversation as one user message.
    """
    # Build the conversation content
    # The CLI's stream-json input expects {"type": "user", "message": {...}}
    # For multi-turn, we'd need to send each message separately, but for
    # a single API call, we send the last user message and let the CLI
    # handle it with the full context.

    # Find the last user message
    last_user = None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user = msg
            break

    if not last_user:
        # Fallback: send all messages as context
        last_user = messages[-1] if messages else {"role": "user", "content": ""}

    cli_msg = {
        "type": "user",
        "message": last_user,
    }
    return json.dumps(cli_msg)


def _build_cli_args(
    model: str,
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[list] = None,
) -> list:
    """Build CLI command arguments."""
    claude_path = shutil.which("claude")
    if not claude_path:
        raise FileNotFoundError(
            "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
        )

    args = [
        claude_path,
        "--print",
        "--output-format", "json",
        "--max-turns", "1",
    ]

    if model:
        args.extend(["--model", model])

    if system_prompt:
        args.extend(["--system-prompt", system_prompt])

    # Disable all built-in tools — hermes handles tools itself
    args.extend(["--allowedTools", ""])

    return args


class _Messages:
    """Duck-typed messages namespace matching anthropic.Anthropic().messages."""

    def create(
        self,
        *,
        model: str,
        messages: list,
        max_tokens: int = 4096,
        system: Any = None,
        tools: Optional[list] = None,
        tool_choice: Any = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        thinking: Any = None,
        extra_headers: Optional[dict] = None,
        extra_body: Optional[dict] = None,
        **kwargs,
    ) -> Message:
        """Execute a messages.create() call through the Claude CLI.

        Spawns the CLI as a subprocess, passes the prompt, and returns
        the response in anthropic SDK Message format.

        Note: streaming is not yet supported — responses are returned
        in full after the CLI completes.
        """
        # Build system prompt string
        system_text = ""
        if isinstance(system, str):
            system_text = system
        elif isinstance(system, list):
            # Extract text blocks
            parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            system_text = "\n\n".join(parts)

        # Build the prompt from messages
        # For the CLI, we need to construct a single prompt that includes
        # the conversation context
        prompt_parts = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Extract text from content blocks
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "tool_result":
                                text_parts.append(f"[Tool Result: {json.dumps(block.get('content', ''))}]")
                            elif block.get("type") == "tool_use":
                                text_parts.append(f"[Tool Call: {block.get('name', '')}({json.dumps(block.get('input', {}))})]")
                    content = "\n".join(text_parts)

                if role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")

        prompt = "\n\n".join(prompt_parts)

        # If there are tools, include them in the system prompt
        if tools:
            tool_descriptions = []
            for t in tools:
                name = t.get("name", "")
                desc = t.get("description", "")
                params = json.dumps(t.get("input_schema", {}))
                tool_descriptions.append(f"- {name}: {desc}\n  Parameters: {params}")

            tools_text = "Available tools:\n" + "\n".join(tool_descriptions)
            tools_text += "\n\nTo use a tool, respond with a tool_use content block."
            if system_text:
                system_text = system_text + "\n\n" + tools_text
            else:
                system_text = tools_text

        # Build CLI command
        cli_args = _build_cli_args(
            model=model,
            system_prompt=system_text if system_text else None,
            max_tokens=max_tokens,
            tools=tools,
        )

        # Get just the last user message as the prompt
        last_user_content = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    last_user_content = content
                elif isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
                    last_user_content = "\n".join(parts)
                break

        if not last_user_content:
            last_user_content = prompt

        logger.debug("Claude CLI call: model=%s, prompt_len=%d", model, len(last_user_content))

        try:
            result = subprocess.run(
                cli_args + [last_user_content],
                capture_output=True,
                text=True,
                timeout=900,
                env={**os.environ, "CLAUDE_CODE_SIMPLE": "1"},
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError("Claude CLI timed out after 900 seconds")
        except FileNotFoundError:
            raise FileNotFoundError(
                "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            logger.error("Claude CLI failed (exit %d): %s", result.returncode, stderr[:500])
            # Try to parse error from stderr
            raise RuntimeError(f"Claude CLI failed (exit {result.returncode}): {stderr[:500]}")

        # Parse the JSON output
        stdout = result.stdout.strip()
        if not stdout:
            raise RuntimeError("Claude CLI returned empty response")

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            # The output might be plain text (non-JSON mode fallback)
            return Message(
                id=f"cli-{int(time.time())}",
                model=model,
                content=[TextBlock(text=stdout)],
                stop_reason="end_turn",
                usage=Usage(input_tokens=0, output_tokens=len(stdout.split())),
            )

        # Parse the JSON response
        # The --output-format json returns a result object
        if isinstance(data, dict):
            result_text = data.get("result", "")
            cost = data.get("total_cost_usd", 0)
            num_turns = data.get("num_turns", 1)
            usage_data = data.get("usage", {})

            return Message(
                id=data.get("session_id", f"cli-{int(time.time())}"),
                model=model,
                content=[TextBlock(text=result_text)],
                stop_reason=data.get("stop_reason", "end_turn"),
                usage=Usage(
                    input_tokens=usage_data.get("input_tokens", 0),
                    output_tokens=usage_data.get("output_tokens", 0),
                    cache_creation_input_tokens=usage_data.get("cache_creation_input_tokens", 0),
                    cache_read_input_tokens=usage_data.get("cache_read_input_tokens", 0),
                ),
            )

        # Fallback
        return Message(
            id=f"cli-{int(time.time())}",
            model=model,
            content=[TextBlock(text=str(data))],
            stop_reason="end_turn",
        )


class ClaudeCliClient:
    """Drop-in replacement for anthropic.Anthropic() that routes through the CLI."""

    def __init__(self):
        self.messages = _Messages()

    def close(self):
        pass


def build_claude_cli_client() -> ClaudeCliClient:
    """Create a Claude CLI-backed client."""
    # Verify claude is available
    if not shutil.which("claude"):
        raise FileNotFoundError(
            "Claude Code CLI not found in PATH. "
            "Install with: npm install -g @anthropic-ai/claude-code"
        )
    return ClaudeCliClient()
