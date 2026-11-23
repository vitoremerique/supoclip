"""Run SupoClip's AI steps through the Claude Code CLI (`claude -p`).

`LLM=claude-code:sonnet` (or `:opus`, `:haiku`) sends each analysis request
to the locally installed Claude Code, which signs in with the user's Claude
subscription (`CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`). Usage
counts against that plan's limits. Structured outputs are requested as a
JSON object matching the schema and validated by pydantic-ai as usual.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings

CLAUDE_CODE_TIMEOUT_SECONDS = int(os.getenv("CLAUDE_CODE_TIMEOUT_SECONDS", "900"))
_PERSISTENT_BIN = "/app/uploads/.claude-code/claude"


class ClaudeCodeError(RuntimeError):
    """The CLI failed or returned no usable answer."""


def claude_code_binary() -> str:
    configured = os.getenv("CLAUDE_CODE_BIN", "").strip()
    for candidate in (configured, _PERSISTENT_BIN, shutil.which("claude") or ""):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise ClaudeCodeError(
        "Claude Code CLI not found. Install it in the worker (CLAUDE_CODE_BIN or "
        f"{_PERSISTENT_BIN})."
    )


def _extract_json(text: str) -> str:
    """The JSON object in a reply, without code fences or surrounding prose."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


@dataclass(init=False)
class ClaudeCodeModel(Model):
    _model_name: str = field(repr=True)

    def __init__(self, model_name: str = "sonnet") -> None:
        super().__init__(
            profile=ModelProfile(
                supports_json_schema_output=True,
                default_structured_output_mode="native",
            )
        )
        self._model_name = model_name or "sonnet"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return "anthropic"

    def _build_prompt(
        self, messages: list[ModelMessage], params: ModelRequestParameters
    ) -> tuple[str, str]:
        system_parts: list[str] = []
        conversation: list[str] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        system_parts.append(part.content)
                    elif isinstance(part, UserPromptPart):
                        content = part.content
                        if not isinstance(content, str):
                            content = "\n".join(c for c in content if isinstance(c, str))
                        conversation.append(content)
                    elif isinstance(part, RetryPromptPart):
                        conversation.append(
                            "Your previous answer was rejected:\n" + part.model_response()
                        )
                if getattr(message, "instructions", None):
                    system_parts.append(message.instructions)
            elif isinstance(message, ModelResponse):
                text = "".join(p.content for p in message.parts if isinstance(p, TextPart))
                if text:
                    conversation.append("Your previous answer:\n" + text)

        prompt = "\n\n".join(conversation)
        output = params.output_object
        if output is not None:
            prompt += (
                "\n\nAnswer with ONLY one JSON object (no prose, no code fences) that "
                "matches this JSON schema exactly:\n" + json.dumps(output.json_schema)
            )
        return "\n\n".join(system_parts), prompt

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        model_settings, model_request_parameters = self.prepare_request(
            model_settings, model_request_parameters
        )
        system_prompt, prompt = self._build_prompt(messages, model_request_parameters)
        command = [
            claude_code_binary(),
            "-p",
            "--output-format", "json",
            "--model", self._model_name,
            # Pure text generation: no file, shell or web tools.
            "--tools", "",
        ]
        if system_prompt:
            command += ["--system-prompt", system_prompt]

        # An empty working directory keeps project files and CLAUDE.md out of the prompt.
        with tempfile.TemporaryDirectory(prefix="claude-code-") as cwd:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(prompt.encode("utf-8")),
                    timeout=CLAUDE_CODE_TIMEOUT_SECONDS,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                process.kill()
                await process.wait()
                raise

        try:
            payload: dict[str, Any] = json.loads(stdout.decode("utf-8", "replace"))
        except ValueError:
            raise ClaudeCodeError(
                f"Claude Code exited {process.returncode}: "
                + stderr.decode("utf-8", "replace")[-500:]
            ) from None
        text = str(payload.get("result") or "")
        if payload.get("is_error") or process.returncode != 0 or not text:
            raise ClaudeCodeError(f"Claude Code error: {text[-500:] or payload.get('subtype')}")

        if model_request_parameters.output_object is not None:
            text = _extract_json(text)
        return ModelResponse(parts=[TextPart(content=text)], model_name=self._model_name)
