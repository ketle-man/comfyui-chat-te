"""Builds Qwen2.5-VL chat prompts (DiffSynth-compatible template, matching
what ComfyUI's own Qwen-Image-Edit encoder nodes use) and parses tool-call
output back into an OpenAI-style `tool_calls` list.

Vision tokens use the convention documented for ComfyUI's Qwen2.5-VL
tokenizer: `<|vision_start|><|image_pad|><|vision_end|>` once per attached
image, placed immediately before that turn's text.
"""

import json
import re
import uuid

VISION_TOKENS = "<|vision_start|><|image_pad|><|vision_end|>"

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Qwen2.5-VL's own tokenizer treats these exact substrings as real special
# tokens (turn/vision boundaries), not ordinary text -- ComfyUI's
# clip.tokenize() matches them verbatim before regular BPE tokenization runs.
# Every piece of text this module embeds into the prompt (system prompt,
# message text, tool names/arguments, tool results, tool schemas) ultimately
# comes from the HTTP request body, so leaving these unescaped lets a
# message -- or a tool result relayed from some other, less-trusted, system
# -- forge fake <|im_start|>system / <|im_end|> boundaries and impersonate
# another role. Dropping just the pipes keeps the text readable (useful for
# debugging why something got flagged) while reliably breaking the
# tokenizer's exact-string special-token match.
_SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>")


def _sanitize(text):
    if not isinstance(text, str):
        return text
    for token in _SPECIAL_TOKENS:
        if token in text:
            text = text.replace(token, token.replace("|", ""))
    return text


def _tools_block(tools):
    if not tools:
        return ""
    fn_lines = "\n".join(_sanitize(json.dumps(t, ensure_ascii=False)) for t in tools)
    return (
        "\n\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{fn_lines}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    )


def _format_tool_calls(tool_calls):
    """Render an assistant message's OpenAI-style `tool_calls` back into
    Qwen's own `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
    blocks (one per call) so a prior tool invocation stays visible to the
    model on the next turn. `function.arguments` arrives JSON-*encoded* per
    the OpenAI spec (this is also what extract_tool_calls() below produces),
    so it's decoded back into an object before re-embedding -- Qwen's
    template expects `arguments` as a nested object, not a string-within-a-
    string."""
    blocks = []
    for call in tool_calls:
        fn = call.get("function") or {}
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = raw_args
        else:
            args = raw_args
        payload = _sanitize(json.dumps({"name": name, "arguments": args}, ensure_ascii=False))
        blocks.append(f"<tool_call>\n{payload}\n</tool_call>")
    return "\n".join(blocks)


def _message_text_and_images(message):
    """Normalize one OpenAI/Ollama-style message into (text, [image_ref, ...])."""
    content = message.get("content")
    images = list(message.get("images") or [])  # Ollama-style flat list, if present

    if isinstance(content, str):
        return content, images

    if isinstance(content, list):
        text_parts = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "image_url":
                url = (block.get("image_url") or {}).get("url", "")
                if url:
                    images.append(url)
        return "\n".join(text_parts), images

    return "", images


def build_prompt(messages, tools=None, system_prompt=None, append_no_think=False):
    """Turn a chat history into (prompt_text, image_refs).

    image_refs is an ordered list of raw image payloads (data URLs / bare
    base64 strings) in the same order their vision-token markers appear in
    prompt_text -- the caller decodes these into tensors before calling
    clip.tokenize(text, images=...).

    append_no_think: Qwen3(-VL) convention for suppressing its default
    "thinking" preamble is an explicit empty <think></think> block right
    after the assistant turn opens. Only meaningful for Qwen3-family models
    -- the caller (chat_engine.py) is responsible for only passing True when
    the loaded model actually supports thinking mode (see chat_state.py's
    supports_thinking_mode()); Qwen2.5-VL has no such convention.
    """
    pending_system = _sanitize(system_prompt) or DEFAULT_SYSTEM_PROMPT
    image_refs = []
    parts = []

    # Qwen's own tool-calling template merges one or more consecutive
    # `tool` result messages into a single user turn, each wrapped in its
    # own <tool_response> block, rather than emitting a separate turn per
    # result -- buffer them here and flush right before the next non-tool
    # message (or at the end).
    pending_tool_responses = []

    def flush_tool_responses():
        if not pending_tool_responses:
            return
        blocks = "\n".join(f"<tool_response>\n{c}\n</tool_response>" for c in pending_tool_responses)
        parts.append(f"<|im_start|>user\n{blocks}<|im_end|>\n")
        pending_tool_responses.clear()

    for message in messages:
        role = message.get("role", "user")
        if role == "system":
            pending_system = _sanitize(message.get("content")) or pending_system
            continue

        if role == "tool":
            text, _images = _message_text_and_images(message)
            pending_tool_responses.append(_sanitize(text))
            continue

        flush_tool_responses()

        text, images = _message_text_and_images(message)
        text = _sanitize(text)
        vision_prefix = VISION_TOKENS * len(images)
        image_refs.extend(images)

        if role == "assistant":
            tool_call_text = _format_tool_calls(message.get("tool_calls") or [])
            body = "\n".join(part for part in (text, tool_call_text) if part)
            parts.append(f"<|im_start|>assistant\n{body}<|im_end|>\n")
        else:
            parts.append(f"<|im_start|>user\n{vision_prefix}{text}<|im_end|>\n")

    flush_tool_responses()

    system_block = f"<|im_start|>system\n{pending_system}{_tools_block(tools)}<|im_end|>\n"
    prompt = system_block + "".join(parts) + "<|im_start|>assistant\n"
    if append_no_think:
        prompt += "<think>\n\n</think>\n\n"
    return prompt, image_refs


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(raw_text):
    """Drop completed `<think>...</think>` reasoning blocks from a response.
    Qwen3.5 reasons by default and emits the block as ordinary text (the tag
    is not a special token, so decode(skip_special_tokens=True) keeps it);
    left in, it fills the visible reply and eats the max_tokens budget before
    the answer. An *unterminated* block (generation hit max_tokens mid-
    thought) is left untouched so the truncation stays visible instead of
    producing an empty reply. A lone closing tag, which appears when the
    prompt itself already opened the block, drops everything before it."""
    if "</think>" not in raw_text:
        return raw_text
    if "<think>" not in raw_text:
        return raw_text.rsplit("</think>", 1)[1].lstrip()
    return _THINK_BLOCK_RE.sub("", raw_text)


def extract_tool_calls(raw_text):
    """Split a generated response into (visible_content, tool_calls_or_None)."""
    matches = _TOOL_CALL_RE.findall(raw_text)
    if not matches:
        return raw_text.strip(), None

    tool_calls = []
    for match in matches:
        try:
            payload = json.loads(match)
        except json.JSONDecodeError:
            continue
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": payload.get("name", ""),
                "arguments": json.dumps(payload.get("arguments", {}), ensure_ascii=False),
            },
        })

    visible = _TOOL_CALL_RE.sub("", raw_text).strip()
    return visible, (tool_calls or None)
