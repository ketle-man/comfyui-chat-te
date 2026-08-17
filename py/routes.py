"""OpenAI-compatible HTTP bridge for the Chat TE backend.

Registered directly on ComfyUI's own PromptServer (same process, same port --
no separate server to run). `POST /v1/chat/completions` intentionally mirrors
the wire format ComfyUI-Workflow-Studio's AI TOOL chat pane already sends to
LM Studio / Lemonade-style backends, so pointing that panel's backend URL at
this ComfyUI instance works without changes on that side.
"""

import asyncio
import logging

from aiohttp import web

from .model_filenames import is_supported_qwen_vl_filename

logger = logging.getLogger(__name__)


def setup_routes(app: web.Application):
    app.router.add_get("/chat_te/text_encoders", handle_list_text_encoders)
    app.router.add_get("/chat_te/status", handle_status)
    app.router.add_post("/chat_te/load", handle_load)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_models)


async def handle_list_text_encoders(request: web.Request) -> web.Response:
    import folder_paths

    names = list(folder_paths.get_filename_list("text_encoders"))
    try:
        names += folder_paths.get_filename_list("clip_gguf")
    except Exception:
        pass  # ComfyUI-GGUF not installed -- no .gguf files registered under this key
    names = sorted(set(n for n in names if is_supported_qwen_vl_filename(n)))
    return web.json_response({"text_encoders": names})


async def handle_status(request: web.Request) -> web.Response:
    from .chat_state import STATE

    return web.json_response({
        "loaded": STATE.is_loaded(),
        "model_name": STATE.model_name,
        "source": STATE.source,
    })


async def handle_load(request: web.Request) -> web.Response:
    from . import gguf_support
    from .chat_state import STATE, ChatBackendUnsupportedError

    try:
        body = await request.json()
        model_name = body.get("model_name")
        if not model_name:
            return web.json_response({"error": "model_name is required"}, status=400)

        import comfy.sd
        import folder_paths

        path = folder_paths.get_full_path("text_encoders", model_name)
        if not path:
            return web.json_response({"error": f"text encoder not found: {model_name}"}, status=404)

        def _load():
            if gguf_support.is_gguf_path(path):
                return gguf_support.load_gguf_clip(path, comfy.sd.CLIPType.QWEN_IMAGE)
            clip = comfy.sd.load_clip(
                ckpt_paths=[path],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
            )
            return clip, None  # lm_head, if missing, is read from disk by chat_state.py itself

        clip, lm_head_weight = await asyncio.to_thread(_load)
        STATE.set_clip(clip, model_name=model_name, source="panel", model_path=path, lm_head_weight=lm_head_weight)
        return web.json_response({"status": "ok", "model_name": model_name})
    except (ChatBackendUnsupportedError, gguf_support.GGUFSupportUnavailableError) as e:
        logger.warning("Chat TE: unsupported ComfyUI install: %s", e)
        return web.json_response({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception("Chat TE: failed to load model")
        return web.json_response({"error": str(e)}, status=500)


async def handle_models(request: web.Request) -> web.Response:
    from .chat_state import STATE

    model_id = STATE.model_name or "qwen2.5-vl-chat-te"
    return web.json_response({"object": "list", "data": [{"id": model_id, "object": "model"}]})


async def handle_chat_completions(request: web.Request) -> web.Response:
    from .chat_engine import run_chat
    from .chat_state import STATE

    if not STATE.is_loaded():
        return web.json_response(
            {"error": {"message": "No chat model loaded. Load a Qwen2.5-VL text encoder from the Chat TE panel first."}},
            status=400,
        )

    try:
        body = await request.json()
        messages = body.get("messages", [])
        tools = body.get("tools")
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 512
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        seed = body.get("seed")
        # Default True -- see chat_engine.run_chat()'s docstring for why
        # (explicitly suppressing thinking mode broke generation on at least
        # one real model, Qwen3-VL-4B).
        enable_thinking = bool(body.get("enable_thinking", True))

        def _run():
            with STATE.lock:
                return run_chat(
                    STATE.clip,
                    messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    enable_thinking=enable_thinking,
                )

        content, tool_calls = await asyncio.to_thread(_run)

        message = {"role": "assistant", "content": content}
        finish_reason = "stop"
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"

        return web.json_response({
            "id": "chatcmpl-chat-te",
            "object": "chat.completion",
            "model": STATE.model_name or "qwen2.5-vl-chat-te",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        })
    except Exception as e:
        logger.exception("Chat TE: generation failed")
        return web.json_response({"error": {"message": str(e)}}, status=500)
