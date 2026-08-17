"""Chat TE -- reuses ComfyUI's already-loaded Qwen2.5-VL text encoder for
multi-turn, vision-capable chat instead of loading a separate LLM.

The chat backend is ComfyUI's own CLIP.tokenize()/generate()/decode()
(comfy/sd.py) applied to the same object a Qwen-Image-Edit workflow already
loaded via CLIPType.QWEN_IMAGE -- there is no separate model to keep in sync,
so VRAM usage does not increase beyond whatever that CLIP already costs.
"""

import logging

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .nodes import QwenChatBridgeLoader

WEB_DIRECTORY = "./web"


def _register_http_routes():
    try:
        from server import PromptServer

        from .py.routes import setup_routes

        setup_routes(PromptServer.instance.app)
    except Exception:
        logging.getLogger(__name__).exception("Chat TE: failed to register HTTP routes")


_register_http_routes()


class ChatTEExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [QwenChatBridgeLoader]


async def comfy_entrypoint() -> ChatTEExtension:
    return ChatTEExtension()
