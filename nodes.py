"""Graph nodes for the Chat TE bridge."""

from comfy_api.latest import io

from .py.chat_state import STATE


class QwenChatBridgeLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenChatBridgeLoader",
            display_name="Qwen Chat Bridge (Register CLIP)",
            category="chat_te",
            description=(
                "Registers an already-loaded Qwen2.5-VL, Qwen3-VL or Qwen3.5 CLIP as the active "
                "backend for the Chat TE sidebar panel and the /v1/chat/completions "
                "endpoint. Passes the CLIP through unchanged -- no extra copy of the "
                "model is created, so VRAM usage does not increase."
            ),
            inputs=[io.Clip.Input("clip")],
            outputs=[io.Clip.Output("CLIP")],
        )

    @classmethod
    def execute(cls, clip):
        STATE.set_clip(clip, model_name="graph-connected CLIP", source="graph")
        return io.NodeOutput(clip)
