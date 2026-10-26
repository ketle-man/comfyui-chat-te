"""Process-wide state for the Chat TE bridge.

Holds a reference to whichever ComfyUI CLIP object is currently acting as the
chat backend, plus a lock that serializes chat generation against concurrent
model loads/swaps. Generation itself is not serialized against ComfyUI's own
prompt queue here -- both share the same underlying weights and ComfyUI's own
model management handles device placement, but running a chat generation
while a heavy image-generation graph is mid-execution on a VRAM-constrained
GPU can still contend for memory. Keep that in mind when testing on small
VRAM budgets.

Two gaps in ComfyUI's own Qwen2.5-VL support (comfy/text_encoders/llama.py,
verified against a real install) are patched here, once, right after a CLIP
is registered. Both patches are no-ops if the thing they'd add is already
present, so applying them unconditionally to every loaded CLIP (Qwen3-VL
included) is safe -- Qwen3-VL's own config already sets `stop_tokens` and
already builds a real `lm_head` (see comfy/text_encoders/llama.py
Qwen3_8BConfig / Llama2_.__init__), so neither patch actually does anything
there; only Qwen25_7BVLI has the gaps described below.

1. `Qwen25_7BVLI_Config` never sets `stop_tokens`, unlike every sibling
   Qwen3/Gemma config in the same file -- `BaseGenerate.generate()` reads
   `self.model.config.stop_tokens` unconditionally and raises AttributeError
   without this.
2. `Qwen25_7BVLI.__init__` never builds an `lm_head` (the line is present but
   commented out with `# todo: should this be tied or not?`), so
   `BaseGenerate.logits()` falls back to reusing `embed_tokens.weight` as the
   output projection. For Qwen2.5-VL-7B that fallback produces garbage --
   embed_tokens and lm_head are *not* tied for this model (confirmed: the
   checkpoint's own `lm_head.weight` differs from `model.embed_tokens.weight`,
   same shape [152064, 3584], both real BF16 tensors). Loading the real
   lm_head fixes generation quality.

Qwen2.5-VL (QwenImageTEModel) and Qwen3-VL (Qwen3VLTEModel) are both
comfy/sd1_clip.py `SD1ClipModel` subclasses that store their one actual
language-model sub-module under a *dynamically* named attribute -- whatever
string was passed as `name=` to `SD1ClipModel.__init__` ("qwen25_7b" for
Qwen2.5-VL, "qwen3vl_8b"/"qwen3vl_4b" for Qwen3-VL), recorded on the instance
as `self.clip`. `find_transformer()` below reads that attribute name back
instead of assuming a fixed one, so both architectures resolve the same way.
"""

import logging
import threading

import torch

logger = logging.getLogger(__name__)

# <|endoftext|>, <|im_end|> -- Qwen2/2.5 tokenizer family.
QWEN25_VL_STOP_TOKENS = [151643, 151645]

# The ComfyUI install this node was developed and verified against
# (comfyanonymous/ComfyUI, confirmed via `git log`). Native CLIP.generate()
# support for text encoders is a recent addition, so this is quoted in the
# error message below as a known-working reference point, not asserted as a
# hard minimum -- older commits may or may not have it.
_VERIFIED_COMFYUI_REF = "ComfyUI 0.42.0 (2026-10-19)"


class ChatBackendUnsupportedError(RuntimeError):
    """Raised when a loaded CLIP doesn't support native chat generation --
    typically an older ComfyUI install that predates CLIP.generate()."""


def _check_generate_support(clip):
    """Returns an error message string if this CLIP can't be used as a chat
    backend, or None if it looks usable. Deliberately checked once at
    registration time (set_clip) rather than left to fail on the first chat
    message, so the panel's Load button / the graph node report the problem
    immediately and clearly."""
    if not callable(getattr(clip, "generate", None)):
        return (
            "This ComfyUI installation's CLIP object has no generate() method. "
            "Native text generation from a loaded text encoder is a recent "
            f"ComfyUI addition (verified working on {_VERIFIED_COMFYUI_REF}) -- "
            "please update ComfyUI."
        )
    if not callable(getattr(clip, "decode", None)):
        return (
            "This ComfyUI installation's CLIP object has no decode() method "
            "(needed to turn generated tokens back into text). Please update "
            f"ComfyUI (verified working on {_VERIFIED_COMFYUI_REF})."
        )

    transformer = find_transformer(clip)
    if transformer is None:
        return (
            "Could not find a supported Qwen language model inside this CLIP "
            "(expected clip.cond_stage_model.<name>.transformer, where <name> "
            "is cond_stage_model.clip -- e.g. Qwen2.5-VL's 'qwen25_7b' or "
            "Qwen3-VL's 'qwen3vl_8b'/'qwen3vl_4b'). Make sure the loaded text "
            "encoder is a Qwen2.5-VL, Qwen3-VL or Qwen3.5 model loaded with "
            "CLIPType.QWEN_IMAGE."
        )
    if not callable(getattr(transformer, "generate", None)):
        return (
            "The loaded Qwen model has no generate() method (missing "
            f"BaseGenerate support). Please update ComfyUI (verified working "
            f"on {_VERIFIED_COMFYUI_REF})."
        )

    return None


def find_transformer(clip):
    """Return the underlying language-model torch.nn.Module for this CLIP, or
    None. See the module docstring for why the sub-model attribute name has
    to be read back from `cond_stage_model.clip` rather than assumed."""
    cond_stage_model = getattr(clip, "cond_stage_model", None)
    if cond_stage_model is None:
        return None

    clip_attr_name = getattr(cond_stage_model, "clip", None)
    if isinstance(clip_attr_name, str):
        sub_model = getattr(cond_stage_model, clip_attr_name, None)
        transformer = getattr(sub_model, "transformer", None)
        if transformer is not None:
            return transformer

    # Fallback for any CLIP object that doesn't follow the SD1ClipModel
    # dynamic-sub-model pattern above.
    transformer = getattr(cond_stage_model, "transformer", None)
    if transformer is not None:
        return transformer

    return None


def supports_thinking_mode(transformer) -> bool:
    """Qwen3 (including Qwen3-VL) defaults to "thinking mode" -- a
    <think>...</think> block emitted before the real answer -- and relies on
    an explicit empty think block in the prompt to suppress it. Qwen2.5-VL
    has no such convention, so unconditionally appending that block there
    would just be confusing, unrequested-looking text. Distinguished via the
    model's own config class name: every Qwen3(-VL) config in
    comfy/text_encoders/llama.py is named "Qwen3...Config"; Qwen25_7BVLI's is
    not."""
    config = getattr(getattr(transformer, "model", None), "config", None)
    return "Qwen3" in type(config).__name__ if config is not None else False


def _guess_qwen_text_encoder_path():
    """Best-effort fallback for the graph-node path, where we're only handed a
    CLIP object and don't know which file it came from. Mirrors the lookup
    ComfyUI-QwenImageWanBridge's prompt-expander node uses for the same gap."""
    try:
        import folder_paths

        from .model_filenames import is_supported_qwen_vl_filename

        for name in folder_paths.get_filename_list("text_encoders"):
            if is_supported_qwen_vl_filename(name):
                return folder_paths.get_full_path("text_encoders", name)
    except Exception:
        pass
    return None


def _patch_stop_tokens(transformer):
    config = getattr(getattr(transformer, "model", None), "config", None)
    if config is None:
        return
    if not getattr(config, "stop_tokens", None):
        config.stop_tokens = QWEN25_VL_STOP_TOKENS
        logger.info("Chat TE: patched missing stop_tokens onto %s", type(config).__name__)


def _load_lm_head_tensor(model_path):
    import os
    import json

    from safetensors import safe_open

    if model_path is None:
        return None

    if os.path.isfile(model_path) and model_path.endswith(".safetensors"):
        with safe_open(model_path, framework="pt", device="cpu") as f:
            if "lm_head.weight" in f.keys():
                return f.get_tensor("lm_head.weight")
        return None

    if os.path.isdir(model_path):
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)
            shard = index.get("weight_map", {}).get("lm_head.weight")
            if shard:
                with safe_open(os.path.join(model_path, shard), framework="pt", device="cpu") as f:
                    return f.get_tensor("lm_head.weight")

    return None


def _resolve_lm_head_device_dtype(inner_model):
    """Device/dtype to build the new lm_head Linear with. Normally just taken
    from any existing parameter -- but for a GGUF-quantized model that
    parameter can be a GGMLTensor whose raw packed dtype (e.g. torch.uint8)
    would silently corrupt the real (already-dequantized) lm_head weight if
    used as a cast target. ComfyUI-GGUF always keeps large embedding tables
    dequantized to fp16 (see gguf_clip_loader's temb_key promotion), so
    embed_tokens.weight is a safe fallback reference for dtype specifically."""
    reference_param = next(inner_model.parameters())
    device = reference_param.device
    dtype = reference_param.dtype
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        embed = getattr(inner_model, "embed_tokens", None)
        embed_weight = getattr(embed, "weight", None)
        if embed_weight is not None and embed_weight.dtype in (torch.float16, torch.bfloat16, torch.float32):
            dtype = embed_weight.dtype
        else:
            dtype = torch.float16
    return device, dtype


def _patch_lm_head(transformer, model_path, lm_head_weight=None):
    inner_model = getattr(transformer, "model", None)
    if inner_model is None or getattr(inner_model, "lm_head", None) is not None:
        return  # already has one (or nothing to attach it to)

    config = getattr(inner_model, "config", None)
    if config is not None and type(config).__name__ == "Qwen35Config":
        # Qwen3.5 (comfy/text_encoders/qwen35.py) builds its own lm_head for the
        # 9B/27B sizes and ties embeddings by design for 0.8B/2B/4B, so there is
        # nothing to patch. This also couldn't work: layer 0 is a linear-attention
        # layer with no `self_attn` for the Linear-class lookup below.
        return

    weight = lm_head_weight
    if weight is None:
        if model_path is None:
            model_path = _guess_qwen_text_encoder_path()
        weight = _load_lm_head_tensor(model_path)
    if weight is None:
        logger.warning(
            "Chat TE: no lm_head.weight found (model_path=%s) -- generation will fall "
            "back to ComfyUI's tied-embedding output. This is low quality for models "
            "that don't actually tie embeddings (known issue for Qwen2.5-VL-7B); for "
            "models that do tie them by design (e.g. Qwen3-VL-4B), it's expected and fine.",
            model_path,
        )
        return

    # Reuse whichever comfy.ops Linear variant the rest of the model was built
    # with (disable_weight_init vs manual_cast, or GGMLOps.Linear for a GGUF
    # checkpoint) so BaseGenerate.logits()'s `module.comfy_cast_weights` check
    # behaves the same way it does for every other Linear layer in this model,
    # instead of guessing an ops class.
    first_layer = inner_model.layers[0]
    linear_cls = type(first_layer.self_attn.q_proj)
    device, dtype = _resolve_lm_head_device_dtype(inner_model)

    hidden_size, vocab_size = weight.shape[1], weight.shape[0]
    lm_head = linear_cls(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)
    lm_head.weight = torch.nn.Parameter(weight.to(device=device, dtype=dtype), requires_grad=False)

    inner_model.lm_head = lm_head
    logger.info("Chat TE: attached real lm_head.weight %s (%s)", tuple(weight.shape), linear_cls.__name__)


class ChatState:
    def __init__(self):
        self.lock = threading.Lock()
        self.clip = None
        self.model_name = None
        self.source = None  # "graph" | "panel"

    def set_clip(self, clip, model_name=None, source="panel", model_path=None, lm_head_weight=None):
        with self.lock:
            problem = _check_generate_support(clip)
            if problem:
                raise ChatBackendUnsupportedError(problem)

            transformer = find_transformer(clip)
            if transformer is not None:
                try:
                    _patch_stop_tokens(transformer)
                except Exception:
                    logger.exception("Chat TE: failed to patch stop_tokens (non-fatal)")
                try:
                    _patch_lm_head(transformer, model_path, lm_head_weight=lm_head_weight)
                except Exception:
                    logger.exception("Chat TE: failed to patch lm_head (non-fatal)")
            self.clip = clip
            self.model_name = model_name
            self.source = source

    def is_loaded(self) -> bool:
        return self.clip is not None


STATE = ChatState()
