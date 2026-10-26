"""Runs Qwen2.5-VL / Qwen3-VL chat generation through ComfyUI's own
already-loaded CLIP.

This deliberately does not reconstruct a separate transformers model. ComfyUI's
CLIP object already exposes tokenize()/generate()/decode() (comfy/sd.py) backed
by the exact same weights used for text-to-image text encoding -- including
the vision tower, since CLIPType.QWEN_IMAGE loads it unconditionally (see
comfy/text_encoders/llama.py Qwen25_7BVLI.__init__, or comfy/text_encoders/
qwen3vl.py Qwen3VL.__init__). Routing chat through that same live object is
what actually avoids a second copy in VRAM: there is nothing to alias or
duplicate because it's the same object throughout.
"""

import base64
import io as _io
import random

import numpy as np
import torch
from PIL import Image

from . import chat_state
from .chat_template import build_prompt, extract_tool_calls, strip_thinking


def _ensure_prompt_server_progress_state():
    """comfy.utils.ProgressBar (used inside CLIP.generate()'s token loop) reads
    PromptServer.instance.last_prompt_id when no prompt_id is otherwise
    supplied. That attribute is only ever set by main.py's normal queue-worker
    loop, so on a freshly started server that has never executed a queued
    graph, calling generate() straight from an HTTP route raises
    AttributeError. last_node_id is initialized in PromptServer.__init__, but
    last_prompt_id is not -- this fills in that one gap."""
    try:
        from server import PromptServer

        if not hasattr(PromptServer.instance, "last_prompt_id"):
            PromptServer.instance.last_prompt_id = None
    except Exception:
        pass


def _decode_image_ref(ref) -> torch.Tensor:
    """Accepts a data: URL or a bare base64 string. Returns [1,H,W,C] float32 0-1."""
    if isinstance(ref, str) and ref.startswith("data:"):
        ref = ref.split(",", 1)[1]
    raw = base64.b64decode(ref)
    img = Image.open(_io.BytesIO(raw)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def run_chat(
    clip,
    messages,
    tools=None,
    system_prompt=None,
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.05,
    seed=None,
    enable_thinking=True,
):
    """Returns (visible_text, tool_calls_or_None).

    enable_thinking defaults to True -- i.e. by default nothing is appended
    to the prompt and the model is left to its own natural behavior. This
    was verified to matter in practice, not just in principle: on
    Qwen3-VL-4B (tied embeddings -- no independent lm_head weight, see
    chat_state.py), explicitly appending the empty <think></think> block
    used to force enable_thinking=False produced broken, echo-like output,
    while leaving the prompt alone (enable_thinking=True) generated
    correctly. Qwen3-VL-8B (real lm_head) handled the no-think block fine.
    Tying embeddings is a legitimate, lower-fidelity way to compute logits
    (see comfy/text_encoders/llama.py BaseQwen3.logits()), and that reduced
    fidelity apparently makes it more sensitive to this particular prompt
    shape -- so True is the safe default across both."""
    _ensure_prompt_server_progress_state()

    transformer = chat_state.find_transformer(clip)
    thinking_capable = transformer is not None and chat_state.supports_thinking_mode(transformer)
    prompt_text, image_refs = build_prompt(
        messages,
        tools=tools,
        system_prompt=system_prompt,
        append_no_think=thinking_capable and not enable_thinking,
    )
    images = [_decode_image_ref(ref) for ref in image_refs]

    if seed is None:
        # BaseGenerate.generate() always calls torch.Generator().manual_seed(seed),
        # even when do_sample=False, so this must never be None.
        seed = random.randint(0, 2**31 - 1)

    tokens = clip.tokenize(prompt_text, images=images)
    token_ids = clip.generate(
        tokens,
        do_sample=temperature > 0,
        max_length=max_tokens,
        temperature=max(temperature, 0.01),
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )
    raw_text = clip.decode(token_ids, skip_special_tokens=True)
    return extract_tool_calls(strip_thinking(raw_text))
