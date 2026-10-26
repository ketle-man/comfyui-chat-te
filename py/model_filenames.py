"""Filename heuristics for picking out text encoders this bridge can drive.

Shared by routes.py (the panel's model dropdown) and chat_state.py (the
graph-node path's lm_head-file fallback) so the two don't drift out of sync.
"""

import re

# Matches "2.5"/"2_5"/"25" or "3" between "qwen" and "vl", with "-"/"_"/"."
# as optional separators (e.g. "qwen25VL_7b.safetensors" with no separator
# at all, "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" which is hyphen-separated, or
# "qwen3vl_8b_fp8_scaled.safetensors"). Deliberately excludes plain "Qwen3"
# (no "vl") -- that's a text-only checkpoint ComfyUI routes to a different,
# non-chat model class entirely (see chat_state.py module docstring).
_QWEN_VL_RE = re.compile(r"qwen[-_.]*(?:2[-_.]?5|3)[-_.]*vl", re.IGNORECASE)

# Qwen3.5 is natively multimodal, so its filenames carry no "vl" ("qwen3.5_9b",
# "Qwen3.5-9B-Q4_K_M.gguf", "qwen35_9b_fp8.safetensors"). The trailing lookahead
# keeps "qwen3.50..." style numbers and "qwen3_5x" from matching.
_QWEN35_RE = re.compile(r"qwen[-_.]*3[-_.]?5(?![0-9])", re.IGNORECASE)

# ComfyUI only implements the dense Qwen3.5 sizes (0.8B/2B/4B/9B/27B, see
# comfy/text_encoders/qwen35.py). The MoE releases ("Qwen3.5-35B-A3B", ...)
# have a different layer layout and would fail to load.
_QWEN35_MOE_RE = re.compile(r"moe|[-_.]a\d+(?:\.\d+)?b(?![a-z0-9])|35b", re.IGNORECASE)


def is_supported_qwen_vl_filename(name: str) -> bool:
    """True for Qwen2.5-VL, Qwen3-VL or (dense) Qwen3.5 checkpoint filenames.
    Excludes "*mmproj*" files -- those are vision-projector companions to a
    GGUF text encoder, auto-detected and loaded alongside it (see
    gguf_support.py), not something to be selected directly."""
    lname = name.lower()
    if "mmproj" in lname:
        return False
    if _QWEN_VL_RE.search(lname) is not None:
        return True
    return _QWEN35_RE.search(lname) is not None and _QWEN35_MOE_RE.search(lname) is None
