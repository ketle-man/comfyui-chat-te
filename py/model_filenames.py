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


def is_supported_qwen_vl_filename(name: str) -> bool:
    """True for Qwen2.5-VL or Qwen3-VL checkpoint filenames. Excludes
    "*mmproj*" files -- those are vision-projector companions to a GGUF text
    encoder, auto-detected and loaded alongside it (see gguf_support.py), not
    something to be selected directly."""
    lname = name.lower()
    if "mmproj" in lname:
        return False
    return _QWEN_VL_RE.search(lname) is not None
