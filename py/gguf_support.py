"""Optional GGUF loading support for Qwen2.5-VL and Qwen3-VL text encoders.

Chat TE does not implement its own GGUF reader. It reuses a `*ClipLoaderGGUF`
node class from an installed GGUF custom node pack -- city96's
`ComfyUI-GGUF` (registry key `CLIPLoaderGGUF`) and/or calcuis's `gguf`
(registry key `ClipLoaderGGUF`) -- accessed via ComfyUI's own node registry
(`nodes.NODE_CLASS_MAPPINGS`) -- the same public surface any workflow graph
uses to run that node -- rather than importing either pack's internal
modules directly. Both packages' directory names can contain characters that
make their `sys.modules` key (derived from the absolute install path)
unstable for another custom node to import from directly;
`NODE_CLASS_MAPPINGS` is the one interface ComfyUI guarantees stays put.
Which pack's node gets used (both can be installed side by side without
colliding, since the registry keys differ in capitalization) is decided in
_get_clip_loader_gguf_cls() below, based on the checkpoint's own GGUF
architecture field -- city96's node doesn't auto-merge a Qwen3-VL vision
tower, calcuis's does.

`*ClipLoaderGGUF.load_patcher()` calls `comfy.sd.load_text_encoder_state_dicts()`
internally, so the CLIP it returns supports `.tokenize()`/`.generate()`/
`.decode()` exactly like the regular safetensors path -- this module only
has to (a) locate that node class and drive it, and (b) pull the checkpoint's
"lm_head.weight" back out of its already-loaded state dict for chat_state.py's
lm_head patch (see the module docstring there for why that patch is needed at
all). If that tensor is still GGUF-quantized (e.g. a Q4_K_M output.weight that
wasn't promoted to F16 the way these packs promote huge embedding tables),
it's dequantized with the `gguf` PyPI package's own `gguf.quants.dequantize`
-- a dependency both GGUF packs already pull in, so this doesn't add a new one.
"""

import logging
import os
import re

import torch

logger = logging.getLogger(__name__)

# Mirrors city96/ComfyUI-GGUF's own strip_quant_suffix() (loader.py) -- used
# below to check a candidate mmproj filename actually corresponds to a given
# text encoder, the same way city96's own mmproj auto-detection does.
_QUANT_SUFFIX_RE = re.compile(r"[-_]?(?:ud-)?i?q[0-9]_[a-z0-9_\-]{1,8}$", re.IGNORECASE)


class GGUFSupportUnavailableError(RuntimeError):
    """Raised when a .gguf text encoder is selected but neither supported
    GGUF custom node pack is installed/enabled."""


def _detect_gguf_arch(path):
    """Read the "general.architecture" header field straight out of the GGUF
    file via the `gguf` PyPI package (already a hard dependency here -- see
    _dequantize_if_needed below) -- independent of either loader pack's own
    (private, differently-shaped) arch-detection code. Returns None if the
    field is missing or unreadable rather than raising, since this is only
    used to pick which loader to prefer."""
    try:
        import gguf

        reader = gguf.GGUFReader(path)
        field = reader.get_field("general.architecture")
        if field is None or len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
            return None
        return str(field.parts[field.data[-1]], encoding="utf-8")
    except Exception:
        logger.exception("Chat TE: failed to read GGUF architecture field (non-fatal)")
        return None


def _text_stem(path):
    """Text encoder filename with its quantization suffix stripped, e.g.
    "qwen3vl-4b-instruct-q4_k_m.gguf" -> "qwen3vl-4b-instruct". Mirrors
    city96/ComfyUI-GGUF's own strip_quant_suffix() (loader.py)."""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    match = _QUANT_SUFFIX_RE.search(stem)
    return stem[:match.start()] if match else stem


def _calcuis_would_pick_matching_mmproj(path):
    """True only if calcuis/gguf's own mmproj auto-detection (find_mmproj_pair
    in gguf_connector/mmj.py) would, if called right now, pick an mmproj file
    that actually corresponds to this text encoder.

    That function has no filename-relationship check at all -- it lists every
    "*mmproj*.gguf" file in the directory via a plain os.listdir() and
    unconditionally returns the first one, full stop. Confirmed two ways
    empirically: (1) with only a Qwen2.5-VL mmproj file present next to a
    Qwen3-VL text encoder, it silently merged that mismatched mmproj in
    ("Using mmproj '<qwen2.5-vl mmproj>' for '<qwen3-vl text encoder>'" in
    its own log line); (2) with both a Qwen2.5-VL mmproj and a
    filename-matching Qwen3-VL mmproj present, its choice depended on
    directory enumeration order, not correctness. A one-shot "does *some*
    matching mmproj exist somewhere in the folder" check (an earlier version
    of this function) doesn't catch case (2) -- calcuis could still pick the
    *wrong* one out of several candidates. So: reproduce its exact selection
    here (list the directory, take the first "*mmproj*.gguf" match) and check
    *that specific file*, not just "some matching file exists". Calling
    os.listdir() on the same directory right before handing off to calcuis
    sees the same filesystem state calcuis's own call will, so this predicts
    its choice reliably rather than merely hoping for the best."""
    root = os.path.dirname(path)
    txt_stem = _text_stem(path)

    try:
        candidates = os.listdir(root)
    except OSError:
        return False

    for fname in candidates:
        lname = fname.lower()
        if lname.endswith(".gguf") and "mmproj" in lname:
            # This is the first mmproj file find_mmproj_pair() would return
            # -- matching or not, it's the one that decides the outcome.
            return txt_stem in lname

    return False


def _get_clip_loader_gguf_cls(arch, path):
    """city96/ComfyUI-GGUF's `CLIPLoaderGGUF` only auto-merges a matching
    "*mmproj*" file for arch == "qwen2vl" (Qwen2.5-VL) -- confirmed by
    reading its loader.py, "qwen3vl" isn't in that check. calcuis/gguf's own
    `ClipLoaderGGUF` (different capitalization from city96's -- a distinct
    NODE_CLASS_MAPPINGS key, so both packs can be installed at once without
    colliding) merges mmproj for both arches, including Qwen3-VL's DeepStack
    vision tensors -- but see _calcuis_would_pick_matching_mmproj's docstring
    for why that merge isn't trusted blindly here. So: keep using city96's
    node (the one this bridge was originally verified against) for everything
    except arch == "qwen3vl" where calcuis would actually pick a matching
    mmproj file, where calcuis/gguf is used if installed; otherwise fall back
    to city96's loader -- the text encoder still loads, just without its
    vision tower merged in, rather than failing outright or merging the wrong
    one."""
    import nodes as comfy_nodes

    if arch == "qwen3vl":
        if not _calcuis_would_pick_matching_mmproj(path):
            logger.warning(
                "Chat TE: either no *mmproj*.gguf file matching this Qwen3-VL "
                "text encoder's filename was found next to it, or a "
                "non-matching one would be picked first -- skipping the "
                "'gguf' (calcuis) loader (it has no filename-relationship "
                "check of its own and would merge in whichever *mmproj* file "
                "happens to be first in the folder, vision tower mismatch "
                "and all) and loading text-only via city96/ComfyUI-GGUF "
                "instead. Make sure the matching mmproj file is the only one "
                "in this folder, or rename/remove the others, for vision "
                "support."
            )
        else:
            calcuis_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("ClipLoaderGGUF")
            if calcuis_cls is not None:
                return calcuis_cls
            logger.warning(
                "Chat TE: loading a Qwen3-VL GGUF text encoder without the 'gguf' "
                "(calcuis) custom node pack installed -- city96/ComfyUI-GGUF alone "
                "doesn't auto-merge its vision tower (a matching *mmproj* file), so "
                "this will likely load text-only. Install "
                "https://github.com/calcuis/gguf for full Qwen3-VL vision support."
            )

    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("CLIPLoaderGGUF")
    if cls is None:
        raise GGUFSupportUnavailableError(
            "This is a .gguf text encoder, which requires the 'ComfyUI-GGUF' "
            "custom node pack (city96/ComfyUI-GGUF) to be installed and enabled. "
            "Install it from ComfyUI Manager or "
            "https://github.com/city96/ComfyUI-GGUF, then restart ComfyUI."
        )
    return cls


def is_gguf_path(path: str) -> bool:
    return path.lower().endswith(".gguf")


def load_gguf_clip(path, clip_type):
    """Load a .gguf Qwen2.5-VL or Qwen3-VL text encoder via a GGUF CLIP
    loader node (see _get_clip_loader_gguf_cls for which one, and why).
    Returns (clip, lm_head_weight_or_None). The loader's own gguf_clip_loader
    logic already auto-detects and merges a matching "*mmproj*" GGUF file, so
    a split text/mmproj pair (e.g. Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf +
    Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf) is handled automatically as long
    as both files sit next to each other."""
    arch = _detect_gguf_arch(path)
    loader_cls = _get_clip_loader_gguf_cls(arch, path)
    loader = loader_cls()

    clip_data = loader.load_data([path])

    # Must extract before load_patcher(): comfy.sd.load_text_encoder_state_dicts()
    # pops "lm_head.weight" out of each state dict in-place (renaming it to
    # "model.lm_head.weight") as part of loading, so it's gone from clip_data
    # afterwards.
    lm_head_weight = None
    for sd in clip_data:
        tensor = sd.get("lm_head.weight")
        if tensor is not None:
            lm_head_weight = _dequantize_if_needed(tensor)
            break

    clip = loader.load_patcher([path], clip_type, clip_data)

    return clip, lm_head_weight


def _dequantize_if_needed(tensor):
    tensor_type = getattr(tensor, "tensor_type", None)
    if tensor_type is None:
        return tensor  # plain torch.Tensor already (e.g. F32/F16 in the gguf file)

    import gguf

    oshape = getattr(tensor, "tensor_shape", tensor.shape)
    logger.info(
        "Chat TE: dequantizing GGUF lm_head.weight (qtype=%s)",
        getattr(tensor_type, "name", tensor_type),
    )
    array = gguf.quants.dequantize(tensor.cpu().numpy(), tensor_type)
    return torch.from_numpy(array).reshape(oshape)
