"""Thread-safety guard for ComfyUI's model manager (comfy.model_management).

Why this exists
---------------
ComfyUI assumes only its prompt-worker thread touches
``comfy.model_management.current_loaded_models``; there is no lock around it.
Chat TE breaks that assumption: ``/v1/chat/completions`` runs
``CLIP.generate()`` from an aiohttp ``asyncio.to_thread`` worker, and
``generate()`` calls ``load_models_gpu()`` / ``free_memory()`` on that thread.

When that overlaps with the prompt worker (a queued prompt, a ``POST /free``,
or the worker's periodic ``gc.collect()`` that fires ``cleanup_models``
finalizers), ``free_memory()`` pops entries by indices computed before the
other thread inserted/removed entries. The wrong entry gets popped and an
already-unloaded ``LoadedModel`` (``real_model = None``) stays in the list.
The next ``is_dead()`` / ``cleanup_models()`` then raises
``TypeError: 'NoneType' object is not callable`` -- and when that happens in
``prompt_worker``'s ``unload_all_models()`` the worker thread dies, so every
later prompt sits in the queue forever until ComfyUI is restarted
(observed 2026-11-10 with Comic Creator's OCR tab: Chat TE full-read, then
a MOSS-Image generation that never ran).

What it does
------------
1. Serializes every function that mutates ``current_loaded_models``
   (``load_models_gpu``, ``free_memory``, ``unload_all_models``,
   ``cleanup_models_gc``) with one re-entrant lock shared by all threads.
2. Replaces the ``cleanup_models`` finalizer with a version that never blocks
   (finalizers can fire on any thread, including the aiohttp event loop): if
   the lock is busy it defers the cleanup to the next guarded call.
3. Makes ``LoadedModel.is_dead()`` and the cleanup tolerate a ``None``
   ``real_model`` instead of raising, and -- only from the outermost guarded
   call, when no ``free_memory()`` can be mid-way -- drops entries that were
   already unloaded but left in the list, so a corrupted list heals instead of
   killing the prompt worker.

Everything is applied once, at import time, and is a no-op if the expected
ComfyUI internals are missing (older/newer ComfyUI): Chat TE still works,
just without the guard.
"""

import functools
import logging
import threading

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_TLS = threading.local()
_pending_cleanup = False
_installed = False

_GUARDED = ("load_models_gpu", "free_memory", "unload_all_models", "cleanup_models_gc")


def _real_model_alive(lm):
    """real_model() result, or None if real_model was cleared (unloaded) / is not a weakref."""
    rm = getattr(lm, "real_model", None)
    if rm is None or not callable(rm):
        return None
    try:
        return rm()
    except Exception:
        return None


def _cleanup_dead(mm):
    """Original cleanup_models(): drop entries whose underlying model was garbage-collected.
    Entries whose real_model was cleared are skipped here (see _drop_unloaded)."""
    lst = mm.current_loaded_models
    for i in range(len(lst) - 1, -1, -1):
        rm = getattr(lst[i], "real_model", None)
        if rm is not None and callable(rm) and _real_model_alive(lst[i]) is None:
            lst.pop(i)


def _drop_unloaded(mm):
    """Drop entries left in the list after model_unload() (real_model is None).
    Only safe when no free_memory() is in progress, i.e. from the outermost guarded call."""
    lst = mm.current_loaded_models
    for i in range(len(lst) - 1, -1, -1):
        rm = getattr(lst[i], "real_model", None)
        if rm is None or not callable(rm):
            bad = lst.pop(i)
            name = "?"
            try:
                name = bad.model.model.__class__.__name__
            except Exception:
                pass
            logger.warning("Chat TE: removed a stale model-manager entry (%s) left by a concurrent load/unload", name)


def _wrap(mm, fn):
    @functools.wraps(fn)
    def guarded(*args, **kwargs):
        global _pending_cleanup
        with _LOCK:
            depth = getattr(_TLS, "depth", 0)
            _TLS.depth = depth + 1
            try:
                if depth == 0:
                    try:
                        _drop_unloaded(mm)
                        if _pending_cleanup:
                            _pending_cleanup = False
                            _cleanup_dead(mm)
                    except Exception:
                        logger.exception("Chat TE: model-manager repair failed")
                return fn(*args, **kwargs)
            finally:
                _TLS.depth = depth
    guarded._chat_te_guard = True
    return guarded


def install():
    """Patch comfy.model_management once. Safe to call multiple times."""
    global _installed
    if _installed:
        return
    try:
        import comfy.model_management as mm
    except Exception:
        logger.warning("Chat TE: comfy.model_management not importable; model-manager guard not installed")
        return

    missing = [n for n in _GUARDED + ("cleanup_models", "current_loaded_models", "LoadedModel") if not hasattr(mm, n)]
    if missing:
        logger.warning("Chat TE: model-manager guard not installed (ComfyUI internals changed: %s)", ", ".join(missing))
        return

    for name in _GUARDED:
        fn = getattr(mm, name)
        if not getattr(fn, "_chat_te_guard", False):
            setattr(mm, name, _wrap(mm, fn))

    # Finalizer callback (weakref.finalize(real_model, cleanup_models) in LoadedModel.model_load).
    # model_management looks the name up at registration time, so finalizers created after this
    # patch call the safe version. Never block here: finalizers run on arbitrary threads.
    def cleanup_models():
        global _pending_cleanup
        # Busy on another thread, or fired from GC in the middle of a guarded call on this thread
        # (e.g. inside free_memory(), whose pending pops use indices into the list): defer.
        if getattr(_TLS, "depth", 0) > 0 or not _LOCK.acquire(blocking=False):
            _pending_cleanup = True
            return
        try:
            _cleanup_dead(mm)
        except Exception:
            logger.exception("Chat TE: cleanup_models failed")
        finally:
            _LOCK.release()
    cleanup_models._chat_te_guard = True
    mm.cleanup_models = cleanup_models

    # is_dead() is called from cleanup_models_gc() / free_memory(); upstream raises if real_model is None
    def is_dead(self):
        return _real_model_alive(self) is not None and self.model is None
    mm.LoadedModel.is_dead = is_dead

    _installed = True
    logger.info("Chat TE: model-manager thread guard installed")
