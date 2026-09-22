"""Refresh stateless answer modules when a running desktop app is updated."""
import importlib
from pathlib import Path
import threading

_lock = threading.RLock()
_loaded = {}


def refresh_answer_modules(*modules):
    # A browser refresh reruns app.py, but Python may retain imported modules
    # when Streamlit's file watcher misses a change. Reload only changed code;
    # never clear the index, chat history, OCR locks, or model processes.
    with _lock:
        for module in modules:
            path = Path(module.__file__)
            signature = (path.stat().st_mtime_ns, path.stat().st_size)
            if _loaded.get(module.__name__) != signature:
                importlib.invalidate_caches()
                importlib.reload(module)
                _loaded[module.__name__] = signature
        return tuple((module.__name__, _loaded[module.__name__]) for module in modules)
