# run_app.py - Launcher for PyInstaller / auto-py-to-exe (RamanPhaseID)
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

try:
    from streamlit.web import cli as stcli
except Exception:
    # Fallback for older Streamlit layouts
    from streamlit import cli as stcli  # type: ignore

# Optional imports so PyInstaller sees common matplotlib backends used by st.pyplot
try:
    import matplotlib.backends.backend_agg  # noqa: F401
    import matplotlib.backends.backend_svg  # noqa: F401
except Exception:
    pass

APP_CANDIDATES = [
    "RamanPhaseID_0p98beta.py",
    "RamanPhaseID*.py",
]


def _free_port(start: int = 8501, max_tries: int = 10) -> int:
    for port in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free port found")


def _resource_dir() -> Path:
    # PyInstaller onefile extracts to _MEIPASS.
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def _find_app() -> Path:
    base = _resource_dir()
    search_roots = (base, base / "_internal")

    # 1) exact matches first
    for cand in APP_CANDIDATES:
        if "*" in cand:
            continue
        for root in search_roots:
            p = root / cand
            if p.exists():
                return p

    # 2) wildcard fallbacks
    for cand in APP_CANDIDATES:
        if "*" not in cand:
            continue
        for root in search_roots:
            hits = sorted(root.glob(cand))
            if hits:
                return hits[0]

    raise FileNotFoundError(f"App script not bundled. Looked for: {APP_CANDIDATES}")


def _user_cache_dir(appname: str = "RamanPhaseID") -> str:
    # Ensure Streamlit cache is writable when running from bundled app.
    home = Path.home()
    for env in ("LOCALAPPDATA", "APPDATA"):
        val = os.getenv(env)
        if val:
            return str(Path(val) / appname / "cache")
    return str(home / f".{appname}" / "cache")


def _open_browser_later(url: str, delay: float = 0.6) -> None:
    time.sleep(delay)
    try:
        webbrowser.open_new_tab(url)
    except Exception:
        pass


if __name__ == "__main__":
    app_path = _find_app()
    port = _free_port(8501, 10)

    os.environ.update(
        STREAMLIT_GLOBAL_DEVELOPMENT_MODE="false",
        STREAMLIT_SERVER_HEADLESS="true",
        STREAMLIT_SERVER_PORT=str(port),
        STREAMLIT_SERVER_ENABLECORS="true",
        STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION="true",
        STREAMLIT_BROWSER_GATHER_USAGE_STATS="false",
        STREAMLIT_SERVER_FILE_WATCHER_TYPE="none",
        STREAMLIT_CACHE_DIR=_user_cache_dir(),
    )
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    sys.argv = ["streamlit", "run", str(app_path)]

    threading.Thread(
        target=_open_browser_later,
        args=(f"http://localhost:{port}", 0.6),
        daemon=True,
    ).start()

    sys.exit(stcli.main())
