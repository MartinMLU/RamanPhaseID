# run_app_baseline.py – Startpunkt für PyInstaller / auto-py-to-exe
import os, sys, socket, time, webbrowser, threading
from pathlib import Path
from streamlit.web import cli as stcli

# ---- Optional: Backends & Daten, die PyInstaller sonst evtl. übersieht ----
try:
    import matplotlib.backends.backend_agg  # für st.pyplot
    import matplotlib.backends.backend_svg  # falls du später SVG exportierst
except Exception:
    pass

# ---- Konfiguration: Dateinamen deiner Streamlit-App ----
APP_CANDIDATES = [
    "baseline_app_01c.py",       # dein aktuelles File
    "baseline_app_*.py",     # Fallback: erstes passendes
]

def _free_port(start: int = 8501, max_tries: int = 10) -> int:
    for p in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("No free port found")

def _resource_dir() -> Path:
    # Bei --onefile entpackt PyInstaller hierhin
    return Path(getattr(sys, "_MEIPASS", Path(__file__).parent))

def _find_app() -> Path:
    base = _resource_dir()
    # 1) exakte Kandidaten
    for name in APP_CANDIDATES:
        if "*" not in name:
            if (base / name).exists(): return base / name
            if (base / "_internal" / name).exists(): return base / "_internal" / name
    # 2) Wildcards
    for pat in APP_CANDIDATES:
        for folder in (base, base / "_internal"):
            hits = sorted(folder.glob(pat))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"App script not bundled. Looked for: {APP_CANDIDATES}")

def _user_cache_dir(appname: str = "BaselineSubtractor") -> str:
    home = Path.home()
    for env in ("LOCALAPPDATA", "APPDATA"):
        if os.getenv(env):
            return str(Path(os.getenv(env)) / appname / "cache")
    return str(home / f".{appname}" / "cache")

def _open_browser_later(url: str, delay: float = 0.6):
    time.sleep(delay)
    try:
        webbrowser.open_new_tab(url)
    except Exception:
        pass

if __name__ == "__main__":
    app_path = _find_app()
    port = _free_port(8501, 10)

    # Streamlit-Umgebung setzen (robust & portfest)
    os.environ.update(
        STREAMLIT_GLOBAL_DEVELOPMENT_MODE="false",
        STREAMLIT_SERVER_HEADLESS="true",
        STREAMLIT_SERVER_PORT=str(port),
        STREAMLIT_SERVER_ENABLECORS="true",
        STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION="true",
        STREAMLIT_BROWSER_GATHER_USAGE_STATS="false",
        STREAMLIT_CACHE_DIR=_user_cache_dir(),  # schreibbarer Cache
    )
    # Optional, falls du die App mal im LAN teilen willst:
    # os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"

    # Optional: MKL/OpenMP-„meckern“ auf manchen PCs unterdrücken
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Streamlit aufrufen
    sys.argv = ["streamlit", "run", str(app_path)]

    # Genau EIN Browser-Tab via Thread öffnen (kein multiprocessing!)
    threading.Thread(
        target=_open_browser_later,
        args=(f"http://localhost:{port}", 0.6),
        daemon=True,
    ).start()

    sys.exit(stcli.main())
