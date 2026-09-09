from pathlib import Path
import subprocess
import socket
import time
import webbrowser
import sys

def wait_port(host: str, port: int, timeout_sec: float = 25.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False

def get_base_dir() -> Path:
    # onefile exe: sys.executable == exe 경로
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # dev: launcher.py가 있는 폴더
    return Path(__file__).resolve().parent

def main():
    base = get_base_dir()

    py = base / "runtime" / "Scripts" / "python.exe"
    app = base / "app.py"

    if not py.exists():
        raise FileNotFoundError(f"runtime python not found: {py}")
    if not app.exists():
        raise FileNotFoundError(f"app.py not found: {app}")

    host, port = "127.0.0.1", 8501
    url = f"http://{host}:{port}"

    cmd = [
        str(py), "-m", "streamlit", "run", str(app),
        "--server.address", host,
        "--server.port", str(port),
        "--server.headless", "true",
    ]

    creationflags = 0
    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
    if hasattr(subprocess, "DETACHED_PROCESS"):
        creationflags |= subprocess.DETACHED_PROCESS

    subprocess.Popen(
        cmd,
        cwd=str(base),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    if wait_port(host, port, timeout_sec=25.0):
        webbrowser.open(url)

if __name__ == "__main__":
    main()
