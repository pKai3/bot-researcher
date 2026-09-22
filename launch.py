"""Keep Research Desk attached to Terminal until Ctrl+C or window closure."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser

from stop import belongs_to_project, listeners, stop_service

ROOT = Path(__file__).resolve().parent
DATA = ROOT / '.data'
URL = 'http://127.0.0.1:8501'
OLLAMA_URL = 'http://127.0.0.1:11434/api/tags'


def available(url):
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def wait_until_ready(url, process):
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError('The server exited during startup. See the output above or .data/ollama.log.')
        if available(url):
            return
        time.sleep(0.5)
    raise RuntimeError('The server did not become ready. See the output above or .data/ollama.log.')


def stop_child(process, name):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f'{name} is still shutting down. Use Stop Research Assistant.command if needed.', flush=True)


def interrupted(signum, frame):
    raise KeyboardInterrupt


def main():
    DATA.mkdir(exist_ok=True)
    os.chdir(ROOT)
    app = ollama = None
    attached_app = attached_ollama = False
    log = None
    previous_handlers = {sig: signal.signal(sig, interrupted)
                         for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        # Older launches may already have a detached server. Take control of that
        # instance without restarting an in-progress indexing job.
        existing = listeners(8501)
        if existing:
            if not all(belongs_to_project(pid, 'Research Desk') for pid in existing):
                raise RuntimeError('Port 8501 belongs to another app. Close that app before launching Research Desk.')
            attached_app = True

        if available(OLLAMA_URL):
            existing_ollama = listeners(11434)
            attached_ollama = bool(existing_ollama) and all(
                belongs_to_project(pid, 'Ollama') for pid in existing_ollama)
        else:
            executable = shutil.which('ollama') or '/opt/homebrew/bin/ollama'
            if not Path(executable).exists():
                raise RuntimeError('Ollama is missing. See README.md for installation instructions.')
            with (DATA / 'ollama.log').open('ab') as ollama_log:
                ollama = subprocess.Popen(
                    [executable, 'serve'], stdout=ollama_log, stderr=ollama_log,
                    cwd=ROOT, env={**os.environ, 'OLLAMA_HOST': '127.0.0.1:11434',
                                   'OLLAMA_NO_CLOUD': '1', 'OLLAMA_NUM_PARALLEL': '1'})
            wait_until_ready(OLLAMA_URL, ollama)

        if attached_app:
            print('Connected to the already-running Research Desk.', flush=True)
            if (DATA / 'app.log').exists():
                log = (DATA / 'app.log').open('r', errors='replace')
                log.seek(0, os.SEEK_END)
        else:
            # Inherit this Terminal's output and process group. Keep the parent
            # alive so it also shuts down the Ollama instance it started.
            app = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app.py'], cwd=ROOT)
            wait_until_ready(URL + '/_stcore/health', app)

        print(f'\nResearch Desk: {URL}\nKeep this Terminal window open. Press Ctrl+C here to shut down.\n', flush=True)
        webbrowser.open(URL)
        while app.poll() is None if app is not None else bool(listeners(8501) & existing):
            if log:
                text = log.read()
                if text:
                    print(text, end='', flush=True)
            time.sleep(0.5)
        return app.returncode if app is not None else 0
    except KeyboardInterrupt:
        print('\nShutting down Research Desk…', flush=True)
        return 0
    except (RuntimeError, OSError) as exc:
        print(f'Could not launch Research Desk: {exc}', file=sys.stderr, flush=True)
        return 1
    finally:
        # Finish graceful cleanup even if Ctrl+C is pressed a second time.
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if log:
                log.close()
            stop_child(app, 'Research Desk')
            if attached_app:
                stop_service('Research Desk', 8501)
            stop_child(ollama, 'Ollama')
            if attached_ollama:
                stop_service('Ollama', 11434)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        print('Launcher finished. Completed paper indexes and downloaded models are saved.', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
