"""Stop only servers listening from this project; --dry-run never sends signals."""
import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
LSOF = '/usr/sbin/lsof'


def output(args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or 'Unable to inspect local servers.')
    return result.stdout.strip()


def listeners(port):
    return {int(line) for line in output([LSOF, '-nP', f'-iTCP:{port}', '-sTCP:LISTEN', '-t']).splitlines() if line.isdigit()}


def belongs_to_project(pid, service):
    uid = output(['/bin/ps', '-p', str(pid), '-o', 'uid='])
    if uid != str(os.getuid()):
        return False
    cwd = output([LSOF, '-a', '-p', str(pid), '-d', 'cwd', '-Fn'])
    if not any(line.startswith('n') and Path(line[1:]).resolve() == ROOT for line in cwd.splitlines()):
        return False
    command = output(['/bin/ps', '-p', str(pid), '-o', 'command='])
    try:
        args = shlex.split(command)
    except ValueError:
        return False
    if service == 'Ollama':
        return len(args) == 2 and Path(args[0]).name == 'ollama' and args[1] == 'serve'
    return len(args) == 5 and args[1:4] == ['-m', 'streamlit', 'run'] and Path(args[4]).name == 'app.py'


def stop_service(service, port, dry_run=False):
    pids = listeners(port)
    if not pids:
        print(f'{service} is already stopped.')
        return True
    for pid in pids:
        if not belongs_to_project(pid, service):
            print(f'Leaving {service} on port {port} running: it was not started from this project.')
            if service == 'Research Desk':
                return False
            continue
        if dry_run:
            print(f'Would stop {service} (process {pid}).')
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        for _ in range(40):
            if pid not in listeners(port):
                print(f'Stopped {service}.')
                break
            time.sleep(0.25)
        else:
            print(f'{service} is still shutting down. Try Stop again shortly.')
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not stop_service('Research Desk', 8501, args.dry_run):
        return 1
    if not stop_service('Ollama', 11434, args.dry_run):
        return 1
    if not args.dry_run:
        print('Completed paper indexes and downloaded models are saved. Use Start Research Assistant.command to reopen.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
