#!/usr/bin/env python3
"""Start the published bot with the team's saved launch settings."""
import getpass
import os
from pathlib import Path

import bot

LAUNCH_ARGUMENTS = ['--live', '--allow-orders', '--inventory-mode', 'profit',
                    '--price-discount', '0.40']


def main(root=None):
    root = Path(root or Path(__file__).resolve().parent).resolve()
    runtime = root / '.nightshift'
    pid_path = runtime / 'live.pid'
    if pid_path.exists():
        try:
            previous_pid = int(pid_path.read_text().strip())
            if previous_pid <= 0:
                raise ValueError('Invalid process ID')
            if os.name != 'posix':
                raise SystemExit('A previous controller is recorded. Verify it stopped before removing .nightshift/live.pid.')
            os.kill(previous_pid, 0)
        except ProcessLookupError:
            pass
        except (ValueError, PermissionError):
            raise SystemExit('Verify the existing .nightshift/live.pid before starting another controller.')
        else:
            raise SystemExit('A controller already appears to be running on this computer. Do not start a second one.')
    key = os.environ.get('NIGHTSHIFT_KEY', '').strip()
    key_file = runtime / 'truck-key'
    if not key and key_file.exists():
        key = key_file.read_text().strip()
    if not key:
        key = getpass.getpass('Existing team truck key (hidden): ').strip()
    if not key:
        raise SystemExit('No key supplied; nothing was submitted.')
    old_key = os.environ.get('NIGHTSHIFT_KEY')
    old_directory = Path.cwd()
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ['NIGHTSHIFT_KEY'] = key
    pid_path.write_text(str(os.getpid()) + '\n')
    try:
        os.chdir(root)
        return bot.main(LAUNCH_ARGUMENTS)
    finally:
        os.chdir(old_directory)
        if old_key is None:
            os.environ.pop('NIGHTSHIFT_KEY', None)
        else:
            os.environ['NIGHTSHIFT_KEY'] = old_key
        if pid_path.exists() and pid_path.read_text().strip() == str(os.getpid()):
            pid_path.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
